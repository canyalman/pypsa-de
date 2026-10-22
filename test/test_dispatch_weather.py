# SPDX-License-Identifier: MIT

"""Fixed-fleet weather operation, profile transfer and bounded shortage diagnostics."""

import ast
import copy
import json
import runpy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytest
from pypsa.descriptors import nominal_attrs
from snakemake.parser import parse
from snakemake.sourcecache import LocalSourceFile

from scripts import _helpers
from scripts.dispatch_weather import (
    MARKER,
    SHEDDING,
    add_demand_shedding,
    build_dispatch_weather,
    dispatch_result_summary,
    fix_reference_fleet,
    validate_dispatch_weather,
)
from test import test_controlled_weather
from test.test_controlled_weather import PRICE, ROOT, SCENARIO
from test.test_fixed_neighbor_constraints import MODULE


@pytest.fixture
def effective_config():
    return test_controlled_weather.effective_config.__wrapped__()


def fixture_pair(year=1996):
    pair = []
    for weather_year in [2013, year]:
        n = pypsa.Network()
        n.set_snapshots(pd.date_range(f"{weather_year}-01-01", periods=3, freq="3h"))
        n.snapshot_weightings.loc[:, :] = 3.0
        for carrier in ["AC", "solar", "OCGT", "heat", "heat pump", "battery", "co2"]:
            n.add("Carrier", carrier)
        n.add("Bus", ["DE0", "FR0"], carrier="AC", country=["DE", "FR"])
        n.add("Bus", "DE0 heat", carrier="heat", country="DE")
        n.add("Bus", "co2 atmosphere", carrier="co2")
        n.add(
            "Generator",
            "DE0 solar-2035",
            bus="DE0",
            carrier="solar",
            p_nom_extendable=True,
            p_nom_opt=4,
            capital_cost=2,
            p_max_pu=[0.1, 0.2, 0.3],
        )
        n.add(
            "Generator",
            "DE0 OCGT-2035",
            bus="DE0",
            carrier="OCGT",
            p_nom_extendable=True,
            p_nom_opt=5,
            capital_cost=3,
            marginal_cost=80,
        )
        n.add("Load", "DE0", bus="DE0", p_set=[8, 8, 8])
        n.add("Load", "DE0 heat", bus="DE0 heat", p_set=[2, 2, 2])
        n.add(
            "Link",
            "DE0 heat pump-2035",
            bus0="DE0",
            bus1="DE0 heat",
            carrier="heat pump",
            p_nom_extendable=True,
            p_nom_opt=2,
            efficiency=[3, 3, 3],
            capital_cost=1,
        )
        n.add(
            "Store",
            "DE0 battery-2035",
            bus="DE0",
            carrier="battery",
            e_cyclic=True,
            e_nom_extendable=True,
            e_nom_opt=4,
            capital_cost=2,
        )
        n.add("Store", "co2 atmosphere", bus="co2 atmosphere", e_nom=1e9, e_min_pu=-1)
        n.add("GlobalConstraint", "CO2Limit", mu=-PRICE, constant=100)
        n.meta = {"wildcards": {"planning_horizons": "2035"}}
        pair.append(n)
    reference, weather = pair
    weather.generators_t.p_max_pu["DE0 solar-2035"] = [0.4, 0.5, 0.6]
    weather.links_t.efficiency["DE0 heat pump-2035"] = [2, 2, 2]
    weather.loads_t.p_set["DE0 heat"] = [2, 3, 4]
    reference.generators_t.p.loc[:, "DE0 OCGT-2035"] = [1, 2, 3]
    reference.stores_t.e.loc[:, "DE0 battery-2035"] = [4, 4, 4]
    return reference, weather


def settings():
    return {
        "reference_network": "reference_2035.nc",
        "frozen_generator_carriers": ["solar"],
        "dispatch_unserved_energy": True,
        "dispatch_unserved_energy_cost": 100000.0,
        "resize_transport_oil_delivery": True,
    }


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_dispatch_preserves_complete_fleet_and_ages_but_replaces_weather(year):
    reference, weather = fixture_pair(year)
    weather.generators.at["DE0 OCGT-2035", "p_nom_opt"] = 999
    before = reference.links.copy()
    n, ledger = build_dispatch_weather(reference, weather, settings())
    assert all(
        not n.components[c].static[f"{attr}_extendable"].any()
        for c, attr in nominal_attrs.items()
    )
    assert n.generators.at["DE0 solar-2035", "p_nom"] == 4
    assert n.generators.at["DE0 OCGT-2035", "p_nom"] == 5
    assert n.stores.at["DE0 battery-2035", "e_nom"] == 4
    assert n.stores.at["DE0 battery-2035", "e_cyclic"]
    np.testing.assert_array_equal(
        n.generators_t.p_max_pu["DE0 solar-2035"], [0.4, 0.5, 0.6]
    )
    np.testing.assert_array_equal(n.links_t.efficiency["DE0 heat pump-2035"], [2, 2, 2])
    np.testing.assert_array_equal(n.loads_t.p_set["DE0 heat"], [2, 3, 4])
    assert n.generators_t.p.empty and n.stores_t.e.empty
    pd.testing.assert_frame_equal(before, reference.links)
    for component in ["Generator", "Link", "Store"]:
        old, new = (
            reference.components[component].static,
            n.components[component].static,
        )
        for column in ["capital_cost", "build_year", "lifetime"]:
            pd.testing.assert_series_equal(old[column], new.loc[old.index, column])
    assert ledger.annualized_capex_eur.sum() == 33


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_dispatch_discards_auxiliary_results_without_changing_reference(year):
    reference, weather = fixture_pair(year)
    for component, attribute in [("Link", "p-incoming-DE-aux-pos"), ("Line", "mu_pos")]:
        reference.components[component].dynamic[attribute] = pd.DataFrame(
            1.0, index=reference.snapshots, columns=["old auxiliary result"]
        )
    n, ledger = build_dispatch_weather(reference, weather, settings())
    assert n.links_t["p-incoming-DE-aux-pos"].empty
    assert n.lines_t["mu_pos"].empty
    assert not reference.links_t["p-incoming-DE-aux-pos"].empty
    assert n.generators.at["DE0 OCGT-2035", "p_nom"] == 5
    assert ledger.annualized_capex_eur.sum() == 33


def test_all_six_nominal_component_groups_are_frozen():
    n, _ = fixture_pair()
    n.add(
        "Line",
        "line",
        bus0="DE0",
        bus1="FR0",
        x=0.1,
        s_nom_extendable=True,
        s_nom_opt=17,
    )
    n.add(
        "Transformer",
        "transformer",
        bus0="DE0",
        bus1="FR0",
        x=0.1,
        s_nom_extendable=True,
        s_nom_opt=3,
    )
    n.add("StorageUnit", "hydro", bus="DE0", p_nom_extendable=True, p_nom_opt=2)
    ledger = fix_reference_fleet(n)
    assert set(ledger.component) == set(nominal_attrs)
    assert n.lines.at["line", "s_nom"] == 17
    assert n.transformers.at["transformer", "s_nom"] == 3
    assert n.storage_units.at["hydro", "p_nom"] == 2


@pytest.mark.parametrize("value,accepted", [(-0.012, True), (-0.06, False)])
def test_reference_negative_solver_residuals_have_a_strict_limit(value, accepted):
    n, _ = fixture_pair()
    n.generators.at["DE0 solar-2035", "p_nom_opt"] = value
    if accepted:
        ledger = fix_reference_fleet(n)
        assert n.generators.at["DE0 solar-2035", "p_nom"] == 0
        assert (
            ledger.set_index("asset").at["DE0 solar-2035", "reference_nominal"] == value
        )
    else:
        with pytest.raises(ValueError, match="Invalid solved Generator"):
            fix_reference_fleet(n)


def test_profile_node_locations_must_match_the_reference():
    reference, weather = fixture_pair()
    weather.buses.at["DE0", "x"] += 0.1
    with pytest.raises(ValueError, match="bus locations differ"):
        build_dispatch_weather(reference, weather, settings())


@pytest.mark.parametrize(
    "problem", ["missing_profile", "wrong_calendar", "wrong_weights", "wrong_horizon"]
)
def test_incompatible_weather_inputs_fail_before_solve(problem):
    reference, weather = fixture_pair()
    if problem == "missing_profile":
        weather.remove("Generator", "DE0 solar-2035")
    elif problem == "wrong_calendar":
        weather.set_snapshots(weather.snapshots[:-1])
    elif problem == "wrong_weights":
        weather.snapshot_weightings.loc[:, :] = 1
    else:
        reference.meta["wildcards"]["planning_horizons"] = "2030"
    with pytest.raises(ValueError):
        build_dispatch_weather(reference, weather, settings())


def test_unserved_demand_is_bounded_and_not_fuel_or_carbon_supply():
    n, _ = fixture_pair()
    n.add("Load", "injection", bus="FR0", p_set=[-5, -5, -5])
    add_demand_shedding(n, 100000)
    assets = n.generators.index[n.generators.carrier.eq(SHEDDING)]
    assert set(n.generators.loc[assets, "bus"]) == {"DE0", "DE0 heat"}
    capacities = n.generators.loc[assets, "p_nom"]
    maximum = n.get_switchable_as_dense("Generator", "p_max_pu", inds=assets).mul(
        capacities
    )
    np.testing.assert_array_equal(maximum.sum(axis=1), [10, 10, 10])


def test_dispatch_has_no_investment_variables_and_reports_shortages():
    reference, weather = fixture_pair()
    n, _ = build_dispatch_weather(reference, weather, settings())
    n.optimize.create_model()
    validate_dispatch_weather(n, model=True)
    status, condition = n.optimize.solve_model(
        solver_name="highs", log_to_console=False
    )
    assert (status, condition) == ("ok", "optimal")
    result = dispatch_result_summary(n)
    assert result["unserved_energy_mwh_by_country_carrier"]["DE|AC"] > 0
    assert result["unserved_energy_penalty_eur"] > 0
    assert result["objective_including_reference_capex_eur"] == pytest.approx(
        n.objective + 33
    )
    assert not n.links.p_nom_extendable.any()


def test_validation_rejects_a_reopened_capacity():
    reference, weather = fixture_pair()
    n, _ = build_dispatch_weather(reference, weather, settings())
    n.links.at["DE0 heat pump-2035", "p_nom_extendable"] = True
    with pytest.raises(ValueError, match="extendable Link"):
        validate_dispatch_weather(n)


def test_validation_rejects_changed_fixed_capacity():
    reference, weather = fixture_pair()
    n, _ = build_dispatch_weather(reference, weather, settings())
    n.generators.at["DE0 OCGT-2035", "p_nom"] += 0.1
    with pytest.raises(ValueError, match="capacity or location changed"):
        validate_dispatch_weather(n)


def test_config_is_dispatch_only_with_unchanged_reference_and_carbon_price(
    effective_config,
):
    c = effective_config
    year = pd.Timestamp(c["snapshots"]["start"]).year
    assert c["run"]["prefix"] == f"thesis_weather_dispatch_{year}"
    assert c["controlled_weather"]["mode"] == "dispatch"
    assert not c["solving"]["constraints"]["fixed_neighbor_capacities"]["enable"]
    assert not c["solving"]["options"]["noisy_costs"]
    assert not c["solving"]["options"]["load_shedding"]["enable"]
    assert c["controlled_weather"]["dispatch_unserved_energy_cost"] == 100000
    assert c["controlled_weather"]["resize_transport_oil_delivery"] is True
    assert c["sector"]["bev_dsm_availability"] == {2030: 0.2, 2035: 0.35}
    assert c["costs"]["emission_prices"]["co2"][2035] == PRICE


def test_actual_dispatch_preflight_runs_without_creating_model(
    tmp_path, monkeypatch, effective_config
):
    reference, weather = fixture_pair()
    n, _ = build_dispatch_weather(reference, weather, settings())
    path = tmp_path / "weather.nc"
    n.export_to_netcdf(path)
    config = copy.deepcopy(effective_config)
    report = tmp_path / "audit.json"
    workflow = SimpleNamespace(
        config=config,
        input=SimpleNamespace(network=str(path)),
        params=SimpleNamespace(solving=config["solving"]),
        wildcards=SimpleNamespace(
            planning_horizons="2035", clusters="27", run=SCENARIO
        ),
        output=[str(report)],
    )
    monkeypatch.setattr(_helpers, "configure_logging", lambda sm: None)
    monkeypatch.setattr(_helpers, "set_scenario_config", lambda sm: None)
    runpy.run_path(
        str(ROOT / "scripts/pypsa-de/check_controlled_weather.py"),
        run_name="__main__",
        init_globals={"snakemake": workflow},
    )
    result = json.loads(report.read_text(encoding="utf8"))
    assert result["status"] == "passed_without_solve"
    assert result["dispatch_weather"]["fixed_asset_count"] > 0
    assert "fixed_neighbor" not in result


def test_fixed_new_boilers_do_not_gain_an_existing_boiler_profile_rule():
    n, _ = fixture_pair()
    n.add(
        "Link",
        "new boiler",
        bus0="DE0",
        bus1="DE0 heat",
        carrier="gas boiler",
        p_nom=2,
        p_nom_extendable=False,
        weather_dispatch_was_extendable=True,
    )
    n.links[MARKER] = n.links[MARKER].fillna(False).astype(bool)
    n.links.at["new boiler", MARKER] = True
    # No model is needed: the newly fixed boiler must be excluded before model access.
    MODULE.force_boiler_profiles_existing_per_boiler(n)


def test_hydro_inflows_and_pit_weather_inputs_change_without_resizing():
    reference, weather = fixture_pair()
    for n in [reference, weather]:
        n.add("Carrier", ["hydro", "water pits"])
        n.add(
            "StorageUnit",
            "hydro",
            bus="DE0",
            carrier="hydro",
            p_nom=2,
            inflow=[0.1, 0.1, 0.1],
        )
        n.add(
            "Store",
            "DE0 water pits-2035",
            bus="DE0 heat",
            carrier="water pits",
            e_nom=8,
            e_cyclic=True,
            e_max_pu=[0.7, 0.7, 0.7],
        )
    weather.storage_units_t.inflow["hydro"] = [0.3, 0.4, 0.5]
    weather.stores_t.e_max_pu["DE0 water pits-2035"] = [0.5, 0.6, 0.7]
    n, _ = build_dispatch_weather(reference, weather, settings())
    np.testing.assert_allclose(n.storage_units_t.inflow["hydro"], [0.3, 0.4, 0.5])
    np.testing.assert_allclose(
        n.stores_t.e_max_pu["DE0 water pits-2035"], [0.5, 0.6, 0.7]
    )
    assert n.stores.at["DE0 water pits-2035", "e_nom"] == 8
    assert n.storage_units.at["hydro", "p_nom"] == 2


def test_ev_and_heat_source_calendar_profiles_are_not_left_in_2013():
    reference, weather = fixture_pair()
    for n in [reference, weather]:
        n.add("Carrier", ["EV battery", "urban central river heat"])
        n.add(
            "Store",
            "DE0 EV battery",
            bus="DE0",
            carrier="EV battery",
            e_nom=7,
            e_cyclic=True,
            e_min_pu=[0.1, 0.1, 0.1],
        )
        n.add(
            "Generator",
            "river heat",
            bus="DE0 heat",
            carrier="urban central river heat",
            p_nom=2,
            p_max_pu=[0.2, 0.2, 0.2],
        )
        n.add(
            "Link",
            "direct heat",
            bus0="DE0",
            bus1="DE0 heat",
            carrier="heat direct utilisation",
            p_nom=1,
            efficiency=[1, 1, 1],
        )
    weather.stores_t.e_min_pu["DE0 EV battery"] = [0.2, 0.3, 0.4]
    weather.generators_t.p_max_pu["river heat"] = [0.5, 0.6, 0.7]
    weather.links_t.efficiency["direct heat"] = [0, 1, 0]
    n, _ = build_dispatch_weather(reference, weather, settings())
    np.testing.assert_array_equal(
        n.stores_t.e_min_pu["DE0 EV battery"], [0.2, 0.3, 0.4]
    )
    np.testing.assert_array_equal(
        n.generators_t.p_max_pu["river heat"], [0.5, 0.6, 0.7]
    )
    np.testing.assert_array_equal(n.links_t.efficiency["direct heat"], [0, 1, 0])
    assert n.stores.at["DE0 EV battery", "e_nom"] == 7


def test_actual_preparation_script_exports_only_fixed_reference_capacities(
    tmp_path, monkeypatch, effective_config
):
    reference, weather = fixture_pair()
    ref_path, weather_path = tmp_path / "reference_2035.nc", tmp_path / "weather.nc"
    reference.export_to_netcdf(ref_path)
    weather.export_to_netcdf(weather_path)
    output, ledger_path = tmp_path / "dispatch.nc", tmp_path / "ledger.csv"
    workflow = SimpleNamespace(
        config=effective_config,
        input=SimpleNamespace(reference=str(ref_path), weather=str(weather_path)),
        params=SimpleNamespace(settings=settings()),
        wildcards={"planning_horizons": "2035"},
        output=SimpleNamespace(network=str(output), ledger=str(ledger_path)),
    )
    monkeypatch.setattr(_helpers, "configure_logging", lambda sm: None)
    monkeypatch.setattr(_helpers, "set_scenario_config", lambda sm: None)
    runpy.run_path(
        str(ROOT / "scripts/pypsa-de/prepare_dispatch_weather.py"),
        run_name="__main__",
        init_globals={"snakemake": workflow},
    )
    n = pypsa.Network(output)
    validate_dispatch_weather(n)
    assert n.links.at["DE0 heat pump-2035", MARKER]
    assert n.generators_t.p.empty and n.stores_t.e.empty
    assert pd.read_csv(ledger_path).annualized_capex_eur.sum() == 33


def test_workflow_separates_weather_preparation_from_fixed_fleet_solve(
    effective_config,
):
    from functools import lru_cache, partial

    from snakemake.io import Wildcards
    from snakemake.utils import update_config

    namespace = {
        "config": effective_config,
        "scenarios": {SCENARIO: {}},
        "copy": copy,
        "partial": partial,
        "lru_cache": lru_cache,
        "update_config": update_config,
        "update_config_from_wildcards": _helpers.update_config_from_wildcards,
        "resources": lambda path: "resources/test/{run}/" + path,
    }
    selected = {
        "get_config",
        "merge_configs",
        "scenario_config",
        "static_getter",
        "dynamic_getter",
        "config_provider",
        "solved_previous_horizon",
        "weather_solve_input",
    }
    for filename in [
        "common.smk",
        "solve_myopic.smk",
        "pypsa-de/controlled_weather.smk",
    ]:
        path = ROOT / "rules" / filename
        cache = SimpleNamespace(
            open=lambda p: open(p.get_path_or_uri(secret_free=True), encoding="utf8")
        )
        compiled, _ = parse(
            LocalSourceFile(str(path)), SimpleNamespace(sourcecache=cache), {}
        )
        tree = ast.parse(compiled)
        body = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in selected
        ]
        exec(
            compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    w = Wildcards(
        fromdict={"run": SCENARIO, "clusters": "27", "planning_horizons": "2035"}
    )
    assert namespace["weather_solve_input"](w).endswith("_dispatch.nc")
    assert namespace["solved_previous_horizon"](w).endswith("_2030.nc")
    source = (ROOT / "rules/pypsa-de/controlled_weather.smk").read_text(encoding="utf8")
    preparation = source.split("rule prepare_dispatch_weather_network:", 1)[1].split(
        "rule controlled_weather_preflight:", 1
    )[0]
    assert (
        "_final.nc" in preparation
        and 'controlled_settings["reference_network"]' in preparation
    )
