# SPDX-License-Identifier: MIT

"""Common-history weather regressions without downloads or large solves."""

import ast
import copy
import json
import runpy
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytest
import xarray as xr
import yaml
from pydantic import ValidationError
from snakemake.io import Wildcards
from snakemake.parser import parse
from snakemake.sourcecache import LocalSourceFile
from snakemake.utils import update_config

from scripts import _helpers
from scripts.add_brownfield import add_brownfield, adjust_renewable_profiles
from scripts.build_electricity_demand import align_load_to_snapshots
from scripts.controlled_weather import (
    align_calendar,
    cached_weather_resources,
    freeze_domestic_renewables,
    refresh_brownfield_weather,
    validate_input_series,
)
from scripts.lib.validation.config import validate_config
from scripts.lib.validation.config.controlled_weather import ControlledWeatherConfig

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "KN2045_Mix_FixedRenewables"
PRICE = 308.2489253762274


def calendar(year):
    return _helpers.get_snapshots(
        {"start": f"{year}-01-01", "end": f"{year + 1}-01-01", "inclusive": "left"},
        True,
    )[::3]


@pytest.fixture
def effective_config():
    config = {}
    for filename in ["config.default.yaml", "config.de.yaml"]:
        update_config(
            config,
            yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8")),
        )
    manual = yaml.safe_load(
        (ROOT / "config/scenarios.manual.yaml").read_text(encoding="utf-8")
    )
    generated = {SCENARIO: copy.deepcopy(manual[SCENARIO])}
    builder = runpy.run_path(str(ROOT / "scripts/pypsa-de/build_scenarios.py"))
    builder["write_weather_dependent_config"](
        generated, SCENARIO, manual[SCENARIO]["weather_year"]
    )
    update_config(config, generated[SCENARIO])
    validate_config(config)
    return config


def test_controlled_config_preserves_the_reference_assumptions(effective_config):
    c = effective_config
    year = pd.Timestamp(c["snapshots"]["start"]).year
    assert year in [1996, 2010, 2023]
    assert c["run"]["name"] == [SCENARIO]
    dispatch = c["controlled_weather"].get("mode") == "dispatch"
    label = "dispatch" if dispatch else "controlled"
    assert c["run"]["prefix"] == f"thesis_weather_{label}_{year}"
    assert c["run"]["shared_resources"]["policy"] is False
    assert c["scenario"]["planning_horizons"] == [2025, 2030, 2035]
    assert c["clustering"]["temporal"]["resolution_sector"] == "3H"
    assert len(calendar(year)) == 2920
    assert c["sector"]["bev_dsm_availability"] == {2030: 0.2, 2035: 0.35}
    assert not c["sector"]["v2g"]
    assert c["load"]["fixed_year"] == 2013
    assert c["conventional"]["nuclear_availability_year"] == 2013
    assert (
        "battery_storage_duration" not in c
    )  # preserve the endogenous reference formulation
    assert c["costs"]["emission_prices"]["co2"][2035] == PRICE
    limits = c["solving"]["constraints"]
    assert limits["fixed_co2_price"] == {"enable": True, "planning_horizons": [2035]}
    assert limits["fixed_neighbor_capacities"]["formulation"] == "static"
    assert limits["fixed_neighbor_capacities"]["enable"] is not dispatch
    for component, carrier in [("Store", "battery"), ("Link", "battery discharger")]:
        for bound in ["limits_capacity_min", "limits_capacity_max"]:
            assert 2035 not in limits[bound].get(component, {}).get(carrier, {}).get(
                "DE", {}
            )
    control = c["controlled_weather"]
    fixed = limits["fixed_neighbor_capacities"]
    assert control["previous_network"] == fixed["reference_networks"][2030]
    assert control["reference_network"] == fixed["reference_networks"][2035]
    assert "fixedgas_2026-09-14/" in control["reference_network"]
    assert (
        control["weather_resources_directory"]
        == f"resources/thesis_weather_{year}/{SCENARIO}"
    )
    cutout = f"europe-{year}-sarah3-era5"
    assert c["atlite"]["default_cutout"] == cutout
    for tech in [
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
        "solar",
        "solar-hsat",
        "hydro",
    ]:
        assert c["renewable"][tech]["cutout"] == cutout
    assert c["data"]["cutout"]["source"] == "archive"


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_month_day_alignment_handles_leap_years_and_reordered_rows(year):
    source = pd.DataFrame({"profile": np.arange(2920)}, index=calendar(2013))
    result = align_calendar(source.iloc[::-1], calendar(year))
    assert result.index.equals(calendar(year))
    np.testing.assert_array_equal(result.profile, np.arange(2920))
    assert not ((result.index.month == 2) & (result.index.day == 29)).any()


@pytest.mark.parametrize("failure", ["leap", "missing", "duplicate", "nan"])
def test_bad_calendars_fail_before_solving(failure):
    source = pd.DataFrame({"value": 1.0}, index=calendar(2013))
    target = calendar(1996)
    if failure == "leap":
        target = pd.date_range("1996-01-01", "1997-01-01", freq="3h", inclusive="left")
    elif failure == "missing":
        source = source.iloc[1:]
    elif failure == "duplicate":
        source = pd.concat([source, source.iloc[[0]]])
    else:
        source.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        align_calendar(source, target)


def network(year):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range(f"{year}-01-01", periods=3, freq="3h"))
    n.snapshot_weightings.loc[:, :] = 3.0
    n.add("Bus", ["DE0", "FR0"], country=["DE", "FR"], carrier="AC")
    n.add("Carrier", ["solar", "rural air heat pump", "battery", "OCGT"])
    return n


def settings():
    return {
        "previous_horizon": 2030,
        "planning_horizon": 2035,
        "domestic_country": "DE",
        "frozen_generator_carriers": ["solar"],
    }


def history_and_weather(year):
    previous = network(2013)
    previous.meta = {"wildcards": {"planning_horizons": "2030"}}
    previous.add(
        "Generator",
        "DE0 solar-2030",
        bus="DE0",
        carrier="solar",
        build_year=2030,
        lifetime=25,
        p_nom_opt=10,
        p_max_pu=[0.1, 0.2, 0.3],
    )
    previous.add(
        "Link",
        "DE0 rural air heat pump-2030",
        bus0="DE0",
        bus1="DE0",
        carrier="rural air heat pump",
        build_year=2030,
        lifetime=20,
        p_nom_opt=5,
        efficiency=[2, 2, 2],
    )
    previous.add(
        "Store",
        "DE0 battery-2030",
        bus="DE0",
        carrier="battery",
        build_year=2030,
        lifetime=20,
        e_nom_opt=40,
    )
    previous.generators_t.p = pd.DataFrame(
        999.0, index=previous.snapshots, columns=previous.generators.index
    )
    previous.stores_t.e = pd.DataFrame(
        999.0, index=previous.snapshots, columns=previous.stores.index
    )
    current = network(year)
    profiles = {}
    for vintage in [2025, 2030]:
        n = network(year)
        n.add(
            "Generator",
            "DE0 solar",
            bus="DE0",
            carrier="solar",
            p_max_pu=[0.7, 0.8, 0.9],
        )
        n.add(
            "Link",
            "DE0 rural air heat pump",
            bus0="DE0",
            bus1="DE0",
            carrier="rural air heat pump",
            efficiency=[3, 4, 5],
        )
        profiles[vintage] = n
    return previous, current, profiles


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_native_brownfield_uses_common_capacities_but_new_weather(year):
    previous, current, profiles = history_and_weather(year)
    old = previous.generators.copy()
    refresh_brownfield_weather(previous, current, profiles, settings())
    pd.testing.assert_frame_equal(previous.generators, old)
    assert previous.generators_t.p.empty and previous.stores_t.e.empty
    assert previous.generators_t.p_max_pu.iloc[:, 0].tolist() == [0.7, 0.8, 0.9]
    assert previous.links_t.efficiency.iloc[:, 0].tolist() == [3, 4, 5]
    add_brownfield(current, previous, 2035, capacity_threshold=0.1)
    assert current.generators.at["DE0 solar-2030", "p_nom"] == 10
    assert not current.generators.at["DE0 solar-2030", "p_nom_extendable"]
    assert current.generators.at["DE0 solar-2030", "build_year"] == 2030
    assert current.generators.at["DE0 solar-2030", "lifetime"] == 25
    assert current.stores.at["DE0 battery-2030", "e_nom"] == 40
    assert current.generators_t.p.empty and current.stores_t.e.empty
    validate_input_series(current)


def test_missing_vintage_profile_fails_instead_of_reusing_2013_weather():
    previous, current, profiles = history_and_weather(1996)
    profiles[2030].remove("Generator", "DE0 solar")
    with pytest.raises(ValueError, match="Missing fresh"):
        refresh_brownfield_weather(previous, current, profiles, settings())


def test_snapshot_weights_are_not_silently_changed():
    previous, current, profiles = history_and_weather(2010)
    current.snapshot_weightings.loc[:, :] = 1
    with pytest.raises(ValueError, match="weights differ"):
        refresh_brownfield_weather(previous, current, profiles, settings())


def fleet_pair():
    current, reference = network(2010), network(2013)
    for n in [current, reference]:
        n.add(
            "Generator",
            "DE0 solar-2035",
            bus="DE0",
            carrier="solar",
            p_nom_extendable=True,
            p_nom_max=100,
            p_nom_opt=30,
            capital_cost=2,
        )
        n.add(
            "Generator",
            "FR0 solar-2035",
            bus="FR0",
            carrier="solar",
            p_nom_extendable=True,
            p_nom_opt=50,
        )
        n.add(
            "Link",
            "DE0 OCGT-2035",
            bus0="DE0",
            bus1="DE0",
            carrier="OCGT",
            p_nom_extendable=True,
        )
        n.add(
            "Store",
            "DE0 battery-2035",
            bus="DE0",
            carrier="battery",
            e_nom_extendable=True,
        )
    return current, reference


def test_domestic_fleet_fix_preserves_endogenous_battery_and_backup():
    n, reference = fleet_pair()
    n.generators.at["DE0 solar-2035", "capital_cost"] = 99.0
    ledger = freeze_domestic_renewables(n, reference, settings(), 0.05)
    assert n.generators.at["DE0 solar-2035", "p_nom"] == 30
    assert not n.generators.at["DE0 solar-2035", "p_nom_extendable"]
    assert n.generators.at[
        "FR0 solar-2035", "p_nom_extendable"
    ]  # handled by the existing non-DE materializer
    assert n.links.at["DE0 OCGT-2035", "p_nom_extendable"]
    assert n.stores.at["DE0 battery-2035", "e_nom_extendable"]
    assert ledger.removed_annualized_capex_eur.sum() == 60
    assert n.generators.at["DE0 solar-2035", "capital_cost"] == 2.0


@pytest.mark.parametrize("failure", ["missing", "bounds", "location"])
def test_fleet_mismatch_is_strict(failure):
    n, reference = fleet_pair()
    if failure == "missing":
        n.remove("Generator", "DE0 solar-2035")
    elif failure == "bounds":
        n.generators.at["DE0 solar-2035", "p_nom_max"] = 20
    else:
        n.generators.at["DE0 solar-2035", "bus"] = "FR0"
    with pytest.raises(ValueError):
        freeze_domestic_renewables(n, reference, settings(), 0.05)


def test_weather_cache_never_reuses_networks(tmp_path):
    (tmp_path / "profile_27_solar.nc").touch()
    (tmp_path / "cop_profiles_2030.nc").touch()
    (tmp_path / "networks").mkdir()
    (tmp_path / "networks/solved.nc").touch()
    assert cached_weather_resources(tmp_path) == [
        "cop_profiles_2030.nc",
        "profile_27_solar.nc",
    ]


def test_validation_rejects_wrong_reference_years(effective_config):
    c = copy.deepcopy(effective_config["controlled_weather"])
    c["previous_network"] = "wrong_2035.nc"
    with pytest.raises(ValidationError):
        ControlledWeatherConfig(**c)


def test_workflow_routes_directly_to_common_2030_and_fresh_vintage_inputs(
    effective_config,
):
    namespace = {
        "config": effective_config,
        "scenarios": {SCENARIO: {}},
        "copy": copy,
        "partial": partial,
        "lru_cache": lru_cache,
        "update_config": update_config,
        "update_config_from_wildcards": _helpers.update_config_from_wildcards,
        "resources": lambda path: f"resources/test/{{run}}/{path}",
    }
    selected = {
        "get_config",
        "merge_configs",
        "scenario_config",
        "static_getter",
        "dynamic_getter",
        "config_provider",
        "solved_previous_horizon",
        "controlled_weather_vintage_inputs",
    }
    for filename in [
        "common.smk",
        "solve_myopic.smk",
        "pypsa-de/controlled_weather.smk",
    ]:
        path = ROOT / "rules" / filename
        cache = SimpleNamespace(
            open=lambda p: open(p.get_path_or_uri(secret_free=True), encoding="utf-8")
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
    wildcards = Wildcards(
        fromdict={"run": SCENARIO, "clusters": "27", "planning_horizons": "2035"}
    )
    assert (
        namespace["solved_previous_horizon"](wildcards)
        == effective_config["controlled_weather"]["previous_network"]
    )
    inputs = namespace["controlled_weather_vintage_inputs"](wildcards)
    assert set(inputs) == {
        "controlled_weather_vintage_2025",
        "controlled_weather_vintage_2030",
    }
    assert all(
        "resources/" in path and "results/" not in path for path in inputs.values()
    )


def test_actual_preflight_script_checks_network_without_an_lp(
    tmp_path, monkeypatch, effective_config
):
    n, reference = fleet_pair()
    for model in [n, reference]:
        model.add("Bus", "co2 atmosphere", carrier="co2", country="")
        model.add(
            "Store", "co2 atmosphere", bus="co2 atmosphere", e_nom=1e9, e_min_pu=-1
        )
        model.add(
            "GlobalConstraint",
            "CO2Limit",
            type="co2_atmosphere",
            constant=100,
            mu=-PRICE,
        )
        model.add("GlobalConstraint", "co2_limit-DE", constant=50, mu=0)
    n.generators.at["FR0 solar-2035", "capital_cost"] = 99.0
    reference.generators.at["FR0 solar-2035", "capital_cost"] = 2.0
    reference_path = tmp_path / "reference_2035.nc"
    network_path = tmp_path / "weather_2035.nc"
    reference.export_to_netcdf(reference_path)
    n.export_to_netcdf(network_path)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "year": 2035,
                "component": "Generator",
                "asset": "FR0 solar-2035",
                "nominal": 50.0,
                "operational_nominal": 50.0,
            }
        ]
    ).to_csv(manifest, index=False)
    config = copy.deepcopy(effective_config)
    config["controlled_weather"]["mode"] = "capacity_expansion"
    config["solving"]["constraints"]["fixed_neighbor_capacities"]["enable"] = True
    config["controlled_weather"]["reference_network"] = str(reference_path)
    config["solving"]["constraints"]["fixed_neighbor_capacities"]["reference_networks"][
        2035
    ] = str(reference_path)
    report = tmp_path / "report.json"
    workflow = SimpleNamespace(
        config=config,
        input=SimpleNamespace(
            network=str(network_path),
            controlled_weather_reference=str(reference_path),
            fixed_neighbor_capacities=str(manifest),
        ),
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
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "passed_without_solve"
    assert result["fixed_co2_price_eur_per_t"] == PRICE
    assert result["controlled_weather"]["fixed_domestic_renewable_assets"] == 1
    headroom = config["solving"]["constraints"]["fixed_neighbor_capacities"][
        "capacity_tolerance"
    ]
    assert result["fixed_neighbor"]["removed_annualized_capex_eur"] == pytest.approx(
        (50.0 + headroom) * 2.0
    )
    assert not (report.parent / "preflight/networks/audit.nc").exists()


def test_expired_or_unbuilt_assets_do_not_require_irrelevant_profiles():
    previous, current, profiles = history_and_weather(2010)
    previous.generators.at["DE0 solar-2030", "lifetime"] = 5
    profiles[2030].remove("Generator", "DE0 solar")
    refresh_brownfield_weather(previous, current, profiles, settings())


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_fixed_2013_basic_load_is_preserved(year):
    source = pd.DataFrame(
        {"DE": np.arange(2920, dtype=float), "FR": 100.0}, index=calendar(2013)
    )
    target = align_load_to_snapshots(source, calendar(year), 2013)
    np.testing.assert_array_equal(target.to_numpy(), source.to_numpy())
    np.testing.assert_array_equal(target.sum(), source.sum())


def test_controlled_cost_accounting_includes_both_fixed_cost_constants():
    from test.test_fixed_co2_price import MODULE, co2_network, mock_snakemake

    n = co2_network()
    n.stores_t.p["co2 atmosphere"] = [-1.0, -1.0]
    n._objective = 1000.0
    n.meta["fixed_neighbor_static"] = {"removed_annualized_capex_eur": 100.0}
    n._controlled_weather_summary = {
        "removed_annualized_domestic_renewable_capex_eur": 50.0
    }
    MODULE.finalize_network_after_solve(n, mock_snakemake())
    result = n.meta["controlled_weather_result"]
    assert result["objective_including_removed_fixed_capex_eur"] == 1150.0
    assert result[
        "objective_including_removed_fixed_capex_excluding_co2_eur"
    ] == pytest.approx(1150.0 - 6 * PRICE)


def test_inherited_wind_uses_vintage_turbine_not_the_earliest_profile(tmp_path):
    previous, current = network(2013), network(2010)
    previous.add("Carrier", "onwind")
    previous.add(
        "Generator",
        "DE0 0 onwind-2030",
        bus="DE0",
        carrier="onwind",
        build_year=2030,
        lifetime=25,
        p_nom_opt=10,
        p_max_pu=[0.1, 0.1, 0.1],
    )
    values = np.array([0.1, 0.8])[:, None, None, None] * np.ones((2, 3, 1, 1))
    dataset = xr.Dataset(
        {"profile": (("year", "time", "bus", "bin"), values)},
        coords={
            "year": [2020, 2030],
            "time": current.snapshots.to_numpy(),
            "bus": ["DE0"],
            "bin": [0],
        },
    )
    path = tmp_path / "wind.nc"
    dataset.to_netcdf(path)
    profiles = {}
    for vintage, expected in [(2025, 0.1), (2030, 0.8)]:
        n = network(2010)
        n.add("Carrier", "onwind")
        n.add(
            "Generator",
            "DE0 0 onwind",
            bus="DE0",
            carrier="onwind",
            p_max_pu=[0.1, 0.1, 0.1],
        )
        params = {
            "carriers": ["onwind"],
            "snapshots": {
                "start": "2010-01-01",
                "end": "2010-01-01 09:00",
                "inclusive": "left",
            },
            "drop_leap_day": True,
        }
        adjust_renewable_profiles(
            n, SimpleNamespace(profile_onwind=str(path)), params, vintage
        )
        np.testing.assert_allclose(n.generators_t.p_max_pu["DE0 0 onwind"], expected)
        profiles[vintage] = n
    control = settings()
    control["frozen_generator_carriers"] = ["onwind"]
    refresh_brownfield_weather(previous, current, profiles, control)
    np.testing.assert_allclose(previous.generators_t.p_max_pu["DE0 0 onwind-2030"], 0.8)
