# SPDX-License-Identifier: MIT

"""Controlled LDES config, production attachment/hooks and result publication."""

import ast
import copy
import runpy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytest
import yaml
from snakemake.io import Namedlist, Wildcards

from scripts import add_electricity, solve_network
from test.test_controlled_weather import PRICE, ROOT, SCENARIO, calendar
from test.test_controlled_weather import effective_config as effective_config
from test.test_fixed_neighbor_constraints import MODULE


def technology(config):
    return (
        "iron-air"
        if "iron-air" in config["electricity"]["storage_options"]
        else "aCAES_RESC_continuous"
    )


def test_active_endogenous_technology_keeps_the_common_weather_design(request):
    c = request.getfixturevalue("effective_config")
    key = technology(c)
    options = c["electricity"]["storage_options"]
    assert set(options) == {key}
    family = "iron_air" if key == "iron-air" else "caes69"
    year = pd.Timestamp(c["snapshots"]["start"]).year
    assert year in (1996, 2010, 2023)
    assert c["run"]["name"] == [
        SCENARIO,
        *[f"{SCENARIO}_BESS_{power}GW" for power in (15, 35, 55)],
    ]
    assert c["run"]["prefix"] == f"thesis_weather_{family}_{year}"
    assert len(calendar(year)) == 2920
    assert c["clustering"]["temporal"]["resolution_sector"] == "3H"
    assert c["scenario"]["planning_horizons"] == [2025, 2030, 2035]
    assert c["sector"]["bev_dsm_availability"] == {2030: 0.2, 2035: 0.35}
    assert not c["sector"]["v2g"]
    assert c["load"]["fixed_year"] == 2013
    assert c["conventional"]["nuclear_availability_year"] == 2013
    assert c["costs"]["emission_prices"]["co2"][2035] == PRICE
    assert "battery_storage_duration" not in c
    constraints = c["solving"]["constraints"]
    assert "capacity_limit_basis" not in constraints
    for bound in ("limits_capacity_min", "limits_capacity_max"):
        for component, carrier in (
            ("Link", "battery discharger"),
            ("Store", "battery"),
        ):
            assert 2035 not in constraints[bound][component][carrier]["DE"]
    fixed = constraints["fixed_neighbor_capacities"]
    control = c["controlled_weather"]
    assert fixed["enable"] and fixed["formulation"] == "static"
    assert fixed["strict_asset_match"] is True
    assert fixed["capacity_tolerance"] == 0.012
    assert fixed["bound_clip_tolerance"] == 0.05
    assert fixed["capacity_manifest"] == (
        f"resources/thesis_weather_{family}_{year}/fixed_neighbor_capacities.csv"
    )
    reference = (
        f"cluster-results/endo_4h_3h_2025base_fixedgas_2026-09-14/{SCENARIO}/networks/"
    )
    assert fixed["reference_networks"] == {
        y: reference + f"base_s_27__none_{y}.nc" for y in (2025, 2030, 2035)
    }
    assert control["previous_network"] == fixed["reference_networks"][2030]
    assert control["reference_network"] == fixed["reference_networks"][2035]
    assert control["planning_horizon"] == 2035 and control["previous_horizon"] == 2030
    assert control["foreign_heat_unserved_energy"] is True
    assert control["foreign_heat_unserved_energy_cost"] == 100000.0
    assert control["resize_foreign_transport_oil_delivery"] is True
    assert (
        control["weather_resources_directory"]
        == f"resources/thesis_weather_{year}/{SCENARIO}"
    )
    assert c["atlite"]["default_cutout"] == f"europe-{year}-sarah3-era5"
    assert c["data"]["cutout"]["source"] == "archive"
    if key == "iron-air":
        o = options[key]
        assert c["electricity"]["extendable_carriers"]["Store"] == [
            "battery",
            "H2",
            "iron-air",
        ]
        assert o["countries"] == ["DE"] and o["bus_carriers"] == ["AC"]
        assert o["duration_hours_at_rated_ac_output"] == 100.0
        assert (o["charging_efficiency"], o["discharging_efficiency"]) == (0.71, 0.60)
        assert o["combined_capex_eur_per_kw_ac"] == 1547.09
        assert (o["lifetime_years"], o["discount_rate"], o["fom_percent"]) == (
            17.5,
            0.07,
            1.0,
        )
        assert o["currency_year"] == 2020 and o["source_currency_year_assumed"] == 2023
        assert o["source_capex_usd_per_kwh"] == 20.0
        assert o["source_fom_usd_per_kw_year"] == 20.0
    else:
        o = options[key]
        assert "iron-air" not in c["electricity"]["extendable_carriers"]["Store"]
        assert o["enable"] is True and o["first_investment_year"] == 2035
        assert o["round_trip_efficiency"] == 0.69
        assert o["minimum_output_duration_hours"] == 8
        assert o["maximum_output_duration_hours"] == 48
        assert o["lifetime_years"] == 30 and o["discount_rate"] == 0.07
        assert o["geological_output_capacity_twh"] == 7.6
        assert sum(o["site_output_capacities_twh"].values()) == pytest.approx(7.6)


def costs():
    names = {
        value
        for carrier in ("iron-air", "battery")
        for field, value in add_electricity.STORE_LOOKUP[carrier].items()
        if field in ("store", "charger", "bicharger", "discharger")
    }
    frame = pd.DataFrame(
        1.0,
        index=sorted(names),
        columns=[
            "capital_cost",
            "investment",
            "lifetime",
            "efficiency",
            "marginal_cost",
        ],
    )
    frame["lifetime"] = 20.0
    frame["efficiency"] = 0.81
    return frame


def small_network(config):
    n = pypsa.Network()
    year = pd.Timestamp(config["snapshots"]["start"]).year
    n.set_snapshots(calendar(year)[:3])
    n.snapshot_weightings.loc[:, :] = 3.0
    key = technology(config)
    sites = (
        ["DE test"]
        if key == "iron-air"
        else list(
            config["electricity"]["storage_options"][key]["site_output_capacities_twh"]
        )
    )
    n.add("Carrier", ["AC", "heat"])
    n.add("Bus", sites, country="DE", carrier="AC", x=0.0, y=0.0)
    n.add("Bus", "FR test", country="FR", carrier="AC", x=1.0, y=1.0)
    n.add("Bus", "DE heat", country="DE", carrier="heat", x=0.0, y=0.0)
    n.config = config
    return n, sites


def attach_via_sector_script(n, config, year=2035):
    """Execute the actual preparation call so omitted options/imports are caught."""
    key = technology(config)
    name = "attach_stores" if key == "iron-air" else "attach_continuous_acaes"
    path = ROOT / "scripts/prepare_sector_network.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]
    assert len(calls) == 1
    code = compile(
        ast.fix_missing_locations(
            ast.Module(body=[ast.Expr(calls[0])], type_ignores=[])
        ),
        str(path),
        "exec",
    )
    exec(
        code,
        {
            name: getattr(add_electricity, name),
            "n": n,
            "costs": costs(),
            "pop_layout": SimpleNamespace(index=n.buses.index),
            "extendable_stores": ["iron-air"],
            "investment_year": year,
            "snakemake": SimpleNamespace(
                params=SimpleNamespace(electricity=config["electricity"])
            ),
        },
    )


def constrain_via_solve_hook(n, config):
    if technology(config) == "iron-air":
        path = ROOT / "scripts/solve_network.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "extra_functionality"
        )
        body = [
            node
            for node in function.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "add_iron_air_constraints"
        ]
        assert len(body) == 1
        namespace = {
            "n": n,
            "add_iron_air_constraints": solve_network.add_iron_air_constraints,
        }
    else:
        path = ROOT / "scripts/pypsa-de/additional_functionality.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "additional_functionality"
        )
        i = next(
            i
            for i, node in enumerate(function.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "acaes_options"
                for target in node.targets
            )
        )
        body = function.body[i : i + 2]
        assert isinstance(body[1], ast.If)
        namespace = {
            "n": n,
            "snakemake": SimpleNamespace(config=config),
            "investment_year": 2035,
            "add_continuous_acaes_constraints": MODULE.add_continuous_acaes_constraints,
        }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def publish_via_solve_script(n, workflow):
    path = ROOT / "scripts/solve_network.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    )
    boundary = next(
        i
        for i, node in enumerate(main.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "status == 'warning'"
    )
    code = compile(
        ast.Module(body=main.body[boundary + 1 :], type_ignores=[]), str(path), "exec"
    )
    namespace = runpy.run_path(str(path), run_name="weather_ldes_publication_test")
    exec(code, {**namespace, "n": n, "snakemake": workflow})


def test_production_attachment_sizing_solve_and_publication(request, tmp_path):
    c = request.getfixturevalue("effective_config")
    key = technology(c)
    n, sites = small_network(c)
    attach_via_sector_script(n, c)
    o = c["electricity"]["storage_options"][key]
    carrier = "iron-air" if key == "iron-air" else o["carrier"]
    selected = n.stores.index[n.stores.carrier.eq(carrier)]
    assert len(selected) == len(sites)
    assert n.stores.loc[selected, "e_nom_extendable"].all()
    assert n.buses.loc[n.stores.loc[selected, "bus"], "country"].eq("DE").all()
    assert (
        n.buses.loc[n.links.loc[n.links.p_nom_extendable, "bus1"], "carrier"]
        .isin(["AC", carrier])
        .all()
    )
    n.add(
        "Generator",
        "charging supply",
        bus=sites[0],
        p_nom=100,
        p_max_pu=[1, 0, 0],
        marginal_cost=1,
    )
    n.add("Load", "output demand", bus=sites[0], p_set=[0, 0, 1])
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add(
        "Store",
        "co2 atmosphere",
        bus="co2 atmosphere",
        carrier="co2",
        e_nom=1e6,
        e_min_pu=-1,
    )
    n.add("GlobalConstraint", "CO2Limit", type="co2_atmosphere", constant=1000)
    n.add("GlobalConstraint", "co2_limit-DE", type="", constant=1000)
    workflow = SimpleNamespace(
        config=c,
        params=SimpleNamespace(
            solving=c["solving"],
            custom_extra_functionality=str(
                ROOT / "scripts/pypsa-de/additional_functionality.py"
            ),
        ),
        wildcards=Wildcards(
            fromdict={
                "planning_horizons": "2035",
                "clusters": "27",
                "run": SCENARIO,
                "opts": "",
                "sector_opts": "none",
            }
        ),
        output=Namedlist(
            fromdict={
                "network": str(tmp_path / "result.nc"),
                "config": str(tmp_path / "config.yaml"),
            }
        ),
    )
    MODULE.prepare_fixed_co2_price(n, workflow)

    def extra(network, snapshots):
        constrain_via_solve_hook(network, c)
        MODULE.validate_fixed_co2_price_model(network, workflow)

    status, condition = n.optimize(
        solver_name="highs", extra_functionality=extra, log_to_console=False
    )
    assert (status, condition) == ("ok", "optimal")
    store = selected[0]
    charger = f"{store} charger"
    discharger = f"{store} discharger"
    power = n.links.at[discharger, "p_nom_opt"] * n.links.at[discharger, "efficiency"]
    energy = n.stores.at[store, "e_nom_opt"] * n.links.at[discharger, "efficiency"]
    assert power > 0
    assert n.links.at[charger, "p_nom_opt"] == pytest.approx(power)
    if key == "iron-air":
        assert energy / power == pytest.approx(100)
        expected_rte = o["charging_efficiency"] * o["discharging_efficiency"]
        expected_cost = (
            (add_electricity.calculate_annuity(17.5, 0.07) + 0.01)
            * 1547.09
            * 1000
            * 9
            / 8760
        )
        assert n.links.at[charger, "capital_cost"] == pytest.approx(expected_cost)
        assert n.stores.at[store, "capital_cost"] == 0
    else:
        assert 8 - 1e-6 <= energy / power <= 48 + 1e-6
        assert energy <= o["site_output_capacities_twh"][sites[0]] * 1e6 + 1e-6
        expected_rte = 0.69
    charge_energy = n.links_t.p0[charger].dot(n.snapshot_weightings.generators)
    output_energy = (-n.links_t.p1[discharger]).dot(n.snapshot_weightings.generators)
    assert output_energy / charge_energy == pytest.approx(expected_rte)
    n._fixed_neighbor_static_summary = {"removed_annualized_capex_eur": 123.0}
    n._controlled_weather_summary = {
        "weather_year": n.snapshots[0].year,
        "removed_annualized_domestic_renewable_capex_eur": 45.0,
    }
    publish_via_solve_script(n, workflow)
    restored = pypsa.Network(workflow.output.network)
    metadata = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    for result in (restored.meta, metadata):
        assert (
            result["controlled_weather_result"]["weather_year"] == n.snapshots[0].year
        )
        assert result["controlled_weather_result"]["unserved_heat_mwh_th"] == 0
        assert result["controlled_weather_result"][
            "objective_including_removed_fixed_capex_excluding_co2_and_unserved_heat_penalty_eur"
        ] == pytest.approx(n.objective + 168)
    np.testing.assert_allclose(restored.stores_t.e, n.stores_t.e)


def test_legacy_storage_or_earlier_vintage_inputs_are_unchanged(request):
    c = request.getfixturevalue("effective_config")
    n, _ = small_network(c)
    if technology(c) == "iron-air":
        add_electricity.attach_stores(
            n,
            costs(),
            pd.Index(["DE test", "FR test"]),
            ["battery"],
            c["electricity"]["storage_options"],
        )
        assert len(n.stores) == 2
        assert n.stores.capital_cost.eq(1).all()
        assert n.stores.lifetime.eq(20).all()
        assert n.links.efficiency.eq(0.9).all()
        assert (
            n.links.loc[n.links.carrier.eq("battery charger"), "capital_cost"]
            .eq(1)
            .all()
        )
    else:
        for year in (2025, 2030):
            attach_via_sector_script(n, c, year)
        assert n.stores.empty and n.links.empty


def test_invalid_technology_options_fail_explicitly(request):
    c = copy.deepcopy(request.getfixturevalue("effective_config"))
    key = technology(c)
    if key == "iron-air":
        del c["electricity"]["storage_options"][key]["lifetime_years"]
        message = "Missing iron-air storage options"
    else:
        c["electricity"]["storage_options"][key]["round_trip_efficiency"] = 0
        message = "round-trip efficiency"
    n, _ = small_network(c)
    with pytest.raises(ValueError, match=message):
        attach_via_sector_script(n, c)
