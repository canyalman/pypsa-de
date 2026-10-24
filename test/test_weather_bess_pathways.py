# SPDX-License-Identifier: MIT

"""Fixed AC-output BESS pathways reuse the controlled weather experiment."""

import copy
import runpy

import numpy as np
import pandas as pd
import pytest
import yaml
from snakemake.utils import update_config

from scripts.lib.validation.config import validate_config
from scripts.solve_network import add_battery_constraints
from test.test_battery_ac_pathways import battery_network
from test.test_controlled_weather import PRICE, ROOT, SCENARIO, calendar
from test.test_fixed_neighbor_constraints import MODULE
from test.test_weather_workflow import actual_dag, fixture_workflow

POWERS = (15, 35, 55)
SCENARIOS = [f"{SCENARIO}_BESS_{power}GW" for power in POWERS]


def manual_scenarios():
    return yaml.safe_load(
        (ROOT / "config/scenarios.manual.yaml").read_text(encoding="utf-8")
    )


def scenario_config(power):
    config = {}
    for filename in ("config.default.yaml", "config.de.yaml"):
        update_config(
            config,
            yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8")),
        )
    name = f"{SCENARIO}_BESS_{power}GW"
    generated = {name: copy.deepcopy(manual_scenarios()[name])}
    builder = runpy.run_path(str(ROOT / "scripts/pypsa-de/build_scenarios.py"))
    builder["write_weather_dependent_config"](
        generated, name, generated[name]["weather_year"]
    )
    update_config(config, generated[name])
    validate_config(config)
    return config


@pytest.mark.parametrize("power", POWERS)
def test_only_utility_bess_pathway_changes_the_manual_assumptions(power):
    manual = manual_scenarios()
    case = copy.deepcopy(manual[f"{SCENARIO}_BESS_{power}GW"])
    assert case.pop("battery_storage_duration") == {"exclude": {"DE": [2025]}}
    constraints = case["solving"]["constraints"]
    assert constraints.pop("capacity_limit_basis") == {
        "Link": {"battery discharger": {"DE": {2035: "output"}}}
    }
    for bound in ("limits_capacity_min", "limits_capacity_max"):
        limits = constraints[bound]
        assert limits["Link"]["battery discharger"]["DE"].pop(2035) == power
        assert 2035 not in limits["Store"]["battery"]["DE"]
    assert case == manual[SCENARIO]


@pytest.mark.parametrize("power", POWERS)
def test_effective_pathways_keep_weather_references_flexibility_and_carbon_price(power):
    c = scenario_config(power)
    year = pd.Timestamp(c["snapshots"]["start"]).year
    assert year in (1996, 2010, 2023)
    assert c["run"]["name"] == SCENARIOS
    assert c["run"]["prefix"] == f"thesis_weather_bess_{year}"
    assert c["run"]["shared_resources"]["policy"] is False
    assert c["scenario"]["planning_horizons"] == [2025, 2030, 2035]
    assert c["clustering"]["temporal"]["resolution_sector"] == "3H"
    assert len(calendar(year)) == 2920
    assert c["load"]["fixed_year"] == 2013
    assert c["conventional"]["nuclear_availability_year"] == 2013
    assert c["sector"]["bev_dsm_availability"] == {2030: 0.2, 2035: 0.35}
    assert not c["sector"]["v2g"]
    assert c["costs"]["emission_prices"]["co2"][2035] == PRICE
    fixed = c["solving"]["constraints"]["fixed_neighbor_capacities"]
    control = c["controlled_weather"]
    assert control["enable"] is True
    assert control.get("mode", "capacity_expansion") == "capacity_expansion"
    assert (control["previous_horizon"], control["planning_horizon"]) == (2030, 2035)
    reference = (
        f"cluster-results/endo_4h_3h_2025base_fixedgas_2026-09-14/{SCENARIO}/networks/"
    )
    assert control["previous_network"] == reference + "base_s_27__none_2030.nc"
    assert control["reference_network"] == reference + "base_s_27__none_2035.nc"
    assert fixed["reference_networks"] == {
        y: reference + f"base_s_27__none_{y}.nc" for y in (2025, 2030, 2035)
    }
    assert fixed["enable"] is True
    assert fixed["formulation"] == "static"
    assert fixed["strict_asset_match"] is True
    assert fixed["capacity_tolerance"] == 0.012
    assert fixed["bound_clip_tolerance"] == 0.05
    assert fixed["capacity_manifest"] == (
        f"resources/thesis_weather_bess_{year}/fixed_neighbor_capacities.csv"
    )
    assert c["solving"]["constraints"]["fixed_co2_price"] == {
        "enable": True,
        "planning_horizons": [2035],
    }
    assert control["foreign_heat_unserved_energy"] is True
    assert control["foreign_heat_unserved_energy_cost"] == 100000.0
    assert control["resize_foreign_transport_oil_delivery"] is True
    assert control["weather_resources_directory"] == (
        f"resources/thesis_weather_{year}/{SCENARIO}"
    )
    cutout = f"europe-{year}-sarah3-era5"
    assert c["atlite"]["default_cutout"] == cutout
    assert c["data"]["cutout"]["source"] == "archive"
    for carrier in (
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
        "solar",
        "solar-hsat",
        "hydro",
    ):
        assert c["renewable"][carrier]["cutout"] == cutout
    assert c["lines"]["dynamic_line_rating"]["cutout"] == cutout


@pytest.mark.parametrize("power", POWERS)
def test_configured_ac_power_and_four_hour_energy_include_inherited_batteries(power):
    c = scenario_config(power)
    n = battery_network()
    n.config = c
    constraints = c["solving"]["constraints"]
    for bound, sense in (
        ("limits_capacity_min", "minimum"),
        ("limits_capacity_max", "maximum"),
    ):
        limits = {
            "Link": {
                "battery discharger": constraints[bound]["Link"]["battery discharger"]
            }
        }
        MODULE.add_capacity_limits(
            n,
            2035,
            limits,
            sense,
            capacity_limit_basis=constraints["capacity_limit_basis"],
        )
    add_battery_constraints(n, planning_horizons=2035)
    status, condition = n.optimize.solve_model(solver_name="highs")
    assert (status, condition) == ("ok", "optimal")
    links = n.links.loc[["DE battery discharger-2030", "DE battery discharger-2035"]]
    power_input = links.p_nom_opt.sum()
    power_output = (links.p_nom_opt * links.efficiency).sum()
    energy_internal = n.stores.loc[
        ["DE battery-2025", "DE battery-2035"], "e_nom_opt"
    ].sum()
    assert power_output == pytest.approx(power * 1000)
    assert energy_internal == pytest.approx(4 * power_input)
    assert energy_internal * power_output / power_input == pytest.approx(power * 4000)
    assert energy_internal > power * 4000
    assert n.links.at["DE battery discharger-2030", "p_nom_opt"] == pytest.approx(1)
    assert n.links.at["PL battery discharger-2030", "p_nom_opt"] == pytest.approx(10)
    assert n.stores.at["PL battery-2030", "e_nom_opt"] == pytest.approx(40)
    assert n.stores.at["DE home battery-2030", "e_nom_opt"] == pytest.approx(10)
    assert np.isfinite(n.objective)


def test_three_pathways_build_separate_preflights_using_the_same_weather_cache(
    tmp_path,
):
    snakefile = fixture_workflow(
        tmp_path,
        cache_files=(
            "regions_onshore.geojson",
            "networks/base.nc",
            "profile_27_solar.nc",
            "regions_by_class_27_solar.geojson",
        ),
        run_names=SCENARIOS,
    )
    jobs = actual_dag(snakefile, ["controlled_weather_preflight"])
    checks = [
        job for job in jobs if job.rule.name == "check_controlled_weather_network"
    ]
    assert {job.wildcards.run for job in checks} == set(SCENARIOS)
    assert all(job.wildcards.planning_horizons == "2035" for job in checks)
    assert len({str(job.output[0]) for job in checks}) == 3
    assert all(
        job.input.controlled_weather_reference == "reference_2035.nc" for job in checks
    )
    prepared = [job for job in jobs if job.rule.name == "prepare_final"]
    assert {job.wildcards.run for job in prepared} == set(SCENARIOS)
    assert all("reference_2030.nc" in job.input for job in prepared)
    copies = [
        job for job in jobs if job.rule.name == "reuse_controlled_weather_resource"
    ]
    assert len(copies) == 3
    assert len({str(job.input.source) for job in copies}) == 1
    assert not any(job.rule.name == "build_renewable_profiles" for job in jobs)
    assert not any(job.rule.name.startswith("solve_") for job in jobs)
    assert not any("/networks/" in str(job.output[0]) for job in copies)
