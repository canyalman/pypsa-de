# SPDX-License-Identifier: MIT

"""Endogenous and fixed-BESS LDES scenarios share references, not power targets."""

import ast
import copy
import runpy
from functools import lru_cache, partial
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from snakemake.io import Wildcards
from snakemake.parser import parse
from snakemake.sourcecache import LocalSourceFile
from snakemake.utils import update_config

from scripts import _helpers
from scripts.lib.validation.config import validate_config
from scripts.solve_network import add_battery_constraints
from test.test_battery_ac_pathways import battery_network
from test.test_controlled_weather import ROOT, SCENARIO
from test.test_fixed_neighbor_constraints import MODULE
from test.test_weather_ldes import (
    attach_via_sector_script,
    constrain_via_solve_hook,
    technology,
)
from test.test_weather_workflow import actual_dag, fixture_workflow

POWERS = (15, 35, 55)
SCENARIOS = [SCENARIO, *[f"{SCENARIO}_BESS_{power}GW" for power in POWERS]]


def configurations():
    base = {}
    for filename in ("config.default.yaml", "config.de.yaml"):
        update_config(
            base,
            yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8")),
        )
    manual = yaml.safe_load(
        (ROOT / "config/scenarios.manual.yaml").read_text(encoding="utf-8")
    )
    builder = runpy.run_path(str(ROOT / "scripts/pypsa-de/build_scenarios.py"))
    cases = {name: copy.deepcopy(manual[name]) for name in SCENARIOS}
    for name, case in cases.items():
        builder["write_weather_dependent_config"](cases, name, case["weather_year"])
    return base, cases


def scenario_config(power):
    base, cases = configurations()
    name = SCENARIO if power is None else f"{SCENARIO}_BESS_{power}GW"
    update_config(base, cases[name])
    validate_config(base)
    return base


@pytest.mark.parametrize("power", POWERS)
def test_only_bess_power_basis_and_duration_override_the_endogenous_template(power):
    manual = yaml.safe_load(
        (ROOT / "config/scenarios.manual.yaml").read_text(encoding="utf-8")
    )
    case = copy.deepcopy(manual[f"{SCENARIO}_BESS_{power}GW"])
    assert case.pop("battery_storage_duration") == {"exclude": {"DE": [2025]}}
    limits = case["solving"]["constraints"]
    assert limits.pop("capacity_limit_basis") == {
        "Link": {"battery discharger": {"DE": {2035: "output"}}}
    }
    for bound in ("limits_capacity_min", "limits_capacity_max"):
        assert limits[bound]["Link"]["battery discharger"]["DE"].pop(2035) == power
        assert 2035 not in limits[bound]["Store"]["battery"]["DE"]
    assert case == manual[SCENARIO]


def test_real_scenario_provider_does_not_leak_fixed_targets_into_endogenous_run():
    base, cases = configurations()
    original = copy.deepcopy(base)
    namespace = {
        "config": base,
        "scenarios": cases,
        "copy": copy,
        "partial": partial,
        "lru_cache": lru_cache,
        "update_config": update_config,
        "update_config_from_wildcards": _helpers.update_config_from_wildcards,
        "resources": lambda path: f"resources/test/{{run}}/{path}",
    }
    names = {
        "get_config",
        "merge_configs",
        "scenario_config",
        "static_getter",
        "dynamic_getter",
        "config_provider",
        "solved_previous_horizon",
    }
    path = ROOT / "rules/common.smk"
    cache = SimpleNamespace(
        open=lambda source: open(
            source.get_path_or_uri(secret_free=True), encoding="utf-8"
        )
    )
    compiled, _ = parse(
        LocalSourceFile(str(path)), SimpleNamespace(sourcecache=cache), {}
    )
    body = [
        node
        for node in ast.parse(compiled).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    solving = namespace["config_provider"]("solving")
    electricity = namespace["config_provider"]("electricity")
    common = scenario_config(None)
    assert common["run"]["name"] == SCENARIOS
    year = pd.Timestamp(common["snapshots"]["start"]).year
    assert year in (1996, 2010, 2023)
    for name, power in [*zip(SCENARIOS, (None, *POWERS)), (SCENARIO, None)]:
        w = Wildcards(
            fromdict={"run": name, "clusters": "27", "planning_horizons": "2035"}
        )
        limits = solving(w)["constraints"]
        effective = namespace["scenario_config"](name)
        validate_config(effective)
        assert pd.Timestamp(effective["snapshots"]["start"]).year == year
        assert effective["atlite"]["default_cutout"] == f"europe-{year}-sarah3-era5"
        assert effective["controlled_weather"] == common["controlled_weather"]
        assert effective["costs"] == common["costs"]
        assert effective["sector"] == common["sector"]
        assert electricity(w) == common["electricity"]
        assert (
            limits["fixed_neighbor_capacities"]
            == common["solving"]["constraints"]["fixed_neighbor_capacities"]
        )
        assert (
            limits["fixed_co2_price"]
            == common["solving"]["constraints"]["fixed_co2_price"]
        )
        assert (
            namespace["solved_previous_horizon"](w)
            == common["controlled_weather"]["previous_network"]
        )
        for bound in ("limits_capacity_min", "limits_capacity_max"):
            target = limits[bound]["Link"]["battery discharger"]["DE"].get(2035)
            assert target == power
        if power is None:
            assert "capacity_limit_basis" not in limits
            assert "battery_storage_duration" not in effective
        else:
            assert (
                limits["capacity_limit_basis"]["Link"]["battery discharger"]["DE"][2035]
                == "output"
            )
            assert effective["battery_storage_duration"] == {"exclude": {"DE": [2025]}}
    assert base == original


@pytest.mark.parametrize(
    "power", (None, *POWERS), ids=("endogenous", "15GW", "35GW", "55GW")
)
def test_li_ion_targets_and_four_hour_energy_coexist_with_endogenous_ldes(power):
    config = scenario_config(power)
    n = battery_network()
    n.config = config
    key = technology(config)
    if key == "aCAES_RESC_continuous":
        sites = list(
            config["electricity"]["storage_options"][key]["site_output_capacities_twh"]
        )
        n.add("Bus", sites, country="DE", carrier="AC", x=0.0, y=0.0)
    attach_via_sector_script(n, config)
    carrier = (
        "iron-air"
        if key == "iron-air"
        else config["electricity"]["storage_options"][key]["carrier"]
    )
    ldes_stores = n.stores.index[n.stores.carrier.eq(carrier)]
    ldes_links = n.links.index[
        n.links.carrier.isin(
            ["iron-air charger", "iron-air discharger"]
            if key == "iron-air"
            else [f"{carrier} charge", f"{carrier} discharge"]
        )
    ]
    assert len(ldes_stores) > 0 and len(ldes_links) == 2 * len(ldes_stores)
    assert n.stores.loc[ldes_stores, "e_nom_extendable"].all()
    assert n.links.loc[ldes_links, "p_nom_extendable"].all()
    n.optimize.create_model()
    assert n.model["Store-e_nom"].loc[ldes_stores].size == len(ldes_stores)
    assert n.model["Link-p_nom"].loc[ldes_links].size == len(ldes_links)
    limits = config["solving"]["constraints"]
    for bound, sense in (
        ("limits_capacity_min", "minimum"),
        ("limits_capacity_max", "maximum"),
    ):
        battery_limits = {
            "Link": {"battery discharger": limits[bound]["Link"]["battery discharger"]}
        }
        MODULE.add_capacity_limits(
            n,
            2035,
            battery_limits,
            sense,
            capacity_limit_basis=limits.get("capacity_limit_basis"),
        )
    add_battery_constraints(n, planning_horizons=2035)
    constrain_via_solve_hook(n, config)
    status, condition = n.optimize.solve_model(
        solver_name="highs", log_to_console=False
    )
    assert (status, condition) == ("ok", "optimal")
    links = n.links.loc[["DE battery discharger-2030", "DE battery discharger-2035"]]
    input_power = links.p_nom_opt.sum()
    output_power = (links.p_nom_opt * links.efficiency).sum()
    energy = n.stores.loc[["DE battery-2025", "DE battery-2035"], "e_nom_opt"].sum()
    assert energy == pytest.approx(4 * input_power)
    if power is None:
        assert output_power < 100
    else:
        assert output_power == pytest.approx(power * 1000)
        assert energy * output_power / input_power == pytest.approx(power * 4000)
    assert n.links.at["DE battery discharger-2030", "p_nom_opt"] == pytest.approx(1)
    assert n.links.at["PL battery discharger-2030", "p_nom_opt"] == pytest.approx(10)
    assert n.stores.at["PL battery-2030", "e_nom_opt"] == pytest.approx(40)
    assert n.stores.at["DE home battery-2030", "e_nom_opt"] == pytest.approx(10)
    assert "Store-n_mod" not in n.model.variables
    assert "Link-n_mod" not in n.model.variables


def test_four_scenarios_have_independent_preflights_and_shared_weather_sources(
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
    assert len({str(job.output[0]) for job in checks}) == 4
    assert all(
        job.input.controlled_weather_reference == "reference_2035.nc" for job in checks
    )
    prepared = [job for job in jobs if job.rule.name == "prepare_final"]
    assert {job.wildcards.run for job in prepared} == set(SCENARIOS)
    assert all("reference_2030.nc" in job.input for job in prepared)
    copies = [
        job for job in jobs if job.rule.name == "reuse_controlled_weather_resource"
    ]
    assert len(copies) == 4
    assert len({str(job.input.source) for job in copies}) == 1
    assert not any(job.rule.name == "build_renewable_profiles" for job in jobs)
    assert not any(job.rule.name.startswith("solve_") for job in jobs)
    assert not any("/networks/" in str(job.output[0]) for job in copies)
