# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""Fixed-load-calendar regression tests without downloads or network solves."""

import ast
import copy
import runpy
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from snakemake.io import Wildcards
from snakemake.parser import parse
from snakemake.sourcecache import LocalSourceFile
from snakemake.utils import update_config

from scripts import _helpers
from scripts.build_electricity_demand import (
    align_load_to_snapshots,
    load_source_snapshots,
)
from scripts.lib.validation.config import validate_config

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = [
    "AT",
    "BE",
    "CH",
    "CZ",
    "DE",
    "DK",
    "FR",
    "GB",
    "LU",
    "NL",
    "NO",
    "PL",
    "SE",
    "ES",
    "IT",
]


def calendar(year):
    return _helpers.get_snapshots(
        {"start": f"{year}-01-01", "end": f"{year + 1}-01-01", "inclusive": "left"},
        True,
    )


@pytest.mark.parametrize("year", [1996, 2010, 2013, 2023])
def test_fixed_2013_load_preserves_values_and_country_totals(year):
    source = calendar(2013)
    target = calendar(year)
    data = np.arange(len(source) * len(COUNTRIES), dtype=float).reshape(
        len(source), len(COUNTRIES)
    )
    load = pd.DataFrame(data, index=source, columns=COUNTRIES)
    result = align_load_to_snapshots(load, target, 2013)
    expected = load.set_axis(target)

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_series_equal(result.sum(), load.sum())
    assert len(result) == 8760
    assert not ((result.index.month == 2) & (result.index.day == 29)).any()


def test_unfixed_2013_load_matches_the_original_same_year_path():
    source = calendar(2013)
    load = pd.DataFrame({"DE": np.arange(len(source), dtype=float)}, index=source)
    original = load.loc[slice(source[0], source[-1])].reindex(index=source)

    pd.testing.assert_frame_equal(align_load_to_snapshots(load, source), original)


@pytest.mark.parametrize("missing", ["timestamp", "country", "value"])
def test_missing_source_load_is_rejected_before_export(missing):
    source = calendar(2013)
    load = pd.DataFrame(1.0, index=source, columns=["DE", "GB"])
    if missing == "timestamp":
        load = load.drop(source[10])
    elif missing == "country":
        load["GB"] = np.nan
    else:
        load.loc[source[10], "DE"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        align_load_to_snapshots(load, calendar(1996), 2013)


def test_leap_day_requires_an_explicit_calendar_policy():
    target = pd.date_range("1996-02-28", "1996-03-01", freq="h")
    with pytest.raises(ValueError, match="drop_leap_day"):
        load_source_snapshots(target, 2013)


def test_fixed_load_rejects_multiple_target_years():
    target = pd.date_range("2010-12-31", "2011-01-01", freq="h")
    with pytest.raises(ValueError, match="single target calendar year"):
        load_source_snapshots(target, 2013)


def test_source_leap_day_is_not_carried_into_a_nonleap_target():
    source = pd.date_range("2012-02-28", "2012-03-01", freq="h")
    load = pd.DataFrame({"DE": 1.0}, index=source)
    target = pd.date_range("2013-02-28", "2013-03-01", freq="h")
    result = align_load_to_snapshots(load, target, 2012)
    assert len(result) == len(target)
    assert not result.isna().any().any()


@pytest.mark.parametrize(
    ("year", "fixed_year"),
    [(1996, 2013), (2010, 2013), (2013, 2013), (2013, False), (2023, 2013)],
)
def test_script_supplements_from_source_year_not_weather_year(
    monkeypatch, year, fixed_year
):
    source = calendar(2013)
    values = np.arange(len(source), dtype=float) + 10000.0
    measured = pd.DataFrame(
        {country: values + i for i, country in enumerate(COUNTRIES)}, index=source
    )
    measured["GB"] = np.nan
    synthetic = pd.DataFrame(
        {country: values + 100 + i for i, country in enumerate(COUNTRIES)}, index=source
    )
    captured = {}

    def read_csv(path, **kwargs):
        return (synthetic if path == "synthetic.csv" else measured).copy()

    def to_csv(frame, path, **kwargs):
        captured["load"] = frame.copy()

    monkeypatch.setattr(pd, "read_csv", read_csv)
    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    monkeypatch.setattr(_helpers, "configure_logging", lambda sm: None)
    monkeypatch.setattr(_helpers, "set_scenario_config", lambda sm: None)
    params = SimpleNamespace(
        snapshots={
            "start": f"{year}-01-01",
            "end": f"{year + 1}-01-01",
            "inclusive": "left",
        },
        drop_leap_day=True,
        countries=COUNTRIES,
        load={
            "fixed_year": fixed_year,
            "manual_adjustments": False,
            "supplement_synthetic": True,
            "fill_gaps": {
                "enable": False,
                "interpolate_limit": 6,
                "time_shift_for_large_gaps": "1w",
            },
        },
    )
    snakemake = SimpleNamespace(
        params=params,
        input=SimpleNamespace(
            opsd="opsd.csv",
            entsoe="entsoe.csv",
            neso="neso.csv",
            synthetic="synthetic.csv",
        ),
        output=["unused.csv"],
    )
    runpy.run_path(
        str(ROOT / "scripts/build_electricity_demand.py"),
        run_name="__main__",
        init_globals={"snakemake": snakemake},
    )

    expected = measured.combine_first(synthetic).set_axis(calendar(year))
    pd.testing.assert_frame_equal(captured["load"], expected)
    assert list(captured["load"].columns) == COUNTRIES


@pytest.fixture
def weather_reference_config():
    config = {}
    for filename in ["config.default.yaml", "config.de.yaml"]:
        update_config(
            config,
            yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8")),
        )
    manual = yaml.safe_load(
        (ROOT / "config/scenarios.manual.yaml").read_text(encoding="utf-8")
    )
    scenario = "KN2045_Mix_FixedRenewables"
    year = manual[scenario]["weather_year"]
    generated = {scenario: copy.deepcopy(manual[scenario])}
    builder = runpy.run_path(str(ROOT / "scripts/pypsa-de/build_scenarios.py"))
    builder["write_weather_dependent_config"](generated, scenario, year)
    update_config(config, generated[scenario])
    validate_config(config)
    return config


def test_effective_weather_reference_configuration(weather_reference_config):
    config = weather_reference_config
    scenario = "KN2045_Mix_FixedRenewables"
    year = pd.Timestamp(config["snapshots"]["start"]).year

    cutout = f"europe-{year}-sarah3-era5"
    assert config["run"]["name"] == [scenario]
    assert config["run"]["prefix"] == f"thesis_weather_{year}"
    assert config["run"]["shared_resources"]["policy"] is False
    assert config["atlite"]["default_cutout"] == cutout
    assert config["atlite"]["cutouts"][cutout]["time"] == [str(year), str(year)]
    assert config["load"]["fixed_year"] == 2013
    assert config["load"]["scaling_factor"] == 1.0
    assert config["conventional"]["nuclear_availability_year"] == 2013
    assert config["conventional"]["nuclear"]["p_max_pu"] == "data/nuclear_p_max_pu.csv"
    assert config["sector"]["heat_demand_cutout"] == "default"
    for technology in [
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
        "solar",
        "solar-hsat",
        "hydro",
    ]:
        assert config["renewable"][technology]["cutout"] == cutout
    assert config["lines"]["dynamic_line_rating"]["cutout"] == cutout
    assert config["data"]["cutout"]["source"] == "archive"
    assert config["scenario"]["planning_horizons"] == [2025, 2030, 2035]
    assert config["clustering"]["temporal"]["resolution_sector"] == "3H"
    assert config["sector"]["bev_dsm_availability"] == {2030: 0.2, 2035: 0.35}
    assert config["sector"]["v2g"] is False
    assert config["german_conventional_capacity_pathway"]["fix_gas_capacity_2025"]
    assert config["co2_budget"][2035] == 0.285
    assert config["foresight"] == "myopic"

    target = _helpers.get_snapshots(
        config["snapshots"], config["enable"]["drop_leap_day"]
    )
    assert len(target) == 8760
    assert len(target[::3]) == 2920
    resource = _helpers.get_run_path(
        "profile_27_onwind.nc",
        "resources/",
        _helpers.get_rdir(config["run"]),
        config["run"]["shared_resources"]["policy"],
        config["run"]["shared_resources"]["exclude"],
    )
    assert resource.format(run=scenario).startswith(
        f"resources/thesis_weather_{year}/{scenario}/"
    )
    constraints = config["solving"]["constraints"]
    assert not constraints.get("fixed_neighbor_capacities", {}).get("enable", False)
    assert not constraints.get("fixed_co2_price", {}).get("enable", False)
    assert 2035 not in constraints["limits_capacity_max"]["Store"]["battery"]["DE"]


def test_nuclear_reference_reaches_add_electricity_workflow(weather_reference_config):
    config = weather_reference_config
    scenario = "KN2045_Mix_FixedRenewables"
    namespace = {
        "config": config,
        "scenarios": {scenario: {}},
        "copy": copy,
        "partial": partial,
        "lru_cache": lru_cache,
        "update_config": update_config,
        "update_config_from_wildcards": _helpers.update_config_from_wildcards,
    }
    sourcecache = SimpleNamespace(
        open=lambda source: open(
            source.get_path_or_uri(secret_free=True), encoding="utf-8"
        )
    )
    getter_functions = {
        "get_config",
        "merge_configs",
        "scenario_config",
        "static_getter",
        "dynamic_getter",
        "config_provider",
    }
    rule_ast = None
    for filename in ["common.smk", "build_electricity.smk"]:
        path = ROOT / "rules" / filename
        compiled, _ = parse(
            LocalSourceFile(str(path)), SimpleNamespace(sourcecache=sourcecache), {}
        )
        rule_ast = ast.parse(compiled)
        functions = [
            node
            for node in rule_ast.body
            if isinstance(node, ast.FunctionDef)
            and node.name in getter_functions | {"input_conventional"}
        ]
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
            namespace,
        )

    wildcards = Wildcards(fromdict={"run": scenario, "clusters": "27"})
    inputs = namespace["input_conventional"](wildcards)
    assert inputs["conventional_nuclear_p_max_pu"] == "data/nuclear_p_max_pu.csv"
    assert not any("availability_year" in name for name in inputs)

    param_expressions = [
        keyword.value
        for node in ast.walk(rule_ast)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "params"
        for keyword in node.keywords
        if keyword.arg == "conventional"
    ]
    assert len(param_expressions) == 1
    getter = eval(
        compile(ast.Expression(param_expressions[0]), "params", "eval"), namespace
    )
    assert getter(wildcards)["nuclear_availability_year"] == 2013
