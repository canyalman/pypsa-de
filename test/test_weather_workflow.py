# SPDX-License-Identifier: MIT

"""Real Snakemake rule loading and DAG regressions, without model data or Slurm."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from snakemake.api import SnakemakeApi
from snakemake.settings.types import (
    DAGSettings,
    OutputSettings,
    ResourceSettings,
    SharedFSUsage,
    StorageSettings,
    WorkflowSettings,
)

from scripts.controlled_weather import refresh_brownfield_weather, validate_input_series
from scripts.weather_workflow import reusable_weather_resources
from test.test_controlled_weather import history_and_weather, settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_auxiliary_solver_outputs_are_not_treated_as_weather_inputs(year):
    previous, current, profiles = history_and_weather(year)
    for component, attribute in [("Link", "p-incoming-DE-aux-pos"), ("Line", "mu_pos")]:
        previous.components[component].dynamic[attribute] = pd.DataFrame(
            1.0, index=previous.snapshots, columns=["old auxiliary result"]
        )
    validate_input_series(previous)
    before = previous.links.copy()
    refresh_brownfield_weather(previous, current, profiles, settings())
    assert previous.links_t["p-incoming-DE-aux-pos"].empty
    assert previous.lines_t["mu_pos"].empty
    pd.testing.assert_frame_equal(previous.links, before)
    validate_input_series(previous)


def test_unknown_outputs_do_not_mask_invalid_standard_input():
    previous, _, _ = history_and_weather(2023)
    previous.links_t["p-incoming-DE-aux-pos"] = pd.DataFrame(
        1.0, index=previous.snapshots, columns=["aux"]
    )
    previous.generators_t.p_max_pu.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Missing/non-finite input"):
        validate_input_series(previous)


def fixture_workflow(tmp_path, cache_files=(), dispatch=False):
    """Include the production weather rule file, not an AST-only approximation."""
    if dispatch and "rule prepare_dispatch_weather_network:" not in (
        ROOT / "rules/pypsa-de/controlled_weather.smk"
    ).read_text(encoding="utf-8"):
        pytest.skip("Dispatch-specific workflow is only present on dispatch branches.")
    cache = tmp_path / "cache"
    cache.mkdir()
    for relative in cache_files:
        path = cache / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")
    for relative in [
        "scripts/controlled_weather.py",
        "scripts/weather_workflow.py",
        "scripts/weather_accounting.py",
        "scripts/pypsa-de/additional_functionality.py",
        "scripts/dispatch_weather.py",
        "reference_2035.nc",
        "reference_2030.nc",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config = {
        "controlled_weather": {
            "enable": True,
            "mode": "dispatch" if dispatch else "capacity_expansion",
            "reference_network": "reference_2035.nc",
            "weather_resources_directory": str(cache),
        },
        "run": {"name": ["case"]},
        "scenario": {"clusters": ["27"], "opts": [""], "sector_opts": ["none"]},
        "solving": {},
    }
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(
        "from pathlib import Path\n"
        f"config.update({config!r})\n"
        "def resources(path): return 'resources/{run}/'+path\n"
        "def logs(path): return 'logs/{run}/'+path\n"
        f"def scripts(path): return {str(ROOT / 'scripts')!r}+'/'+path\n"
        "def fixed_neighbor_capacity_manifest(w): return []\n"
        "def config_provider(*keys,default=None):\n"
        " def get(w):\n"
        "  v=config\n"
        "  for key in keys:\n"
        "   if key not in v: return default\n"
        "   v=v[key]\n"
        "  return v\n"
        " return get\n"
        "def weather_solve_input(w):\n"
        " suffix='dispatch' if config['controlled_weather']['mode']=='dispatch' else 'final'\n"
        " return resources('networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_'+suffix+'.nc')\n"
        "wildcard_constraints:\n"
        " opts=r'[-+a-zA-Z0-9.]*',\n"
        " planning_horizons=r'[0-9]{4}',\n"
        "rule base_network:\n"
        " output: 'resources/{run}/networks/base.nc', 'resources/{run}/regions_onshore.geojson'\n"
        " shell: 'echo native base'\n"
        "rule build_renewable_profiles:\n"
        " input: 'resources/{run}/regions_onshore.geojson'\n"
        " output: 'resources/{run}/profile_{clusters}_{technology}.nc', 'resources/{run}/regions_by_class_{clusters}_{technology}.geojson'\n"
        " shell: 'echo costly profiles'\n"
        "rule prepare_final:\n"
        " input: 'resources/{run}/networks/base.nc', 'resources/{run}/profile_{clusters}_solar.nc', 'reference_2030.nc'\n"
        " output: 'resources/{run}/networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_final.nc'\n"
        " shell: 'echo prepared'\n"
        f"include: {str(ROOT / 'rules/pypsa-de/controlled_weather.smk')!r}\n",
        encoding="utf-8",
    )
    return snakefile


def actual_dag(snakefile, targets):
    with SnakemakeApi(OutputSettings(skip_plugin_handlers=True)) as api:
        workflow = api.workflow(
            ResourceSettings(cores=1),
            snakefile=snakefile,
            workdir=snakefile.parent,
            storage_settings=StorageSettings(
                shared_fs_usage=SharedFSUsage.all() - {SharedFSUsage.SOURCE_CACHE}
            ),
            workflow_settings=WorkflowSettings(
                runtime_source_cache_path=snakefile.parent / ".sourcecache"
            ),
        )
        workflow.dag(DAGSettings(targets=set(targets), print_dag_as="dot")).printdag()
        return list(workflow._workflow.dag.jobs)


@pytest.mark.parametrize("dispatch", [False, True])
def test_preflight_rule_loads_and_builds_a_real_dag(tmp_path, dispatch):
    snakefile = fixture_workflow(tmp_path, dispatch=dispatch)
    target = (
        "weather_dispatch_preflight" if dispatch else "controlled_weather_preflight"
    )
    jobs = actual_dag(snakefile, [target])
    names = {job.rule.name for job in jobs}
    assert "check_controlled_weather_network" in names
    assert ("prepare_dispatch_weather_network" in names) is dispatch
    assert not any(name.startswith("solve_") for name in names)


def test_cached_regions_cannot_conflict_with_uncached_base_network_output(tmp_path):
    snakefile = fixture_workflow(
        tmp_path,
        [
            "regions_onshore.geojson",
            "networks/base.nc",
            "profile_27_solar.nc",
            "regions_by_class_27_solar.geojson",
        ],
    )
    jobs = actual_dag(
        snakefile,
        ["resources/case/networks/base.nc", "resources/case/profile_27_solar.nc"],
    )
    assert sum(job.rule.name == "base_network" for job in jobs) == 1
    assert not any(job.rule.name == "build_renewable_profiles" for job in jobs)
    copies = {
        str(output)
        for job in jobs
        if job.rule.name == "reuse_controlled_weather_resource"
        for output in job.output
    }
    assert "resources/case/profile_27_solar.nc" in copies
    assert "resources/case/regions_onshore.geojson" not in copies
    assert not any("/networks/" in output for output in copies)


def test_incomplete_profile_output_group_uses_native_producer(tmp_path):
    snakefile = fixture_workflow(tmp_path, ["profile_27_solar.nc"])
    jobs = actual_dag(snakefile, ["resources/case/profile_27_solar.nc"])
    assert any(job.rule.name == "build_renewable_profiles" for job in jobs)
    assert not any(job.rule.name == "reuse_controlled_weather_resource" for job in jobs)


def test_overlapping_output_groups_are_filtered_to_a_fixed_point(tmp_path):
    for name in ["a.csv", "b.csv"]:
        (tmp_path / name).touch()
    rules = [
        SimpleNamespace(output=["out/a.csv", "out/b.csv"]),
        SimpleNamespace(output=["out/b.csv", "out/networks/base.nc"]),
    ]
    assert reusable_weather_resources(tmp_path, rules, "out/") == []


def test_cache_excludes_investments_audits_and_unrecognised_files(tmp_path):
    names = [
        "weather.csv",
        "unrecognised.csv",
        "networks/base.nc",
        "preflight/networks/audit.nc",
        "controlled_weather_preflight_27__none_2035.json",
        "dispatch_weather_fleet_27__none_2035.csv",
        "fixed_neighbor_static_capex.csv",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    rules = [SimpleNamespace(output=["out/" + name]) for name in names[::2]]
    assert reusable_weather_resources(tmp_path, rules, "out/") == ["weather.csv"]


def test_scenario_producer_is_unambiguous_in_the_real_snakefile(tmp_path):
    # Load the actual top-level rule block and the actual DE producer block.
    original = (ROOT / "Snakefile").read_text(encoding="utf-8")
    create = original[
        original.index('if not config.get("controlled_weather"') : original.index(
            "rule purge:"
        )
    ]
    modifications = (ROOT / "rules/pypsa-de/modifications.smk").read_text(
        encoding="utf-8"
    )
    build = modifications[
        modifications.index("rule build_scenarios:") : modifications.index(
            "rule build_exogenous_mobility_data:"
        )
    ]
    (tmp_path / "data").mkdir()
    (tmp_path / "data/ariadne_database.csv").touch()
    (tmp_path / "manual.yaml").touch()
    config = {
        "controlled_weather": {"enable": True},
        "run": {
            "name": ["case"],
            "scenarios": {"file": "automated.yaml", "manual_file": "manual.yaml"},
        },
        "pypsa-de": {"leitmodelle": []},
    }
    snakefile = tmp_path / "Snakefile"
    snakefile.write_text(
        f"config.update({config!r})\ndef scripts(path): return {str(ROOT / 'scripts')!r}+'/'+path\n"
        + create
        + build,
        encoding="utf-8",
    )
    jobs = actual_dag(snakefile, ["automated.yaml"])
    assert [job.rule.name for job in jobs] == ["build_scenarios"]


spec = importlib.util.spec_from_file_location(
    "weather_slurm_status", ROOT / "scripts/pypsa-de/slurm_job_status.py"
)
SLURM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SLURM)


@pytest.mark.parametrize(
    "state,exit_code,expected",
    [
        ("COMPLETED", "0:0", "success"),
        ("COMPLETED", "1:0", "failed"),
        ("TIMEOUT", "0:0", "failed"),
        ("OUT_OF_MEMORY", "0:125", "failed"),
        ("CANCELLED by 123", "0:15", "failed"),
        ("RUNNING", "0:0", "running"),
        ("PENDING", "0:0", "running"),
    ],
)
def test_slurm_terminal_status_is_not_inferred_from_controller_liveness(
    state, exit_code, expected
):
    assert (
        SLURM.classify_accounting(
            f"42|{state}|{exit_code}\n42.extern|COMPLETED|0:0", "42"
        )
        == expected
    )


def test_slurm_missing_accounting_is_not_success():
    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="")

    assert SLURM.job_status("42", runner) == "running"


def test_documented_cluster_resources_use_explicit_hours():
    text = (ROOT / "doc/weather_cluster_runs.md").read_text(encoding="utf-8")
    for value in [
        "runtime=4h",
        "runtime=8h",
        "runtime=24h",
        "scontrol show job",
        "sacct -X",
        "--cluster-generic-status-cmd",
    ]:
        assert value in text
