# SPDX-License-Identifier: MIT

"""Run the real post-solve script tail, not just the finalizer in isolation."""

import ast
import copy
import runpy

import numpy as np
import pypsa
import pytest
import yaml
from snakemake.io import Namedlist, Wildcards

from scripts.dispatch_weather import build_dispatch_weather, validate_dispatch_weather
from test.test_dispatch_weather import ROOT, SCENARIO, fixture_pair, settings
from test.test_fixed_co2_price import MODULE, mock_snakemake


@pytest.fixture(scope="module")
def publication_tail():
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
    # Includes metadata replacement, the configured finalizer and both exports.
    code = compile(
        ast.Module(body=main.body[boundary + 1 :], type_ignores=[]), str(path), "exec"
    )
    namespace = runpy.run_path(str(path), run_name="solve_network_publication_test")
    return code, namespace


def solved_dispatch(year, tmp_path):
    reference, weather = fixture_pair(year)
    n, _ = build_dispatch_weather(reference, weather, settings())
    workflow = mock_snakemake()
    workflow.config["controlled_weather"] = {
        "enable": True,
        "mode": "dispatch",
        **settings(),
    }
    workflow.params.custom_extra_functionality = str(
        ROOT / "scripts/pypsa-de/additional_functionality.py"
    )
    workflow.wildcards = Wildcards(
        fromdict={
            "planning_horizons": "2035",
            "clusters": "27",
            "run": SCENARIO,
            "opts": "",
            "sector_opts": "none",
        }
    )
    workflow.output = Namedlist(
        fromdict={
            "network": str(tmp_path / "result.nc"),
            "config": str(tmp_path / "config.yaml"),
        }
    )
    MODULE.prepare_network_before_model(n, workflow)
    status, condition = n.optimize(solver_name="highs", log_to_console=False)
    assert (status, condition) == ("ok", "optimal")
    return n, workflow


def publish(n, workflow, publication_tail):
    code, namespace = publication_tail
    exec(code, {**namespace, "n": n, "snakemake": workflow})


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_metadata_refresh_finalizer_and_export_keep_dispatch_audit(
    year, tmp_path, publication_tail
):
    n, workflow = solved_dispatch(year, tmp_path)
    audit = copy.deepcopy(n.meta["dispatch_weather_prepared"])
    n.meta["old_result_should_not_survive"] = True
    n.meta["wildcards"] = {"planning_horizons": "2020"}
    publish(n, workflow, publication_tail)
    restored = pypsa.Network(workflow.output.network)
    metadata = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    for result in [n.meta, restored.meta, metadata]:
        assert result["dispatch_weather_prepared"] == audit
        assert result["dispatch_weather_result"]["weather_year"] == year
        assert result["wildcards"]["planning_horizons"] == "2035"
        assert result["controlled_weather"] == workflow.config["controlled_weather"]
        assert "old_result_should_not_survive" not in result
        assert result["dispatch_weather_result"]["unserved_energy_penalty_eur"] > 0
        assert (
            "objective_including_reference_capex_excluding_co2_and_unserved_penalty_eur"
            in result["dispatch_weather_result"]
        )
    validate_dispatch_weather(restored)
    np.testing.assert_allclose(restored.generators_t.p, n.generators_t.p)


def test_missing_audit_is_not_replaced_by_an_empty_fallback(tmp_path, publication_tail):
    n, workflow = solved_dispatch(1996, tmp_path)
    del n.meta["dispatch_weather_prepared"]
    with pytest.raises(KeyError, match="dispatch_weather_prepared"):
        publish(n, workflow, publication_tail)
    assert not (tmp_path / "result.nc").exists()
    assert not (tmp_path / "config.yaml").exists()


def test_non_dispatch_metadata_and_fixed_neighbor_costs_are_unchanged(
    tmp_path, publication_tail
):
    n, workflow = solved_dispatch(2010, tmp_path)
    del n.meta["dispatch_weather_prepared"]
    del workflow.config["controlled_weather"]
    n._fixed_neighbor_static_summary = {"removed_annualized_capex_eur": 123.0}
    publish(n, workflow, publication_tail)
    restored = pypsa.Network(workflow.output.network)
    assert "dispatch_weather_prepared" not in restored.meta
    assert "dispatch_weather_result" not in restored.meta
    assert restored.meta["fixed_neighbor_static"][
        "objective_including_fixed_nonde_capex_eur"
    ] == pytest.approx(n.objective + 123)
    assert "fixed_co2_price" in restored.meta
