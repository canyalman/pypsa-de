# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Focused tests for tolerance-aware fixed-neighbor capacity bands."""

import importlib.util
import sys
import types
from pathlib import Path

import pypsa
import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "pypsa-de" / "additional_functionality.py"
)
SPEC = importlib.util.spec_from_file_location("additional_functionality", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
PREPARE_SECTOR_NETWORK = "scripts.prepare_sector_network"
ORIGINAL_PREPARE_SECTOR_NETWORK = sys.modules.get(PREPARE_SECTOR_NETWORK)
stub = types.ModuleType(PREPARE_SECTOR_NETWORK)


def determine_emission_sectors_stub(*args, **kwargs):
    return None


stub.determine_emission_sectors = determine_emission_sectors_stub
sys.modules[PREPARE_SECTOR_NETWORK] = stub
try:
    SPEC.loader.exec_module(MODULE)
finally:
    if ORIGINAL_PREPARE_SECTOR_NETWORK is None:
        del sys.modules[PREPARE_SECTOR_NETWORK]
    else:
        sys.modules[PREPARE_SECTOR_NETWORK] = ORIGINAL_PREPARE_SECTOR_NETWORK


def build_generator_model(
    p_nom_min=0.0,
    p_nom_max=152.9999821847514,
    demand=0.0,
):
    n = pypsa.Network()
    n.add("Bus", "DE bus", country="DE")
    n.add("Bus", "BE bus", country="BE")
    n.add(
        "Generator",
        "BE2 0 0 onwind-2035",
        bus="BE bus",
        p_nom_extendable=True,
        p_nom_min=p_nom_min,
        p_nom_max=p_nom_max,
        capital_cost=1.0,
    )
    if demand:
        n.add("Load", "BE load", bus="BE bus", p_set=demand)
    n.set_snapshots([0])
    n.optimize.create_model()
    return n


def write_manifest(path, nominal):
    path.write_text(
        "year,component,asset,nominal\n"
        f"2035,Generator,BE2 0 0 onwind-2035,{nominal}\n",
        encoding="utf-8",
    )


def test_small_upper_bound_mismatch_uses_intersecting_reference_band(
    tmp_path, caplog
):
    asset = "BE2 0 0 onwind-2035"
    reference_target = 153.0012777901072
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, reference_target)
    n = build_generator_model()

    MODULE.add_fixed_neighbor_capacity_constraints(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        capacity_tolerance=0.01,
    )

    fixed_lower = n.model.constraints["FixedNeighbor-Generator-p_nom-lower"]
    fixed_upper = n.model.constraints["FixedNeighbor-Generator-p_nom-upper"]
    upper = n.model.constraints["Generator-ext-p_nom-upper"]
    expected_tolerance = 0.01
    assert fixed_lower.rhs.to_pandas().at[asset] == pytest.approx(
        reference_target - expected_tolerance
    )
    assert fixed_upper.rhs.to_pandas().at[asset] == pytest.approx(
        152.9999821847514
    )
    assert upper.rhs.to_pandas().at[asset] == 152.9999821847514
    assert "delta=0.001295605" in caplog.text

    status, condition = n.model.solve(solver_name="highs")
    assert status == "ok"
    assert condition == "optimal"


def test_small_lower_bound_mismatch_uses_intersecting_reference_band(tmp_path):
    asset = "BE2 0 0 onwind-2035"
    reference_target = 0.999
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, reference_target)
    n = build_generator_model(p_nom_min=1.0, p_nom_max=10.0)

    MODULE.add_fixed_neighbor_capacity_constraints(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        capacity_tolerance=0.01,
    )

    fixed_lower = n.model.constraints["FixedNeighbor-Generator-p_nom-lower"]
    fixed_upper = n.model.constraints["FixedNeighbor-Generator-p_nom-upper"]
    lower = n.model.constraints["Generator-ext-p_nom-lower"]
    assert fixed_lower.rhs.to_pandas().at[asset] == 1.0
    assert fixed_upper.rhs.to_pandas().at[asset] == pytest.approx(1.009)
    assert lower.rhs.to_pandas().at[asset] == 1.0


def test_rejects_reference_band_without_bound_overlap(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, 153.02)
    n = build_generator_model()

    with pytest.raises(ValueError, match="reference bands do not intersect"):
        MODULE.add_fixed_neighbor_capacity_constraints(
            n,
            investment_year=2035,
            manifest_path=manifest,
            domestic_country="DE",
            capacity_tolerance=0.01,
        )

    assert "FixedNeighbor-Generator-p_nom-lower" not in n.model.constraints
    assert "FixedNeighbor-Generator-p_nom-upper" not in n.model.constraints


def test_tolerance_does_not_scale_with_large_reference_capacity(tmp_path):
    asset = "BE2 0 0 onwind-2035"
    reference_target = 42000.0
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, reference_target)
    n = build_generator_model(p_nom_max=50000.0)

    MODULE.add_fixed_neighbor_capacity_constraints(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        capacity_tolerance=0.01,
    )

    fixed_lower = n.model.constraints["FixedNeighbor-Generator-p_nom-lower"]
    fixed_upper = n.model.constraints["FixedNeighbor-Generator-p_nom-upper"]
    assert fixed_lower.rhs.to_pandas().at[asset] == pytest.approx(41999.99)
    assert fixed_upper.rhs.to_pandas().at[asset] == pytest.approx(42000.01)


def test_narrow_band_absorbs_tiny_reference_dispatch_residual(tmp_path):
    asset = "BE2 0 0 onwind-2035"
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, 0.0)
    n = build_generator_model(p_nom_max=10.0, demand=0.005)

    MODULE.add_fixed_neighbor_capacity_constraints(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        capacity_tolerance=0.01,
    )

    status, condition = n.model.solve(solver_name="highs")
    assert status == "ok"
    assert condition == "optimal"
    assert n.model.solution["Generator-p_nom"].to_pandas().at[asset] == pytest.approx(
        0.005
    )


def test_all_fixed_neighbor_nominal_component_groups_are_covered():
    assert set(MODULE.FIXED_NEIGHBOR_COMPONENTS) == {
        "Generator",
        "Link",
        "Store",
        "StorageUnit",
        "Line",
        "Transformer",
    }


def build_two_generator_relation_model():
    n = pypsa.Network()
    n.add("Bus", "BE bus", country="BE")
    for generator in ("BE generator 1", "BE generator 2"):
        n.add(
            "Generator",
            generator,
            bus="BE bus",
            p_nom_extendable=True,
            capital_cost=1.0,
        )
    n.set_snapshots([0])
    n.optimize.create_model()
    p_nom = n.model["Generator-p_nom"]
    relation = p_nom.loc[["BE generator 1"]].sum()
    relation -= p_nom.loc[["BE generator 2"]].sum()
    n.model.add_constraints(relation == 0, name="test-fixed-capacity-relation")
    return n


def write_relation_manifest(path, second_nominal):
    path.write_text(
        "year,component,asset,nominal\n"
        "2035,Generator,BE generator 1,1.0\n"
        f"2035,Generator,BE generator 2,{second_nominal}\n",
        encoding="utf-8",
    )


def test_does_not_mask_other_capacity_relations(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_relation_manifest(manifest, 1.000000000001)
    n = build_two_generator_relation_model()

    MODULE.add_fixed_neighbor_capacity_constraints(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        capacity_tolerance=0.01,
    )

    constraint = n.model.constraints["test-fixed-capacity-relation"]
    assert constraint.labels.item() >= 0

    status, condition = n.model.solve(solver_name="highs")
    assert status == "ok"
    assert condition == "optimal"
