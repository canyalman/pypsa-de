# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Focused tests for fixed-neighbor numerical bound handling."""

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


def build_generator_model():
    n = pypsa.Network()
    n.add("Bus", "DE bus", country="DE")
    n.add("Bus", "BE bus", country="BE")
    n.add(
        "Generator",
        "BE2 0 0 onwind-2035",
        bus="BE bus",
        p_nom_extendable=True,
        p_nom_min=0.0,
        p_nom_max=152.9999821847514,
        capital_cost=1.0,
    )
    n.set_snapshots([0])
    n.optimize.create_model()
    return n


def write_manifest(path, nominal):
    path.write_text(
        "year,component,asset,nominal\n"
        f"2035,Generator,BE2 0 0 onwind-2035,{nominal}\n",
        encoding="utf-8",
    )


def test_clips_small_fixed_neighbor_upper_bound_violation(tmp_path, caplog):
    asset = "BE2 0 0 onwind-2035"
    upper_bound = 152.9999821847514
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, 153.0012777901072)
    n = build_generator_model()

    MODULE.add_fixed_neighbor_capacity_constraints(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        bound_clip_tolerance=0.01,
    )

    constraint = n.model.constraints["FixedNeighbor-Generator-p_nom"]
    assert constraint.rhs.to_pandas().at[asset] == upper_bound
    assert "delta=0.001295605" in caplog.text


def test_rejects_large_fixed_neighbor_bound_violation(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, 153.02)
    n = build_generator_model()

    with pytest.raises(ValueError, match="exceed their Linopy nominal bounds"):
        MODULE.add_fixed_neighbor_capacity_constraints(
            n,
            investment_year=2035,
            manifest_path=manifest,
            domestic_country="DE",
            bound_clip_tolerance=0.01,
        )

    assert "FixedNeighbor-Generator-p_nom" not in n.model.constraints


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


def test_masks_small_fully_fixed_capacity_relation_residual(tmp_path, caplog):
    manifest = tmp_path / "manifest.csv"
    write_relation_manifest(manifest, 1.00002)
    n = build_two_generator_relation_model()

    MODULE.add_fixed_neighbor_capacity_constraints(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        bound_clip_tolerance=0.01,
    )

    constraint = n.model.constraints["test-fixed-capacity-relation"]
    assert constraint.labels.item() == -1
    assert "test-fixed-capacity-relation" in caplog.text
    assert "delta=1.999999" in caplog.text


def test_rejects_large_fully_fixed_capacity_relation_residual(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_relation_manifest(manifest, 1.02)
    n = build_two_generator_relation_model()

    with pytest.raises(ValueError, match="fully fixed capacity constraint residuals"):
        MODULE.add_fixed_neighbor_capacity_constraints(
            n,
            investment_year=2035,
            manifest_path=manifest,
            domestic_country="DE",
            bound_clip_tolerance=0.01,
        )

    constraint = n.model.constraints["test-fixed-capacity-relation"]
    assert constraint.labels.item() >= 0
