# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Focused tests for tolerance-aware fixed-neighbor capacity bands."""

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
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


def build_all_component_static_network():
    n = pypsa.Network()
    n.add("Bus", "DE bus", country="DE", v_nom=380.0)
    n.add("Bus", "BE bus", country="BE", v_nom=380.0)
    n.add(
        "Generator",
        "DE generator",
        bus="DE bus",
        p_nom=1.0,
        p_nom_extendable=True,
        capital_cost=2.0,
    )
    n.add(
        "Generator",
        "BE generator",
        bus="BE bus",
        p_nom=1.0,
        p_nom_extendable=True,
        capital_cost=2.0,
    )
    n.add(
        "Link",
        "BE-DE link",
        bus0="BE bus",
        bus1="DE bus",
        p_nom=1.0,
        p_nom_extendable=True,
        capital_cost=2.0,
    )
    n.add(
        "Store",
        "BE store",
        bus="BE bus",
        e_nom=1.0,
        e_nom_extendable=True,
        capital_cost=2.0,
    )
    n.add(
        "StorageUnit",
        "BE storage unit",
        bus="BE bus",
        p_nom=1.0,
        p_nom_extendable=True,
        capital_cost=2.0,
    )
    n.add(
        "Line",
        "BE-DE line",
        bus0="BE bus",
        bus1="DE bus",
        x=0.1,
        r=0.01,
        s_nom=1.0,
        s_nom_extendable=True,
        capital_cost=2.0,
    )
    n.add(
        "Transformer",
        "BE-DE transformer",
        bus0="BE bus",
        bus1="DE bus",
        x=0.1,
        s_nom=1.0,
        s_nom_extendable=True,
        capital_cost=2.0,
    )
    return n


def write_all_component_manifest(path):
    pd.DataFrame(
        [
            (2035, "Generator", "BE generator", "p", 5.0),
            (2035, "Link", "BE-DE link", "p", 5.0),
            (2035, "Store", "BE store", "e", 5.0),
            (2035, "StorageUnit", "BE storage unit", "p", 5.0),
            (2035, "Line", "BE-DE line", "s", 5.0),
            (2035, "Transformer", "BE-DE transformer", "s", 5.0),
        ],
        columns=[
            "year",
            "component",
            "asset",
            "nominal_attribute",
            "nominal",
        ],
    ).to_csv(path, index=False)


def test_static_formulation_materializes_all_components_and_leaves_de_free(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_all_component_manifest(manifest)
    n = build_all_component_static_network()

    ledger = MODULE.materialize_fixed_neighbor_capacities(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        capacity_tolerance=0.012,
    )

    for component, (
        list_name,
        nominal_attr,
        _,
    ) in MODULE.FIXED_NEIGHBOR_COMPONENTS.items():
        row = ledger.loc[ledger["component"].eq(component)].iloc[0]
        static = getattr(n, list_name)
        assert static.at[row.asset, f"{nominal_attr}_nom"] == pytest.approx(5.0)
        assert not static.at[row.asset, f"{nominal_attr}_nom_extendable"]

    assert n.generators.at["DE generator", "p_nom"] == pytest.approx(1.0)
    assert n.generators.at["DE generator", "p_nom_extendable"]
    assert len(ledger) == 6
    assert ledger["removed_annualized_capex"].sum() == pytest.approx(48.0)


def build_static_generator_network(p_nom_max, p_nom=0.0, extendable=True):
    n = pypsa.Network()
    n.add("Bus", "DE bus", country="DE")
    n.add("Bus", "BE bus", country="BE")
    n.add(
        "Generator",
        "BE generator",
        bus="BE bus",
        p_nom=p_nom,
        p_nom_extendable=extendable,
        p_nom_max=p_nom_max,
        capital_cost=3.0,
    )
    return n


def write_static_generator_manifest(path, nominal):
    path.write_text(
        "year,component,asset,nominal_attribute,nominal\n"
        f"2035,Generator,BE generator,p,{nominal}\n",
        encoding="utf-8",
    )


def test_static_formulation_clips_tiny_native_bound_residual(tmp_path):
    native_upper = 152.9999821847514
    reference = 153.0012777901072
    manifest = tmp_path / "manifest.csv"
    write_static_generator_manifest(manifest, reference)
    n = build_static_generator_network(native_upper)

    ledger = MODULE.materialize_fixed_neighbor_capacities(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        capacity_tolerance=0.012,
    )

    assert n.generators.at["BE generator", "p_nom"] == pytest.approx(native_upper)
    assert not n.generators.at["BE generator", "p_nom_extendable"]
    assert ledger.at[0, "reference_nominal"] == pytest.approx(reference)
    assert ledger.at[0, "applied_nominal"] == pytest.approx(native_upper)
    assert ledger.at[0, "bound_adjustment"] == pytest.approx(native_upper - reference)


def test_static_formulation_rejects_material_native_bound_gap_without_mutation(
    tmp_path,
):
    manifest = tmp_path / "manifest.csv"
    write_static_generator_manifest(manifest, 153.02)
    n = build_static_generator_network(152.9999821847514)

    with pytest.raises(ValueError, match="exceed native nominal bounds"):
        MODULE.materialize_fixed_neighbor_capacities(
            n,
            investment_year=2035,
            manifest_path=manifest,
            domestic_country="DE",
            capacity_tolerance=0.012,
        )

    assert n.generators.at["BE generator", "p_nom"] == pytest.approx(0.0)
    assert n.generators.at["BE generator", "p_nom_extendable"]


def test_static_formulation_canonicalizes_tiny_fixed_brownfield_residual(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_static_generator_manifest(manifest, 100.0)
    n = build_static_generator_network(
        p_nom_max=200.0, p_nom=100.010342, extendable=False
    )

    ledger = MODULE.materialize_fixed_neighbor_capacities(
        n,
        investment_year=2035,
        manifest_path=manifest,
        domestic_country="DE",
        capacity_tolerance=0.012,
    )

    assert n.generators.at["BE generator", "p_nom"] == pytest.approx(100.0)
    assert not ledger.at[0, "was_extendable"]
    assert ledger.at[0, "removed_annualized_capex"] == pytest.approx(0.0)


def test_static_pre_model_hook_writes_capex_ledger_and_summary(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_static_generator_manifest(manifest, 5.0)
    n = build_static_generator_network(p_nom_max=10.0, p_nom=1.0)
    output_network = tmp_path / "results" / "run" / "networks" / "network.nc"
    snakemake = types.SimpleNamespace(
        params=types.SimpleNamespace(
            solving={
                "constraints": {
                    "fixed_neighbor_capacities": {
                        "enable": True,
                        "formulation": "static",
                        "domestic_country": "DE",
                        "strict_asset_match": True,
                        "capacity_tolerance": 0.012,
                    }
                }
            }
        ),
        wildcards=types.SimpleNamespace(planning_horizons="2035"),
        input=types.SimpleNamespace(fixed_neighbor_capacities=str(manifest)),
        output=types.SimpleNamespace(network=str(output_network)),
    )

    MODULE.prepare_network_before_model(n, snakemake)

    ledger_path = (
        output_network.parent.parent
        / "costs"
        / "fixed_neighbor_static_capex_network.csv"
    )
    assert ledger_path.exists()
    ledger = pd.read_csv(ledger_path)
    assert ledger.at[0, "removed_annualized_capex"] == pytest.approx(12.0)
    assert n._fixed_neighbor_static_summary["materialized_assets"] == 1
    assert n._fixed_neighbor_static_summary[
        "removed_annualized_capex_eur"
    ] == pytest.approx(12.0)
