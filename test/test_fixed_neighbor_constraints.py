# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Focused tests for fixed-neighbor numerical bound handling."""

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
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


def test_clips_small_fixed_neighbor_upper_bound_violation():
    asset = "BE2 0 0 onwind-2035"
    targets = pd.Series({asset: 153.0012777901072})
    lower_bounds = pd.Series({asset: 0.0})
    upper_bounds = pd.Series({asset: 152.9999821847514})

    clipped = MODULE._clip_fixed_neighbor_targets_to_bounds(
        targets,
        lower_bounds,
        upper_bounds,
        tolerance=0.01,
        component="Generator",
    )

    assert clipped.at[asset] == upper_bounds.at[asset]


def test_rejects_large_fixed_neighbor_bound_violation():
    asset = "BE2 0 0 onwind-2035"
    targets = pd.Series({asset: 153.02})
    lower_bounds = pd.Series({asset: 0.0})
    upper_bounds = pd.Series({asset: 152.9999821847514})

    with pytest.raises(ValueError, match="exceed their nominal bounds"):
        MODULE._clip_fixed_neighbor_targets_to_bounds(
            targets,
            lower_bounds,
            upper_bounds,
            tolerance=0.01,
            component="Generator",
        )
