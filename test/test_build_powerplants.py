# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>>
#
# SPDX-License-Identifier: MIT

"""
Tests the functionalities of scripts/build_powerplants.py.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append("./scripts")

from build_powerplants import (
    add_custom_powerplants,
    replace_natural_gas_fueltype,
    replace_natural_gas_technology,
    retain_german_lignite_market_fleet,
)

path_cwd = pathlib.Path.cwd()


@pytest.mark.parametrize(
    "query_value,expected",
    [(False, (131, 18)), (True, (137, 18))],
)
def test_add_custom_powerplants(config, query_value, expected):
    """
    Verify what returned by add_custom_powerplants.
    """
    config["electricity"]["custom_powerplants"] = query_value
    custom_powerplants_path = pathlib.Path(
        path_cwd, "test", "test_data", "custom_powerplants_DE.csv"
    )
    ppl_path = pathlib.Path(path_cwd, "test", "test_data", "powerplants_DE.csv")
    ppl_df = pd.read_csv(ppl_path)
    ppl_final = add_custom_powerplants(
        ppl_df,
        custom_powerplants_path,
        config["electricity"]["custom_powerplants"],
    )
    assert ppl_df.shape == (131, 18)
    assert ppl_final.shape == expected


def test_replace_natural_gas_technology():
    """
    Verify what returned by replace_natural_gas_technology.
    """
    input_df = pd.DataFrame(
        {
            "Name": [
                "plant_hydro",
                "plant_ng_1",
                "plant_ng_2",
                "plant_ng_3",
                "plant_ng_4",
            ],
            "Fueltype": [
                "Hydro",
                "Natural Gas",
                "Natural Gas",
                "Natural Gas",
                "Natural Gas",
            ],
            "Technology": [
                "Run-Of-River",
                "Steam Turbine",
                "Combustion Engine",
                "Not Found",
                np.nan,
            ],
        }
    )

    reference_df = pd.DataFrame(
        {
            "Name": [
                "plant_hydro",
                "plant_ng_1",
                "plant_ng_2",
                "plant_ng_3",
                "plant_ng_4",
            ],
            "Fueltype": [
                "Hydro",
                "Natural Gas",
                "Natural Gas",
                "Natural Gas",
                "Natural Gas",
            ],
            "Technology": ["Run-Of-River", "CCGT", "OCGT", "CCGT", "CCGT"],
        }
    )
    modified_df = input_df.assign(Technology=replace_natural_gas_technology)
    comparison_df = modified_df.compare(reference_df)
    assert comparison_df.empty


def test_replace_natural_gas_fueltype():
    """
    Verify what returned by replace_natural_gas_fueltype.
    """
    input_df = pd.DataFrame(
        {
            "Name": [
                "plant_hydro",
                "plant_ng_1",
                "plant_ng_2",
            ],
            "Fueltype": [
                "Hydro",
                "Gas",
                "Natural",
            ],
            "Technology": [
                "Run-Of-River",
                "CCGT",
                "OCGT",
            ],
        }
    )

    reference_df = pd.DataFrame(
        {
            "Name": [
                "plant_hydro",
                "plant_ng_1",
                "plant_ng_2",
            ],
            "Fueltype": [
                "Hydro",
                "Natural Gas",
                "Natural Gas",
            ],
            "Technology": [
                "Run-Of-River",
                "CCGT",
                "OCGT",
            ],
        }
    )
    modified_df = input_df.assign(Fueltype=replace_natural_gas_fueltype)
    comparison_df = modified_df.compare(reference_df)
    assert comparison_df.empty


def test_retain_german_lignite_market_fleet_restores_only_reported_stock():
    source = pd.DataFrame(
        {
            "Name": ["DE large", "DE small", "DE retired", "PL large"],
            "Country": ["DE", "DE", "DE", "PL"],
            "Fueltype": ["Lignite"] * 4,
            "Set": ["CHP"] * 4,
            "Technology": [np.nan] * 4,
            "Capacity": [500.0, 9.0, 400.0, 600.0],
            "DateIn": [1917.0, 2000.0, 2000.0, 2000.0],
            "DateOut": [2035.0, 2035.0, 2025.0, 2035.0],
        }
    )
    filtered = source.loc[source.Country.eq("PL")].copy()
    pathway = {
        "enable": True,
        "market_fleet_2025_mw": {"lignite": 14700},
        "add_mastr_chp_below_mw": 10,
    }

    result = retain_german_lignite_market_fleet(
        filtered, source, pathway, first_grouping_year=1920
    )

    restored = result.loc[result.Country.eq("DE")]
    assert restored.Name.tolist() == ["DE large"]
    assert restored.Set.tolist() == ["PP"]
    assert restored.Technology.tolist() == ["Steam Turbine"]
    assert restored.DateIn.tolist() == [1920.0]
    assert result.loc[result.Country.eq("PL"), "Name"].tolist() == ["PL large"]


def test_retain_german_lignite_market_fleet_is_disabled_without_pathway():
    source = pd.DataFrame(
        {
            "Country": ["DE"],
            "Fueltype": ["Lignite"],
            "Capacity": [500.0],
            "DateIn": [2000.0],
            "DateOut": [2035.0],
        }
    )
    filtered = source.iloc[0:0].copy()

    result = retain_german_lignite_market_fleet(filtered, source, {})

    assert result.empty
