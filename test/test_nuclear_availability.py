# SPDX-License-Identifier: MIT

"""Nuclear availability reference tests without downloads or optimisation."""

import logging
from pathlib import Path

import pandas as pd
import pypsa
import pytest
import yaml
from pydantic import ValidationError
from snakemake.utils import update_config

from scripts.add_electricity import attach_conventional_generators
from scripts.lib.validation.config.conventional import ConventionalConfig
from scripts.prepare_sector_network import remove_elec_base_techs


def attach_from_table(
    tmp_path, weather_year, config, table, carrier="nuclear", use_input=True
):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range(f"{weather_year}-01-01", periods=2, freq="h"))
    n.add("Bus", ["FR0", "SE0"], country=["FR", "SE"], carrier="AC")
    ppl = pd.DataFrame(
        {
            "carrier": carrier,
            "bus": ["FR0", "SE0"],
            "p_nom": [1000.0, 500.0],
            "efficiency": 0.33,
            "marginal_cost": 10.0,
            "capital_cost": 100.0,
            "investment": 1000.0,
            "build_year": 1980,
            "lifetime": 60.0,
        },
        index=[f"FR0 {carrier}", f"SE0 {carrier}"],
    )
    costs = pd.DataFrame({"CO2 intensity": [0.0]}, index=[carrier])
    source = tmp_path / "availability.csv"
    table.to_csv(source)
    attach_conventional_generators(
        n,
        costs,
        ppl,
        conventional_carriers=[carrier],
        extendable_carriers={"Generator": []},
        renewable_carriers=set(),
        conventional_params=config,
        conventional_inputs=(
            {f"conventional_{carrier}_p_max_pu": str(source)} if use_input else {}
        ),
    )
    return n


@pytest.fixture
def availability():
    # Deliberately no 1996 column; other years must not be silently selected.
    return pd.DataFrame(
        {"2010": [0.70, 0.75], "2013": [0.80, 0.85], "2023": [0.90, 0.95]},
        index=["FR", "SE"],
    )


def nuclear_config(reference_year=None):
    return ConventionalConfig(
        nuclear_availability_year=reference_year,
        nuclear={"p_max_pu": "data/nuclear_p_max_pu.csv"},
    ).model_dump()


@pytest.mark.parametrize("weather_year", [1996, 2010, 2013, 2023])
def test_fixed_2013_availability_on_real_generators(
    tmp_path, availability, caplog, weather_year
):
    with caplog.at_level(logging.INFO, logger="scripts.add_electricity"):
        n = attach_from_table(
            tmp_path, weather_year, nuclear_config(2013), availability
        )

    expected = availability["2013"].set_axis(n.generators.index)
    pd.testing.assert_series_equal(n.generators.p_max_pu, expected, check_names=False)
    assert n.snapshots[0].year == weather_year
    assert f"availability year 2013 (weather year {weather_year})" in caplog.text
    assert n.generators.p_nom.tolist() == [1000.0, 500.0]
    assert n.generators.build_year.tolist() == [1980, 1980]


def test_missing_1996_without_reference_is_an_explicit_error(tmp_path, availability):
    with pytest.raises(ValueError, match=r"nuclear.p_max_pu has no data for year 1996"):
        attach_from_table(tmp_path, 1996, nuclear_config(), availability)


def test_missing_fixed_2013_reference_does_not_choose_another_year(
    tmp_path, availability
):
    with pytest.raises(ValueError, match=r"nuclear.p_max_pu has no data for year 2013"):
        attach_from_table(
            tmp_path, 2023, nuclear_config(2013), availability.drop(columns="2013")
        )


def test_2013_base_case_is_unchanged(tmp_path, availability):
    original = attach_from_table(tmp_path, 2013, nuclear_config(), availability)
    fixed = attach_from_table(tmp_path, 2013, nuclear_config(2013), availability)
    pd.testing.assert_frame_equal(original.generators, fixed.generators)


@pytest.mark.parametrize("weather_year", [2010, 2023])
def test_unconfigured_reference_still_uses_snapshot_year(
    tmp_path, availability, weather_year
):
    n = attach_from_table(tmp_path, weather_year, nuclear_config(), availability)
    expected = availability[str(weather_year)].set_axis(n.generators.index)
    pd.testing.assert_series_equal(n.generators.p_max_pu, expected, check_names=False)


def test_nuclear_reference_does_not_change_other_carrier_years(tmp_path, availability):
    config = nuclear_config(2013)
    config["CCGT"] = {"p_max_pu": "data/availability.csv"}
    n = attach_from_table(tmp_path, 2023, config, availability, carrier="CCGT")
    expected = availability["2023"].set_axis(n.generators.index)
    pd.testing.assert_series_equal(n.generators.p_max_pu, expected, check_names=False)


def test_explicit_reference_rejects_non_year_columns(tmp_path):
    table = pd.DataFrame({"availability": [0.8, 0.85]}, index=["FR", "SE"])
    with pytest.raises(ValueError, match="requires integer year columns"):
        attach_from_table(tmp_path, 1996, nuclear_config(2013), table)


def test_non_year_table_without_reference_remains_supported(tmp_path):
    table = pd.DataFrame({"availability": [0.8, 0.85]}, index=["FR", "SE"])
    n = attach_from_table(tmp_path, 1996, nuclear_config(), table)
    assert n.generators.p_max_pu.tolist() == [0.8, 0.85]


def test_scalar_availability_without_csv_remains_supported(tmp_path, availability):
    config = nuclear_config(2013)
    config["nuclear"]["p_max_pu"] = 0.88
    n = attach_from_table(tmp_path, 1996, config, availability, use_input=False)
    assert n.generators.p_max_pu.tolist() == [0.88, 0.88]


def test_intermediate_nuclear_availability_is_not_retained_in_sector_model(
    tmp_path, availability
):
    root = Path(__file__).resolve().parents[1]
    config = {}
    for filename in ["config.default.yaml", "config.de.yaml"]:
        update_config(
            config,
            yaml.safe_load((root / "config" / filename).read_text(encoding="utf-8")),
        )
    assert "nuclear" not in config["pypsa_eur"]["Generator"]
    networks = [
        attach_from_table(tmp_path, 2010, nuclear_config(year), availability)
        for year in [None, 2013]
    ]
    assert not networks[0].generators.p_max_pu.equals(networks[1].generators.p_max_pu)
    for n in networks:
        n.add("Carrier", "onwind")
        n.add(
            "Generator",
            "FR0 onwind",
            bus="FR0",
            carrier="onwind",
            p_nom=100,
            p_max_pu=0.4,
        )
        remove_elec_base_techs(n, config["pypsa_eur"])
        assert not n.generators.carrier.eq("nuclear").any()
        assert "FR0 onwind" in n.generators.index
    pd.testing.assert_frame_equal(networks[0].generators, networks[1].generators)
    pd.testing.assert_frame_equal(networks[0].carriers, networks[1].carriers)


@pytest.mark.parametrize("invalid_year", [True, False, "2013", 2013.5, 0, -1])
def test_reference_year_has_explicit_integer_validation(invalid_year):
    with pytest.raises(ValidationError):
        ConventionalConfig(nuclear_availability_year=invalid_year)
