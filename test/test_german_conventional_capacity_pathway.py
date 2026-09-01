# SPDX-License-Identifier: MIT

import sys

import pandas as pd
import pypsa

sys.path.append("./scripts/pypsa-de")

from modify_prenetwork import (  # noqa: E402
    GERMAN_CONVENTIONAL_CARRIERS,
    _small_mastr_chp_capacity,
    allocate_german_lignite_to_sites,
    apply_german_conventional_capacity_pathway,
)


def add_output_link(n, name, bus, carrier, output_mw, efficiency=0.5):
    fuel_bus = f"{name} fuel"
    if fuel_bus not in n.buses.index:
        n.add("Bus", fuel_bus)
    n.add(
        "Link",
        name,
        bus0=fuel_bus,
        bus1=bus,
        carrier=carrier,
        p_nom=output_mw / efficiency,
        efficiency=efficiency,
    )


def output_capacity(n, carriers):
    links = n.links.loc[n.links.carrier.isin(carriers)]
    return links.eval("p_nom * efficiency").sum()


def test_small_mastr_chp_capacity_uses_site_threshold(tmp_path):
    chp = pd.DataFrame(
        {
            "Name": ["small", "large", "large", "small-coal"],
            "Postleitzahl": ["1", "2", "2", "3"],
            "Fueltype": ["Natural Gas", "Natural Gas", "Natural Gas", "Coal"],
            "Capacity": [4.0, 6.0, 6.0, 2.0],
            "DateIn": [2020, 2020, 2020, 2020],
            "DateOut": [pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
    path = tmp_path / "german_chp.csv"
    chp.to_csv(path, index=False)

    result = _small_mastr_chp_capacity(path, 2025, 10.0)

    assert result["gas"] == 4.0
    assert result["hard_coal"] == 2.0
    assert result["lignite"] == 0.0


def test_2025_market_fleet_adds_small_mastr_chp(tmp_path):
    n = pypsa.Network()
    n.add("Bus", "DE0", carrier="AC", country="DE", x=10.0, y=51.0)
    add_output_link(n, "gas", "DE0", "CCGT", 100.0)
    add_output_link(n, "gas CHP", "DE0", "urban central gas CHP", 20.0)
    add_output_link(n, "coal", "DE0", "coal", 100.0)
    add_output_link(n, "lignite", "DE0", "lignite", 100.0)

    chp = pd.DataFrame(
        {
            "Name": ["small-gas", "small-coal", "small-lignite"],
            "Postleitzahl": ["1", "2", "3"],
            "Fueltype": ["Natural Gas", "Coal", "Lignite"],
            "Capacity": [4.0, 2.0, 1.0],
            "DateIn": [2020, 2020, 2020],
            "DateOut": [pd.NA, pd.NA, pd.NA],
        }
    )
    path = tmp_path / "german_chp.csv"
    chp.to_csv(path, index=False)
    pathway = {
        "enable": True,
        "market_fleet_2025_mw": {
            "gas": 200.0,
            "hard_coal": 150.0,
            "lignite": 120.0,
        },
        "add_mastr_chp_below_mw": 10.0,
    }

    apply_german_conventional_capacity_pathway(n, pathway, 2025, path)

    assert output_capacity(n, GERMAN_CONVENTIONAL_CARRIERS["gas"]) == 204.0
    assert n.links.loc["gas CHP"].p_nom * n.links.loc["gas CHP"].efficiency == 20.0
    assert output_capacity(n, GERMAN_CONVENTIONAL_CARRIERS["hard_coal"]) == 152.0
    assert output_capacity(n, GERMAN_CONVENTIONAL_CARRIERS["lignite"]) == 121.0


def test_2025_gas_capacity_can_be_fixed(tmp_path):
    n = pypsa.Network()
    n.add("Bus", "DE0", carrier="AC", country="DE", x=10.0, y=51.0)
    add_output_link(n, "gas", "DE0", "CCGT", 100.0)
    n.links.loc["gas", "p_nom_extendable"] = True

    chp = pd.DataFrame(
        columns=["Name", "Postleitzahl", "Fueltype", "Capacity", "DateIn", "DateOut"]
    )
    path = tmp_path / "german_chp.csv"
    chp.to_csv(path, index=False)
    pathway = {
        "enable": True,
        "fix_gas_capacity_2025": True,
        "market_fleet_2025_mw": {
            "gas": 100.0,
            "hard_coal": 0.0,
            "lignite": 0.0,
        },
    }

    apply_german_conventional_capacity_pathway(n, pathway, 2025, path)

    assert not n.links.loc["gas", "p_nom_extendable"]
    assert n.links.loc["gas", "p_nom"] * n.links.loc["gas", "efficiency"] == 100.0


def test_2025_gas_fix_does_not_apply_in_later_years():
    n = pypsa.Network()
    n.add("Bus", "DE0", carrier="AC", country="DE", x=10.0, y=51.0)
    add_output_link(n, "gas", "DE0", "CCGT", 100.0)
    n.links.loc["gas", "p_nom_extendable"] = True

    apply_german_conventional_capacity_pathway(
        n, {"enable": True, "fix_gas_capacity_2025": True}, 2030
    )

    assert n.links.loc["gas", "p_nom_extendable"]


def test_lignite_is_allocated_only_to_survivor_buses():
    n = pypsa.Network()
    for bus, x in {"DE-west": 6.0, "DE-east-1": 12.0, "DE-east-2": 14.0}.items():
        n.add("Bus", bus, carrier="AC", country="DE", x=x, y=51.0)
        add_output_link(n, f"{bus} lignite", bus, "lignite", 100.0)

    sites = {
        "east-1": {"lat": 51.0, "lon": 12.0, "capacity_mw": 100.0},
        "east-2": {"lat": 51.0, "lon": 14.0, "capacity_mw": 200.0},
    }
    allocate_german_lignite_to_sites(n, 300.0, sites)

    output = n.links.eval("p_nom * efficiency")
    assert output["DE-west lignite"] == 0.0
    assert output["DE-east-1 lignite"] == 100.0
    assert output["DE-east-2 lignite"] == 200.0
