# SPDX-License-Identifier: MIT

"""Match native sector-first heat names to aggregated brownfield assets."""

import numpy as np
import pandas as pd
import pytest

from scripts.controlled_weather import asset_stem, refresh_brownfield_weather
from test.test_controlled_weather import network, settings


@pytest.mark.parametrize("system", ["rural", "urban decentral"])
@pytest.mark.parametrize("sector", ["residential", "services"])
@pytest.mark.parametrize("technology", ["air heat pump", "ground heat pump"])
def test_heat_names_match_in_both_sector_orders(system, sector, technology):
    expected = f"AT2 0 {system} {technology}"
    for suffix in ["", "-2020", "-2030"]:
        assert asset_stem(f"AT2 0 {sector} {system} {technology}{suffix}") == expected
        assert asset_stem(f"AT2 0 {system} {sector} {technology}{suffix}") == expected
        assert asset_stem(expected + suffix) == expected


def heat_profile_pair(year, system, technology, sector_first):
    previous, current = network(2013), network(year)
    for n in [previous, current]:
        n.add("Bus", "AT2 0", country="AT", carrier="AC")
    previous.meta = {"wildcards": {"planning_horizons": "2030"}}
    carrier = f"{system} {technology}"
    asset = f"AT2 0 {carrier}-2020"
    previous.add(
        "Link",
        asset,
        bus0="AT2 0",
        bus1="AT2 0",
        carrier=carrier,
        build_year=2020,
        lifetime=20,
        p_nom=5,
        efficiency=[2, 2, 2],
        capital_cost=123,
    )
    profiles = {}
    for vintage, values in [(2025, [3, 4, 5]), (2030, [6, 7, 8])]:
        fresh = network(year)
        fresh.add("Bus", "AT2 0", country="AT", carrier="AC")
        for sector in ["residential", "services"]:
            name = f"{sector} {system}" if sector_first else f"{system} {sector}"
            fresh.add(
                "Link",
                f"AT2 0 {name} {technology}",
                bus0="AT2 0",
                bus1="AT2 0",
                carrier=f"{name} {technology}",
                efficiency=values,
            )
        profiles[vintage] = fresh
    return previous, current, profiles, asset


@pytest.mark.parametrize("year", [1996, 2010, 2023])
@pytest.mark.parametrize(
    "system,technology",
    [("rural", "ground heat pump"), ("urban decentral", "air heat pump")],
)
@pytest.mark.parametrize("sector_first", [True, False])
def test_real_heat_names_refresh_weather_without_changing_investments(
    year, system, technology, sector_first
):
    previous, current, profiles, asset = heat_profile_pair(
        year, system, technology, sector_first
    )
    before = previous.links.copy(deep=True)
    refresh_brownfield_weather(previous, current, profiles, settings())
    # The 2020 asset uses the earliest fresh vintage, not the 2030 COP or old COP.
    np.testing.assert_array_equal(previous.links_t.efficiency[asset], [3, 4, 5])
    assert previous.snapshots.equals(current.snapshots)
    pd.testing.assert_frame_equal(previous.links, before)


def test_sector_profiles_must_agree_before_aggregated_heat_matching():
    previous, current, profiles, _ = heat_profile_pair(
        1996, "rural", "ground heat pump", True
    )
    profiles[2025].links_t.efficiency["AT2 0 services rural ground heat pump"] = [
        4,
        5,
        6,
    ]
    with pytest.raises(ValueError, match="Ambiguous fresh profiles"):
        refresh_brownfield_weather(previous, current, profiles, settings())


@pytest.mark.parametrize("mismatch", ["node", "system", "technology"])
def test_unrelated_heat_profiles_are_not_used_as_a_fallback(mismatch):
    previous, current, profiles, _ = heat_profile_pair(
        1996, "rural", "ground heat pump", True
    )
    old = "AT2 0 rural ground heat pump-2020"
    replacement = {
        "node": "AT3 0 rural ground heat pump-2020",
        "system": "AT2 0 urban decentral ground heat pump-2020",
        "technology": "AT2 0 rural air heat pump-2020",
    }[mismatch]
    previous.links.rename(index={old: replacement}, inplace=True)
    previous.links_t.efficiency.rename(columns={old: replacement}, inplace=True)
    with pytest.raises(ValueError, match="Missing fresh Link.efficiency profile"):
        refresh_brownfield_weather(previous, current, profiles, settings())
