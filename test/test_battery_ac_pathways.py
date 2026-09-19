# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""AC BESS targets reuse the existing aggregate four-hour duration constraint."""

import pandas as pd
import pypsa
import pytest

from scripts.solve_network import add_battery_constraints
from test.test_fixed_neighbor_constraints import MODULE

BASIS = {"Link": {"battery discharger": {"DE": {2035: "output"}}}}


def battery_network():
    n = pypsa.Network()
    n.add("Carrier", "AC")
    n.add("Carrier", "battery")
    n.add("Carrier", "home battery")
    for carrier in ("battery charger", "battery discharger", "home battery discharger"):
        n.add("Carrier", carrier)
    for country in ("DE", "PL"):
        n.add("Bus", country + " bus", country=country, carrier="AC")
        n.add("Bus", country + " battery", country=country, carrier="battery")
    n.add("Generator", "DE supply", bus="DE bus", p_nom=10, marginal_cost=1)
    # A Store can outlive its original 2025 power electronics.
    n.add(
        "Store",
        "DE battery-2025",
        bus="DE battery",
        carrier="battery",
        e_nom=8,
        e_cyclic=True,
    )
    n.add(
        "Store",
        "DE battery-2035",
        bus="DE battery",
        carrier="battery",
        e_nom_extendable=True,
        e_cyclic=True,
        capital_cost=1,
    )
    n.add(
        "Link",
        "DE battery discharger-2030",
        carrier="battery discharger",
        bus0="DE battery",
        bus1="DE bus",
        p_nom=1,
        efficiency=0.9,
    )
    n.add(
        "Link",
        "DE battery charger-2035",
        carrier="battery charger",
        bus0="DE bus",
        bus1="DE battery",
        p_nom_extendable=True,
        efficiency=0.98,
        capital_cost=1,
    )
    n.add(
        "Link",
        "DE battery discharger-2035",
        carrier="battery discharger",
        bus0="DE battery",
        bus1="DE bus",
        p_nom_extendable=True,
        efficiency=0.98,
        capital_cost=1,
    )
    n.add(
        "Store",
        "PL battery-2030",
        bus="PL battery",
        carrier="battery",
        e_nom=40,
        e_cyclic=True,
    )
    n.add(
        "Link",
        "PL battery discharger-2030",
        carrier="battery discharger",
        bus0="PL battery",
        bus1="PL bus",
        p_nom=10,
        efficiency=0.8,
    )
    n.add("Bus", "DE home battery", country="DE", carrier="home battery")
    n.add(
        "Store",
        "DE home battery-2030",
        bus="DE home battery",
        carrier="home battery",
        e_nom=10,
        e_cyclic=True,
    )
    n.add(
        "Link",
        "DE home battery discharger-2030",
        carrier="home battery discharger",
        bus0="DE home battery",
        bus1="DE bus",
        p_nom=5,
        efficiency=0.95,
    )
    n.config = {
        "battery_storage_duration": {"exclude": {"DE": [2025]}},
        "solving": {
            "constraints": {
                "fixed_neighbor_capacities": {"enable": True, "domestic_country": "DE"},
            }
        },
    }
    n.optimize.create_model()
    return n


@pytest.mark.parametrize("target_gw", [15, 35, 55])
def test_ac_power_and_four_hour_rated_energy_with_mixed_vintages(target_gw):
    n = battery_network()
    foreign_before = (
        n.links.loc[["PL battery discharger-2030"]].drop(columns="p_nom_opt").copy()
    )
    home_before = (
        n.stores.loc[["DE home battery-2030"]].drop(columns="e_nom_opt").copy()
    )
    limits = {"Link": {"battery discharger": {"DE": {2035: target_gw}}}}
    for sense in ("minimum", "maximum"):
        MODULE.add_capacity_limits(n, 2035, limits, sense, capacity_limit_basis=BASIS)
    add_battery_constraints(n, planning_horizons=2035)
    assert "Battery-storage_duration" in n.model.constraints
    status, condition = n.optimize.solve_model(solver_name="highs")
    assert (status, condition) == ("ok", "optimal")

    dischargers = n.links.loc[
        ["DE battery discharger-2030", "DE battery discharger-2035"]
    ]
    input_power = dischargers.p_nom_opt.sum()
    output_power = (dischargers.p_nom_opt * dischargers.efficiency).sum()
    energy = n.stores.loc[["DE battery-2025", "DE battery-2035"], "e_nom_opt"].sum()
    eta_fleet = output_power / input_power
    assert output_power == pytest.approx(target_gw * 1000)
    assert dischargers.at["DE battery discharger-2035", "p_nom_opt"] == pytest.approx(
        (target_gw * 1000 - 0.9) / 0.98
    )
    assert energy == pytest.approx(4 * input_power)
    assert energy * eta_fleet == pytest.approx(4 * target_gw * 1000)
    assert energy > 4 * target_gw * 1000
    assert n.links.at["PL battery discharger-2030", "p_nom_opt"] == pytest.approx(10)
    assert n.stores.at["DE home battery-2030", "e_nom_opt"] == pytest.approx(10)
    pd.testing.assert_frame_equal(
        n.links.loc[foreign_before.index, foreign_before.columns], foreign_before
    )
    pd.testing.assert_frame_equal(
        n.stores.loc[home_before.index, home_before.columns], home_before
    )


@pytest.mark.parametrize("year,basis", [(2030, BASIS), (2035, None)])
def test_other_years_and_legacy_configuration_keep_nominal_basis(year, basis):
    n = battery_network()
    limits = {"Link": {"battery discharger": {"DE": {year: 0.035}}}}
    for sense in ("minimum", "maximum"):
        MODULE.add_capacity_limits(n, year, limits, sense, capacity_limit_basis=basis)
    add_battery_constraints(n, planning_horizons=2035)
    assert n.optimize.solve_model(solver_name="highs") == ("ok", "optimal")
    dischargers = n.links.loc[
        ["DE battery discharger-2030", "DE battery discharger-2035"]
    ]
    assert dischargers.p_nom_opt.sum() == pytest.approx(35)
    assert (dischargers.p_nom_opt * dischargers.efficiency).sum() < 35


@pytest.mark.parametrize("efficiency", [0.0, -0.5, float("nan")])
def test_invalid_efficiency_rejected(efficiency):
    n = battery_network()
    n.links.loc["DE battery discharger-2035", "efficiency"] = efficiency
    limits = {"Link": {"battery discharger": {"DE": {2035: 35}}}}
    with pytest.raises(ValueError, match="Invalid output efficiency"):
        MODULE.add_capacity_limits(n, 2035, limits, capacity_limit_basis=BASIS)


def test_time_varying_output_rating_is_not_silently_approximated():
    n = battery_network()
    n.links_t.efficiency["DE battery discharger-2035"] = 0.9
    limits = {"Link": {"battery discharger": {"DE": {2035: 35}}}}
    with pytest.raises(ValueError, match="time-varying efficiency"):
        MODULE.add_capacity_limits(n, 2035, limits, capacity_limit_basis=BASIS)


def test_existing_ac_capacity_above_target_is_not_silently_accepted():
    n = battery_network()
    limits = {"Link": {"battery discharger": {"DE": {2035: 0.0005}}}}
    with pytest.raises(ValueError, match="exceeds"):
        MODULE.add_capacity_limits(n, 2035, limits, capacity_limit_basis=BASIS)


def test_fixed_only_fleet_below_ac_target_is_rejected():
    n = battery_network()
    n.links.loc["DE battery discharger-2035", "p_nom_extendable"] = False
    limits = {"Link": {"battery discharger": {"DE": {2035: 35}}}}
    with pytest.raises(ValueError, match="without extendable Links"):
        MODULE.add_capacity_limits(
            n, 2035, limits, "minimum", capacity_limit_basis=BASIS
        )
