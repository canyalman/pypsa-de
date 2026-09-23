# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""German BESS pathway targets are measured at AC output."""

from pathlib import Path

import pypsa
import pytest
import yaml

from scripts.solve_network import add_battery_constraints
from test.test_fixed_neighbor_constraints import MODULE

BASIS = {"Link": {"battery discharger": {"DE": {2035: "output"}}}}


def test_active_iron_air_pathways_match_ac_targets_and_reference_history():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "config/config.de.yaml").read_text(encoding="utf-8"))
    scenarios = yaml.safe_load(
        (root / "config/scenarios.manual.yaml").read_text(encoding="utf-8")
    )
    names = [f"KN2045_Mix_FixedRenewables_BESS_{v}GW" for v in (15, 35, 55)]
    assert config["run"]["name"] == ["KN2045_Mix_FixedRenewables", *names]
    fixed = config["solving"]["constraints"]["fixed_neighbor_capacities"]
    assert fixed["reuse_reference_horizons"] == [2030]
    assert fixed["reuse_reference_runs"] == config["run"]["name"]
    assert config["clustering"]["temporal"]["resolution_sector"] == "3H"
    for target, name in zip((15, 35, 55), names):
        scenario = scenarios[name]
        constraints = scenario["solving"]["constraints"]
        assert constraints["capacity_limit_basis"] == BASIS
        assert scenario["battery_storage_duration"]["exclude"]["DE"] == [2025]
        for sense in ("limits_capacity_min", "limits_capacity_max"):
            limits = constraints[sense]
            assert limits["Link"]["battery discharger"]["DE"][2035] == target
            assert 2035 not in limits["Store"]["battery"]["DE"]


def battery_network():
    n = pypsa.Network()
    for carrier in ("AC", "battery", "battery charger", "battery discharger"):
        n.add("Carrier", carrier)
    n.add("Bus", "DE AC", country="DE", carrier="AC")
    n.add("Bus", "DE battery", country="DE", carrier="battery")
    n.add("Bus", "PL AC", country="PL", carrier="AC")
    n.add("Bus", "PL battery", country="PL", carrier="battery")
    n.add("Generator", "DE supply", bus="DE AC", p_nom=1, marginal_cost=1)
    n.add("Store", "DE battery-2025", bus="DE battery", carrier="battery", e_nom=8)
    n.add(
        "Store",
        "DE battery-2035",
        bus="DE battery",
        carrier="battery",
        e_nom_extendable=True,
        capital_cost=1,
    )
    n.add(
        "Link",
        "DE battery discharger-2030",
        bus0="DE battery",
        bus1="DE AC",
        carrier="battery discharger",
        p_nom=1,
        efficiency=0.9,
    )
    n.add(
        "Link",
        "DE battery charger-2035",
        bus0="DE AC",
        bus1="DE battery",
        carrier="battery charger",
        p_nom_extendable=True,
        efficiency=0.98,
        capital_cost=1,
    )
    n.add(
        "Link",
        "DE battery discharger-2035",
        bus0="DE battery",
        bus1="DE AC",
        carrier="battery discharger",
        p_nom_extendable=True,
        efficiency=0.98,
        capital_cost=1,
    )
    n.add("Store", "PL battery", bus="PL battery", carrier="battery", e_nom=40)
    n.add(
        "Link",
        "PL battery discharger",
        bus0="PL battery",
        bus1="PL AC",
        carrier="battery discharger",
        p_nom=10,
        efficiency=0.8,
    )
    n.config = {
        "battery_storage_duration": {"exclude": {"DE": [2025]}},
        "solving": {
            "constraints": {
                "fixed_neighbor_capacities": {"enable": True, "domestic_country": "DE"}
            }
        },
    }
    n.optimize.create_model()
    return n


@pytest.mark.parametrize("target_gw", [15, 35, 55])
def test_ac_target_and_four_hour_energy(target_gw):
    n = battery_network()
    limits = {"Link": {"battery discharger": {"DE": {2035: target_gw}}}}
    for sense in ("minimum", "maximum"):
        MODULE.add_capacity_limits(n, 2035, limits, sense, capacity_limit_basis=BASIS)
    add_battery_constraints(n, planning_horizons=2035)

    assert n.optimize.solve_model(solver_name="highs") == ("ok", "optimal")
    dischargers = n.links.loc[
        ["DE battery discharger-2030", "DE battery discharger-2035"]
    ]
    input_power = dischargers.p_nom_opt.sum()
    output_power = (dischargers.p_nom_opt * dischargers.efficiency).sum()
    energy = n.stores.loc[["DE battery-2025", "DE battery-2035"], "e_nom_opt"].sum()
    assert output_power == pytest.approx(target_gw * 1000)
    assert energy == pytest.approx(4 * input_power)
    assert n.links.at["PL battery discharger", "p_nom_opt"] == pytest.approx(10)


def test_invalid_discharge_efficiency_is_rejected():
    n = battery_network()
    n.links.loc["DE battery discharger-2035", "efficiency"] = 0
    limits = {"Link": {"battery discharger": {"DE": {2035: 35}}}}
    with pytest.raises(ValueError, match="Invalid output efficiency"):
        MODULE.add_capacity_limits(n, 2035, limits, capacity_limit_basis=BASIS)
