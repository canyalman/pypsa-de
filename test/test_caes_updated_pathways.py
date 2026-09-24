# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""CAES pathway alignment with the current 3H BESS-only experiments."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest
import yaml
from snakemake.utils import update_config

from scripts.lib.validation.config import validate_config
from scripts.solve_network import add_battery_constraints
from test.test_fixed_neighbor_constraints import MODULE

PRICE = 308.2489253762274
BASIS = {"Link": {"battery discharger": {"DE": {2035: "output"}}}}


def test_config_and_scenarios():
    root = Path(__file__).parents[1]
    defaults = yaml.safe_load(
        (root / "config/config.default.yaml").read_text(encoding="utf-8")
    )
    overrides = yaml.safe_load(
        (root / "config/config.de.yaml").read_text(encoding="utf-8")
    )
    scenarios = yaml.safe_load(
        (root / "config/scenarios.manual.yaml").read_text(encoding="utf-8")
    )
    update_config(defaults, overrides)
    assert validate_config(defaults).sector.bev_dsm_availability == {
        2030: 0.2,
        2035: 0.35,
    }
    names = [f"KN2045_Mix_FixedRenewables_BESS_{v}GW" for v in (15, 35, 55)]
    assert overrides["run"]["name"] == ["KN2045_Mix_FixedRenewables", *names]
    assert overrides["run"]["prefix"] == "caes_rte69_fixed_nonde_3h"
    assert overrides["clustering"]["temporal"]["resolution_sector"] == "3H"
    fixed = overrides["solving"]["constraints"]["fixed_neighbor_capacities"]
    assert fixed["reuse_reference_horizons"] == [2030]
    assert fixed["reuse_reference_runs"] == overrides["run"]["name"]
    assert "20260914" in fixed["capacity_manifest"]
    for year, path in fixed["reference_networks"].items():
        assert "fixedgas_2026-09-14/" in path
        assert path.endswith(f"/base_s_27__none_{year}.nc")
    assert overrides["costs"]["emission_prices"]["co2"] == {
        2025: 0.0,
        2030: 0.0,
        2035: PRICE,
    }
    assert overrides["solving"]["constraints"]["fixed_co2_price"] == {
        "enable": True,
        "planning_horizons": [2035],
    }
    caes = overrides["electricity"]["storage_options"]["aCAES_RESC_continuous"]
    assert caes["round_trip_efficiency"] == 0.69
    assert caes["geological_output_capacity_twh"] == 7.6
    for power, name in zip((15, 35, 55), names):
        scenario = scenarios[name]
        limits = scenario["solving"]["constraints"]
        assert limits["capacity_limit_basis"] == BASIS
        assert scenario["battery_storage_duration"]["exclude"]["DE"] == [2025]
        for bound in ("limits_capacity_min", "limits_capacity_max"):
            assert limits[bound]["Link"]["battery discharger"]["DE"][2035] == power
            assert 2035 not in limits[bound]["Store"]["battery"]["DE"]


def battery_network():
    n = pypsa.Network()
    for carrier in ("AC", "battery", "battery charger", "battery discharger"):
        n.add("Carrier", carrier)
    n.add("Bus", "DE AC", country="DE", carrier="AC")
    n.add("Bus", "DE battery", country="DE", carrier="battery")
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


def co2_snakemake(year):
    return SimpleNamespace(
        config={
            "costs": {
                "emission_prices": {
                    "enable": True,
                    "dynamic": False,
                    "co2": {2025: 0.0, 2030: 0.0, 2035: PRICE},
                }
            }
        },
        params=SimpleNamespace(
            solving={
                "constraints": {
                    "fixed_co2_price": {
                        "enable": True,
                        "planning_horizons": [2035],
                    },
                    "fixed_neighbor_capacities": {"enable": False},
                }
            }
        ),
        wildcards=SimpleNamespace(planning_horizons=str(year)),
    )


def co2_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=2, freq="3h"))
    n.snapshot_weightings.loc[:, :] = 3.0
    n.add("Carrier", "co2")
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Store", "co2 atmosphere", bus="co2 atmosphere", e_nom=1e6)
    n.add("GlobalConstraint", "CO2Limit", type="co2_atmosphere", constant=1.0)
    n.add("GlobalConstraint", "co2_limit-DE", type="", constant=0.5)
    n.add("GlobalConstraint", "co2_sequestration_limit", type="", constant=-1e8)
    return n


@pytest.mark.parametrize("year", [2025, 2030])
def test_earlier_carbon_caps_unchanged(year):
    n = co2_network()
    before = n.global_constraints.copy()
    MODULE.prepare_network_before_model(n, co2_snakemake(year))
    pd.testing.assert_frame_equal(n.global_constraints, before)


def test_2035_carbon_price_removes_only_emissions_caps():
    n = co2_network()
    MODULE.prepare_network_before_model(n, co2_snakemake(2035))
    assert n.global_constraints.index.tolist() == ["co2_sequestration_limit"]
    assert n.stores.at["co2 atmosphere", "marginal_cost"] == -PRICE


def test_2035_rejects_reintroduced_carbon_cap():
    n = co2_network()
    smk = co2_snakemake(2035)
    MODULE.prepare_network_before_model(n, smk)
    n.optimize.create_model()
    n.model.add_constraints(
        n.model["Store-e"].sum() <= 1, name="GlobalConstraint-CO2Limit"
    )
    with pytest.raises(ValueError, match="still has emissions caps"):
        MODULE.validate_fixed_co2_price_model(n, smk)


def test_2035_reports_carbon_payments_and_fixed_foreign_capex_separately():
    n = co2_network()
    smk = co2_snakemake(2035)
    n.stores_t.p["co2 atmosphere"] = [-0.4, -0.6]
    n._objective = 100.0 + 3.0 * PRICE
    n.meta["fixed_neighbor_static"] = {"removed_annualized_capex_eur": 123.0}
    MODULE.finalize_network_after_solve(n, smk)
    result = n.meta["fixed_co2_price"]
    assert result["net_atmospheric_emissions_tco2"] == pytest.approx(3.0)
    assert result["co2_payments_in_objective_eur"] == pytest.approx(3.0 * PRICE)
    assert result["objective_including_fixed_nonde_capex_excluding_co2_payments_eur"] == pytest.approx(223.0)
