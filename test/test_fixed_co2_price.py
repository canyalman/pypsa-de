# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Price-only carbon policy, horizon isolation, and cost-accounting regression tests."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest
import yaml
from snakemake.utils import update_config

from test.test_fixed_neighbor_constraints import MODULE

PRICE = 308.2489253762274


def test_active_pathways_use_ac_targets_and_preserve_reference_history():
    root = Path(__file__).parents[1]
    config = yaml.safe_load(
        (root / "config/config.de.yaml").read_text(encoding="utf-8")
    )
    scenarios = yaml.safe_load(
        (root / "config/scenarios.manual.yaml").read_text(encoding="utf-8")
    )
    names = [f"KN2045_Mix_FixedRenewables_BESS_{power}GW" for power in (15, 35, 55)]
    assert config["run"]["name"] == names
    assert config["run"]["prefix"] == "bess_fixed_nonde_co2_price"
    assert config["sector"]["bev_dsm_availability"] == {2030: 0.2, 2035: 0.35}
    assert config["sector"]["v2g"] is False
    assert config["scenario"]["planning_horizons"] == [2025, 2030, 2035]
    assert config["clustering"]["temporal"]["resolution_sector"] == "3H"
    assert config["costs"]["emission_prices"]["co2"][2035] == PRICE
    fixed = config["solving"]["constraints"]["fixed_neighbor_capacities"]
    assert fixed["formulation"] == "static"
    assert fixed["reuse_reference_horizons"] == [2030]
    assert fixed["reuse_reference_runs"] == names
    assert "20260914" in fixed["capacity_manifest"]
    for year, path in fixed["reference_networks"].items():
        assert "fixedgas_2026-09-14/" in path
        assert path.endswith(f"/base_s_27__none_{year}.nc")
    for power, name in zip((15, 35, 55), names):
        merged = yaml.safe_load(yaml.safe_dump(config))
        update_config(merged, scenarios[name])
        limits = merged["solving"]["constraints"]
        assert limits["capacity_limit_basis"] == {
            "Link": {"battery discharger": {"DE": {2035: "output"}}}
        }
        assert merged["battery_storage_duration"]["exclude"]["DE"] == [2025]
        for bound in ("limits_capacity_min", "limits_capacity_max"):
            assert limits[bound]["Link"]["battery discharger"]["DE"][2035] == power
            assert 2035 not in limits[bound]["Store"]["battery"]["DE"]
            assert limits[bound]["Store"]["battery"]["DE"][2025] == 4.058
            assert limits[bound]["Link"]["battery discharger"]["DE"][2025] == 2.705
            assert 2030 not in limits[bound]["Store"]["battery"]["DE"]
            assert 2030 not in limits[bound]["Link"]["battery discharger"]["DE"]
            assert limits[bound]["Store"]["home battery"]["DE"][2035] == 24.29
            assert (
                limits[bound]["Link"]["home battery discharger"]["DE"][2035] == 11.477
            )


def mock_snakemake(year=2035):
    constraints = {
        "fixed_co2_price": {"enable": True, "planning_horizons": [2035]},
        "fixed_neighbor_capacities": {"enable": False},
        "co2_budget_national": {"DE": {2030: 0.364, 2035: 0.216}},
        "limits_capacity_min": {},
        "limits_capacity_max": {},
        "limits_power_max": {},
        "limits_volume_min": {},
        "limits_volume_max": {},
    }
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
        params=SimpleNamespace(solving={"constraints": constraints}),
        wildcards=SimpleNamespace(planning_horizons=str(year), clusters="27"),
    )


def co2_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=2, freq="3h"))
    n.snapshot_weightings.loc[:, :] = 3.0
    n.add("Carrier", "AC")
    n.add("Carrier", "gas")
    n.add("Carrier", "co2", co2_emissions=-1.0)
    n.add("Bus", "DE electricity", carrier="AC")
    n.add("Bus", "EU gas", carrier="gas")
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Generator", "gas source", bus="EU gas", p_nom=100, marginal_cost=20)
    n.add(
        "Link",
        "gas plant",
        bus0="EU gas",
        bus1="DE electricity",
        bus2="co2 atmosphere",
        p_nom=100,
        efficiency=0.5,
        efficiency2=0.2,
    )
    n.add("Load", "electric demand", bus="DE electricity", p_set=1.0)
    n.add(
        "Store",
        "co2 atmosphere",
        bus="co2 atmosphere",
        carrier="co2",
        e_nom=1e6,
        e_min_pu=-1.0,
    )
    n.add("GlobalConstraint", "CO2Limit", type="co2_atmosphere", constant=1.0)
    n.add("GlobalConstraint", "co2_limit-DE", type="", constant=0.5)
    n.add("GlobalConstraint", "co2_sequestration_limit", type="", constant=-1e8)
    return n


@pytest.mark.parametrize("year", [2025, 2030])
def test_previous_horizon_caps_and_costs_unchanged(year):
    n = co2_network()
    before = n.global_constraints.copy()
    MODULE.prepare_network_before_model(n, mock_snakemake(year))
    pd.testing.assert_frame_equal(n.global_constraints, before)
    assert n.stores.at["co2 atmosphere", "marginal_cost"] == 0.0


def test_disabled_policy_keeps_original_behaviour():
    n = co2_network()
    smk = mock_snakemake()
    smk.params.solving["constraints"].pop("fixed_co2_price")
    MODULE.prepare_network_before_model(n, smk)
    assert "CO2Limit" in n.global_constraints.index
    assert n.stores.at["co2 atmosphere", "marginal_cost"] == 0.0


def test_price_only_removes_emissions_caps_not_sequestration_and_is_idempotent():
    n = co2_network()
    before = n.global_constraints.loc["co2_sequestration_limit"].copy()
    smk = mock_snakemake()
    MODULE.prepare_network_before_model(n, smk)
    MODULE.prepare_network_before_model(n, smk)
    assert n.global_constraints.index.tolist() == ["co2_sequestration_limit"]
    pd.testing.assert_series_equal(
        n.global_constraints.loc["co2_sequestration_limit"], before
    )
    assert n.stores.at["co2 atmosphere", "marginal_cost"] == -PRICE
    assert n.generators.at["gas source", "marginal_cost"] == 20.0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, None])
def test_reject_invalid_carbon_price(invalid):
    smk = mock_snakemake()
    smk.config["costs"]["emission_prices"]["co2"][2035] = invalid
    with pytest.raises(ValueError, match="Missing or invalid"):
        MODULE.prepare_network_before_model(co2_network(), smk)


def test_scalar_price_is_rejected_to_prevent_double_charging():
    smk = mock_snakemake()
    smk.config["costs"]["emission_prices"]["co2"] = PRICE
    with pytest.raises(ValueError, match="year mapping"):
        MODULE.prepare_network_before_model(co2_network(), smk)


def test_conflicting_time_varying_carbon_cost_is_rejected():
    n = co2_network()
    n.stores_t.marginal_cost["co2 atmosphere"] = [-PRICE, -PRICE / 2]
    with pytest.raises(ValueError, match="Time-varying"):
        MODULE.prepare_network_before_model(n, mock_snakemake())


def test_national_cap_from_generated_scenario_is_not_readded(monkeypatch):
    n = co2_network()
    smk = mock_snakemake()
    MODULE.prepare_network_before_model(n, smk)
    n.optimize.create_model()
    for function in (
        "add_capacity_limits",
        "add_power_limits",
        "h2_import_limits",
        "electricity_import_limits",
        "h2_production_limits",
        "add_h2_derivate_limit",
        "force_boiler_profiles_existing_per_boiler",
    ):
        monkeypatch.setattr(MODULE, function, lambda *args, **kwargs: None)

    def forbidden(*args, **kwargs):
        pytest.fail("National CO2 cap was reintroduced in a price-only horizon.")

    monkeypatch.setattr(MODULE, "add_national_co2_budgets", forbidden)
    MODULE.additional_functionality(n, n.snapshots, smk)
    assert not len(MODULE.emissions_budget_constraints(n))


def test_pre_solve_validation_detects_reintroduced_model_cap():
    n = co2_network()
    smk = mock_snakemake()
    MODULE.prepare_network_before_model(n, smk)
    n.optimize.create_model()
    n.model.add_constraints(
        n.model["Store-e"].sum() <= 1, name="GlobalConstraint-CO2Limit"
    )
    with pytest.raises(ValueError, match="still has emissions caps"):
        MODULE.validate_fixed_co2_price_model(n, smk)


def test_weighted_emissions_priced_once_and_static_capex_reported_separately():
    n = co2_network()
    smk = mock_snakemake()
    MODULE.prepare_network_before_model(n, smk)
    status, condition = n.optimize(solver_name="highs")
    assert (status, condition) == ("ok", "optimal")
    MODULE.validate_fixed_co2_price_model(n, smk)
    assert n.stores_t.e["co2 atmosphere"].iloc[-1] == pytest.approx(2.4)
    assert n.objective == pytest.approx(240 + 2.4 * PRICE)
    n.meta["fixed_neighbor_static"] = {"removed_annualized_capex_eur": 123.0}
    MODULE.finalize_network_after_solve(n, smk)
    summary = n.meta["fixed_co2_price"]
    assert summary["net_atmospheric_emissions_tco2"] == pytest.approx(2.4)
    assert summary["co2_payments_in_objective_eur"] == pytest.approx(2.4 * PRICE)
    assert summary["objective_excluding_co2_payments_eur"] == pytest.approx(240.0)
    assert summary[
        "objective_including_fixed_nonde_capex_excluding_co2_payments_eur"
    ] == pytest.approx(363.0)


def test_carbon_removal_gets_credit_with_same_sign_convention():
    n = co2_network()
    n.stores_t.p["co2 atmosphere"] = [0.4, 0.6]
    n._objective = 100.0
    MODULE.finalize_network_after_solve(n, mock_snakemake())
    summary = n.meta["fixed_co2_price"]
    assert summary["net_atmospheric_emissions_tco2"] == pytest.approx(-3.0)
    assert summary["co2_payments_in_objective_eur"] == pytest.approx(-3.0 * PRICE)
