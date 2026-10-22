# SPDX-License-Identifier: MIT

"""Dispatch accounting exceptions, bounded shortages and penalty-free costs."""

import numpy as np
import pandas as pd
import pypsa
import pytest
from pydantic import ValidationError

from scripts.dispatch_weather import (
    SHEDDING,
    add_demand_shedding,
    build_dispatch_weather,
    dispatch_result_summary,
    validate_dispatch_weather,
)
from scripts.lib.validation.config.controlled_weather import ControlledWeatherConfig
from test.test_dispatch_weather import PRICE, fixture_pair, settings
from test.test_fixed_co2_price import MODULE, mock_snakemake


def oil_pair(year=1996):
    reference, weather = fixture_pair(year)
    for n in [reference, weather]:
        n.add("Carrier", ["oil", "land transport oil"])
        n.add("Bus", "EU oil", carrier="oil")
        n.add("Generator", "oil supply", bus="EU oil", p_nom=100, marginal_cost=1)
        for country, capacity in [("DE", 2), ("ES", 4)]:
            bus = f"{country}0 land transport oil"
            n.add("Bus", bus, country=country, carrier="land transport oil")
            profile = [3, 5, 3] if country == "DE" else [5, 8, 5]
            n.add("Load", bus, bus=bus, p_set=profile if n is weather else 1)
            n.add(
                "Link",
                bus,
                bus0="EU oil",
                bus1=bus,
                bus2="co2 atmosphere",
                carrier="land transport oil",
                efficiency2=0.27,
                p_nom_extendable=True,
                p_nom_opt=capacity,
                marginal_cost=0.010734,
            )
        for port in [column for column in n.links if column.startswith("bus")]:
            n.links[port] = n.links[port].fillna("")
    return reference, weather


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_only_zero_capex_accounting_links_are_resized_including_germany(year):
    reference, weather = oil_pair(year)
    original = reference.links.copy()
    inputs = weather.get_switchable_as_dense("Load", "p_set").copy()
    n, ledger = build_dispatch_weather(reference, weather, settings())
    changes = n.meta["dispatch_weather_prepared"]["oil_delivery_accounting_adjustments"]
    assert {row["country"] for row in changes} == {"DE", "ES"}
    assert n.links.at["DE0 land transport oil", "p_nom"] == 5
    assert n.links.at["ES0 land transport oil", "p_nom"] == 8
    assert not n.links.p_nom_extendable.any()
    assert n.links.at["DE0 heat pump-2035", "p_nom"] == 2
    assert n.generators.at["DE0 OCGT-2035", "p_nom"] == 5
    assert n.stores.at["DE0 battery-2035", "e_nom"] == 4
    assert ledger.annualized_capex_eur.sum() == 33
    record = ledger.set_index("asset").loc["ES0 land transport oil"]
    assert record.reference_nominal == 4 and record.fixed_nominal == 8
    assert record.weather_accounting_adjustment_mw == 4
    assert n.links.at["ES0 land transport oil", "marginal_cost"] == 0.010734
    assert n.links.at["ES0 land transport oil", "efficiency2"] == 0.27
    prepared = n.meta["dispatch_weather_prepared"]
    assert (
        prepared["reference_fixed_fleet_fingerprint"]
        != prepared["fixed_fleet_fingerprint"]
    )
    pd.testing.assert_frame_equal(reference.links, original)
    pd.testing.assert_frame_equal(
        n.get_switchable_as_dense("Load", "p_set"), inputs, check_dtype=False
    )
    validate_dispatch_weather(n)


def test_accounting_fix_is_opt_in_and_does_not_change_a_sufficient_connector():
    reference, weather = oil_pair()
    off = {**settings(), "resize_transport_oil_delivery": False}
    n, _ = build_dispatch_weather(reference, weather, off)
    assert n.links.at["ES0 land transport oil", "p_nom"] == 4
    assert (
        n.meta["dispatch_weather_prepared"]["oil_delivery_accounting_adjustments"] == []
    )
    reference.links.loc[
        reference.links.carrier.eq("land transport oil"), "p_nom_opt"
    ] = 20
    n, _ = build_dispatch_weather(reference, weather, settings())
    assert n.links.at["ES0 land transport oil", "p_nom"] == 20
    assert (
        n.meta["dispatch_weather_prepared"]["oil_delivery_accounting_adjustments"] == []
    )


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("capital_cost", 1),
        ("length", 1),
        ("efficiency", 0.9),
        ("p_min_pu", -1),
        ("p_nom_max", 6),
    ],
)
def test_physical_or_nonstandard_oil_link_is_not_silently_relaxed(attribute, value):
    reference, weather = oil_pair()
    reference.links.at["ES0 land transport oil", attribute] = value
    original = reference.links.copy()
    with pytest.raises(ValueError, match="[Oo]il accounting link"):
        build_dispatch_weather(reference, weather, settings())
    pd.testing.assert_frame_equal(reference.links, original)


def test_accounting_connector_must_be_a_dedicated_final_sink():
    reference, weather = oil_pair()
    for n in [reference, weather]:
        n.add(
            "Link",
            "unexpected output",
            bus0="ES0 land transport oil",
            bus1="EU oil",
            p_nom=1,
        )
    with pytest.raises(ValueError, match="dedicated final-demand sink"):
        build_dispatch_weather(reference, weather, settings())


def test_accounting_exception_and_fixed_fleet_guard_survive_netcdf_roundtrip(tmp_path):
    reference, weather = oil_pair()
    n, ledger = build_dispatch_weather(reference, weather, settings())
    path = tmp_path / "dispatch.nc"
    n.export_to_netcdf(path)
    restored = pypsa.Network(path)
    validate_dispatch_weather(restored)
    assert (
        len(
            restored.meta["dispatch_weather_prepared"][
                "oil_delivery_accounting_adjustments"
            ]
        )
        == 2
    )
    assert ledger.annualized_capex_eur.sum() == 33
    restored.links.at["ES0 land transport oil", "p_nom"] += 1
    with pytest.raises(ValueError, match="capacity or location changed"):
        validate_dispatch_weather(restored)


def test_stale_dispatch_preparation_is_rejected_before_solving():
    reference, weather = fixture_pair()
    n, _ = build_dispatch_weather(reference, weather, settings())
    del n.meta["dispatch_weather_prepared"]["resize_transport_oil_delivery"]
    workflow = mock_snakemake()
    workflow.config["controlled_weather"] = {
        "enable": True,
        "mode": "dispatch",
        **settings(),
    }
    with pytest.raises(ValueError, match="regenerate the _dispatch.nc"):
        MODULE.prepare_network_before_model(n, workflow)


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_small_dispatch_lp_preserves_fuel_carbon_and_has_no_new_investments(year):
    reference, weather = oil_pair(year)
    n, ledger = build_dispatch_weather(reference, weather, settings())
    workflow = mock_snakemake()
    workflow.config["controlled_weather"] = {
        "enable": True,
        "mode": "dispatch",
        **settings(),
    }
    MODULE.prepare_network_before_model(n, workflow)
    n.optimize.create_model()
    validate_dispatch_weather(n, model=True)
    status, condition = n.optimize.solve_model(
        solver_name="highs", log_to_console=False
    )
    assert (status, condition) == ("ok", "optimal")
    MODULE.finalize_network_after_solve(n, workflow)
    result = n.meta["dispatch_weather_result"]
    demand = n.get_switchable_as_dense("Load", "p_set")
    oil_assets = ["DE0 land transport oil", "ES0 land transport oil"]
    for asset in oil_assets:
        np.testing.assert_allclose(n.links_t.p0[asset], demand[asset])
        np.testing.assert_allclose(-n.links_t.p2[asset], demand[asset] * 0.27)
        assert result["unserved_energy_by_bus"][asset]["unserved_mwh"] == pytest.approx(
            0
        )
    carbon = float(
        demand[oil_assets].sum(axis=1).dot(n.snapshot_weightings.objective)
        * 0.27
        * PRICE
    )
    assert result["co2_payments_in_objective_eur"] == pytest.approx(carbon)
    assert result[
        "objective_including_reference_capex_excluding_co2_and_unserved_penalty_eur"
    ] == pytest.approx(
        n.objective + 33 - carbon - result["unserved_energy_penalty_eur"]
    )
    assert ledger.annualized_capex_eur.sum() == 33


def synthetic_report():
    reference, weather = fixture_pair()
    for n in [reference, weather]:
        for name, demand in [("NL0 heat A", 6), ("NL0 heat B", 4)]:
            n.add("Bus", name, carrier="heat", country="NL")
            n.add("Load", name, bus=name, p_set=demand)
    n, _ = build_dispatch_weather(reference, weather, settings())
    n.generators_t.p = pd.DataFrame(0.0, index=n.snapshots, columns=n.generators.index)
    for bus, p in [
        ("NL0 heat A", [1, 2, 0]),
        ("NL0 heat B", [3, 0, 0]),
        ("DE0", [0, 1, 0]),
        ("DE0 heat", [0, 0, 1]),
    ]:
        n.generators_t.p[bus + " " + SHEDDING] = p
    n.stores_t.p["co2 atmosphere"] = [-1, -2, -1]
    n._objective = 5000000.0
    return n


def test_shortage_reports_group_simultaneous_peaks_and_use_weighted_hours():
    n = synthetic_report()
    result = dispatch_result_summary(n)
    nl = result["unserved_energy_by_country_carrier"]["NL|heat"]
    assert nl == {
        "unserved_mwh": 18,
        "demand_mwh": 90,
        "unserved_share_pct": 20,
        "hours": 6,
        "peak_mw": 4,
    }
    assert result["unserved_energy_mwh_by_country_carrier"]["NL|heat"] == 18
    assert result["unserved_energy_by_bus"]["NL0 heat A"]["peak_mw"] == 2
    assert result["unserved_energy_by_bus"]["NL0 heat B"]["hours"] == 3
    assert result["hours_with_any_unserved_final_demand"] == 9
    assert result["unserved_energy_penalty_eur"] == 2400000
    assert result["operating_objective_excluding_unserved_penalty_eur"] == 2600000
    assert (
        result["objective_including_reference_capex_excluding_unserved_penalty_eur"]
        == 2600033
    )
    assert result["co2_payments_in_objective_eur"] == pytest.approx(12 * PRICE)
    assert result[
        "objective_including_reference_capex_excluding_co2_and_unserved_penalty_eur"
    ] == pytest.approx(2600033 - 12 * PRICE)


def test_reporting_distinguishes_energy_weights_and_objective_weights():
    n = synthetic_report()
    n.snapshot_weightings.objective = 6
    result = dispatch_result_summary(n)
    assert result["unserved_energy_mwh_by_country_carrier"]["NL|heat"] == 18
    assert result["unserved_energy_by_country_carrier"]["NL|heat"]["hours"] == 6
    assert result["unserved_energy_penalty_eur"] == 4800000
    assert result["co2_payments_in_objective_eur"] == pytest.approx(24 * PRICE)


def test_hour_threshold_does_not_discard_small_unserved_energy():
    n = synthetic_report()
    n.generators_t.p.loc[:, :] = 0
    n.generators_t.p.loc[n.snapshots[0], "DE0 " + SHEDDING] = 0.0005
    result = dispatch_result_summary(n)
    assert result["hours_with_any_unserved_final_demand"] == 0
    assert result["unserved_energy_mwh_by_country_carrier"]["DE|AC"] == pytest.approx(
        0.0015
    )


@pytest.mark.parametrize("problem", ["missing", "nan", "over_demand", "carbon_missing"])
def test_invalid_solved_shortfall_cannot_be_reported_as_zero(problem):
    n = synthetic_report()
    asset = "DE0 " + SHEDDING
    if problem == "missing":
        n.generators_t.p = n.generators_t.p.drop(columns=[asset])
    elif problem == "nan":
        n.generators_t.p.loc[n.snapshots[0], asset] = np.nan
    elif problem == "over_demand":
        n.generators_t.p.loc[n.snapshots[0], asset] = 100
    else:
        n.stores_t.p = n.stores_t.p.drop(columns=["co2 atmosphere"])
    with pytest.raises(ValueError):
        dispatch_result_summary(n)


def test_no_shedding_report_is_valid_when_diagnostic_is_disabled():
    reference, weather = fixture_pair()
    options = {**settings(), "dispatch_unserved_energy": False}
    n, _ = build_dispatch_weather(reference, weather, options)
    n._objective = 1000
    n.stores_t.p["co2 atmosphere"] = 0
    result = dispatch_result_summary(n)
    assert result["unserved_energy_by_bus"] == {}
    assert result["hours_with_any_unserved_final_demand"] == 0
    assert (
        result["objective_including_reference_capex_excluding_unserved_penalty_eur"]
        == 1033
    )


@pytest.mark.parametrize("penalty", [0, -1, float("nan"), float("inf")])
def test_invalid_penalty_is_rejected(penalty):
    with pytest.raises(ValidationError):
        ControlledWeatherConfig(dispatch_unserved_energy_cost=penalty)
    reference, _ = fixture_pair()
    with pytest.raises(ValueError, match="positive and finite"):
        add_demand_shedding(reference, penalty)
