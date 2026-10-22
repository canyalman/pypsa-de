# SPDX-License-Identifier: MIT

"""Foreign heat shortfalls are not generation; oil delivery is not a power plant."""

import numpy as np
import pandas as pd
import pypsa
import pytest
from pydantic import ValidationError

from scripts.controlled_weather_adequacy import (
    UNSERVED_HEAT,
    add_foreign_heat_shortfall,
    foreign_heat_demand,
    foreign_heat_result_summary,
    resize_foreign_oil_delivery,
)
from scripts.lib.validation.config.controlled_weather import ControlledWeatherConfig


def settings():
    return {
        "domestic_country": "DE",
        "foreign_heat_unserved_energy": True,
        "foreign_heat_unserved_energy_cost": 100000.0,
        "resize_foreign_transport_oil_delivery": True,
    }


def heat_network(year=1996):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range(f"{year}-01-01", periods=3, freq="3h"))
    n.snapshot_weightings.loc[:, :] = 3.0
    n.add("Carrier", ["rural heat", "AC", "gas"])
    n.add(
        "Bus",
        ["NL2 0 rural heat", "DE0 heat", "NL0 electricity", "EU gas"],
        carrier=["rural heat", "rural heat", "AC", "gas"],
        country=["NL", "DE", "NL", ""],
    )
    n.add("Load", "NL residential", bus="NL2 0 rural heat", p_set=[7, 3, 7])
    n.add("Load", "NL agriculture", bus="NL2 0 rural heat", p_set=5)
    n.add(
        "Generator",
        "fixed foreign heat",
        bus="NL2 0 rural heat",
        p_nom=10,
        marginal_cost=20,
    )
    return n


def oil_network(year=2023):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range(f"{year}-01-01", periods=2, freq="3h"))
    n.snapshot_weightings.loc[:, :] = 3.0
    n.add("Carrier", ["oil", "land transport oil", "co2"])
    n.add(
        "Bus",
        [
            "EU oil",
            "ES2 0 land transport oil",
            "DE0 land transport oil",
            "co2 atmosphere",
        ],
        carrier=["oil", "land transport oil", "land transport oil", "co2"],
        country=["", "ES", "DE", ""],
    )
    for country, capacity in [("ES2 0", 41495.3587313), ("DE0", 1)]:
        bus = country + " land transport oil"
        n.add(
            "Load",
            bus,
            bus=bus,
            p_set=[41845.6741712, 40000] if country.startswith("ES") else 2,
        )
        n.add(
            "Link",
            bus,
            bus0="EU oil",
            bus1=bus,
            bus2="co2 atmosphere",
            carrier="land transport oil",
            efficiency2=0.27,
            p_nom=capacity,
        )
    return n


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_heat_slack_is_bounded_foreign_only_and_includes_static_agriculture(year):
    n = heat_network(year)
    n.add("Load", "German heat", bus="DE0 heat", p_set=100)
    n.add("Load", "Dutch electricity", bus="NL0 electricity", p_set=200)
    n.add("Load", "fuel demand", bus="EU gas", p_set=300)
    loads, generators = n.loads.copy(), n.generators.copy()
    summary = add_foreign_heat_shortfall(n, settings())
    assets = n.generators.index[n.generators.carrier.eq(UNSERVED_HEAT)]
    assert summary["asset_count"] == 1
    assert n.generators.loc[assets, "bus"].tolist() == ["NL2 0 rural heat"]
    assert not n.generators.loc[assets, "p_nom_extendable"].any()
    maximum = n.get_switchable_as_dense("Generator", "p_max_pu", inds=assets).mul(
        n.generators.loc[assets, "p_nom"]
    )
    np.testing.assert_allclose(maximum.iloc[:, 0], [12, 8, 12])
    pd.testing.assert_frame_equal(n.loads, loads)
    pd.testing.assert_frame_equal(
        n.generators.loc[generators.index, generators.columns], generators
    )


def test_foreign_heat_scope_is_not_hardcoded_to_netherlands():
    n = heat_network()
    n.buses.at["NL2 0 rural heat", "country"] = "IT"
    add_foreign_heat_shortfall(n, settings())
    assert n.generators.carrier.eq(UNSERVED_HEAT).sum() == 1


def test_shortfall_option_is_opt_in_and_rejects_duplicates():
    n = heat_network()
    assert not add_foreign_heat_shortfall(n, {})["enabled"]
    assert UNSERVED_HEAT not in n.carriers.index
    add_foreign_heat_shortfall(n, settings())
    with pytest.raises(ValueError, match="collision"):
        add_foreign_heat_shortfall(n, settings())


@pytest.mark.parametrize("penalty", [0, -1, float("nan"), float("inf")])
def test_invalid_heat_penalties_fail_validation_and_preparation(penalty):
    with pytest.raises(ValidationError):
        ControlledWeatherConfig(foreign_heat_unserved_energy_cost=penalty)
    with pytest.raises(ValueError, match="positive and finite"):
        add_foreign_heat_shortfall(
            heat_network(), {**settings(), "foreign_heat_unserved_energy_cost": penalty}
        )


def test_heat_demand_requires_explicit_country():
    n = heat_network()
    n.buses.at["NL2 0 rural heat", "country"] = ""
    with pytest.raises(ValueError, match="explicit countries"):
        foreign_heat_demand(n, "DE")


@pytest.mark.parametrize("year", [1996, 2010, 2023])
def test_small_heat_lp_serves_available_heat_and_reports_actual_shortfall(year):
    n = heat_network(year)
    add_foreign_heat_shortfall(n, settings())
    status, condition = n.optimize(solver_name="highs", log_to_console=False)
    assert (status, condition) == ("ok", "optimal")
    result = foreign_heat_result_summary(n, "DE")
    assert result["unserved_heat_mwh_th"] == pytest.approx(12)
    assert result["foreign_heat_demand_mwh_th"] == pytest.approx(96)
    assert result["unserved_heat_share_pct"] == pytest.approx(12.5)
    assert result["unserved_heat_hours"] == 6
    assert result["peak_unserved_heat_mw_th"] == pytest.approx(2)
    assert result["unserved_heat_penalty_eur"] == pytest.approx(1200000)
    assert n.objective - result["unserved_heat_penalty_eur"] == pytest.approx(1680)
    bus = result["unserved_heat_by_bus"]["NL2 0 rural heat"]
    assert bus["country"] == "NL" and bus["hours"] == 6
    assert n.generators_t.p["fixed foreign heat"].tolist() == [10, 8, 10]


def test_different_objective_weights_do_not_change_reported_energy_or_hours():
    n = heat_network()
    add_foreign_heat_shortfall(n, settings())
    n.optimize(solver_name="highs", log_to_console=False)
    n.snapshot_weightings.objective = 6.0
    result = foreign_heat_result_summary(n, "DE")
    assert result["unserved_heat_mwh_th"] == pytest.approx(12)
    assert result["unserved_heat_hours"] == 6
    assert result["unserved_heat_penalty_eur"] == pytest.approx(2400000)


def test_heat_costs_separate_penalty_carbon_and_both_fixed_capex_constants():
    from test.test_fixed_co2_price import MODULE, PRICE, mock_snakemake

    n = heat_network()
    add_foreign_heat_shortfall(n, settings())
    n.optimize(solver_name="highs", log_to_console=False)
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Store", "co2 atmosphere", bus="co2 atmosphere", e_nom=100)
    n.stores_t.p["co2 atmosphere"] = [-1, -1, -1]
    n.meta["fixed_neighbor_static"] = {"removed_annualized_capex_eur": 100.0}
    n._controlled_weather_summary = {
        "removed_annualized_domestic_renewable_capex_eur": 50.0
    }
    sm = mock_snakemake()
    sm.config["controlled_weather"] = settings()
    MODULE.finalize_network_after_solve(n, sm)
    result = n.meta["controlled_weather_result"]
    assert result["unserved_heat_penalty_eur"] == pytest.approx(1200000)
    assert result["objective_including_removed_fixed_capex_eur"] == pytest.approx(
        n.objective + 150
    )
    assert result[
        "objective_including_removed_fixed_capex_excluding_unserved_heat_penalty_eur"
    ] == pytest.approx(1830)
    assert result[
        "objective_including_removed_fixed_capex_excluding_co2_and_unserved_heat_penalty_eur"
    ] == pytest.approx(1830 - 9 * PRICE)


def test_only_foreign_zero_cost_oil_accounting_capacity_is_adjusted():
    n = oil_network()
    # Native prepare_network adds small marginal-cost noise before this hook.
    n.links.loc[:, "marginal_cost"] = 0.010734
    before, loads = n.links.copy(), n.loads.copy()
    changes = resize_foreign_oil_delivery(n, settings())
    assert len(changes) == 1
    change = changes[0]
    assert change["asset"] == "ES2 0 land transport oil"
    assert change["increase_mw"] == pytest.approx(350.3154399)
    assert n.links.at[change["asset"], "p_nom"] == pytest.approx(41845.6741712)
    assert not n.links.p_nom_extendable.any()
    before.at[change["asset"], "p_nom"] = change["applied_nominal_mw"]
    pd.testing.assert_frame_equal(n.links, before)
    pd.testing.assert_frame_equal(n.loads, loads)
    assert resize_foreign_oil_delivery(n, settings()) == []


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("capital_cost", 1),
        ("efficiency", 0.9),
        ("p_min_pu", -1),
        ("p_nom_extendable", True),
        ("p_nom_max", 41500),
    ],
)
def test_nonstandard_or_physically_bounded_oil_link_is_not_relaxed(attribute, value):
    n = oil_network()
    n.links.at["ES2 0 land transport oil", attribute] = value
    before = n.links.copy()
    with pytest.raises(ValueError, match="[Oo]il accounting link"):
        resize_foreign_oil_delivery(n, settings())
    pd.testing.assert_frame_equal(n.links, before)


def test_oil_accounting_preserves_fuel_consumption_and_carbon_output():
    n = oil_network()
    n.loads.at["DE0 land transport oil", "p_set"] = 0
    resize_foreign_oil_delivery(n, settings())
    n.add("Generator", "oil supply", bus="EU oil", p_nom=1e6, marginal_cost=1)
    n.add("Store", "co2 atmosphere", bus="co2 atmosphere", e_nom=1e6)
    status, condition = n.optimize(solver_name="highs", log_to_console=False)
    assert (status, condition) == ("ok", "optimal")
    asset = "ES2 0 land transport oil"
    demand = n.get_switchable_as_dense("Load", "p_set")[asset]
    np.testing.assert_allclose(n.links_t.p0[asset], demand)
    np.testing.assert_allclose(-n.links_t.p2[asset], demand * 0.27)
    np.testing.assert_allclose(n.generators_t.p["oil supply"], demand)


def test_oil_resize_is_opt_in_and_does_not_apply_to_other_zero_cost_links():
    n = oil_network()
    before = n.links.copy()
    assert resize_foreign_oil_delivery(n, {}) == []
    pd.testing.assert_frame_equal(n.links, before)
    n.links.at["ES2 0 land transport oil", "carrier"] = "other"
    assert resize_foreign_oil_delivery(n, settings()) == []


def test_oil_final_demand_sink_cannot_be_repurposed_as_a_general_network_link():
    n = oil_network()
    n.add(
        "Link",
        "unexpected route",
        bus0="ES2 0 land transport oil",
        bus1="EU oil",
        p_nom=1,
    )
    before = n.links.copy()
    with pytest.raises(ValueError, match="dedicated final-demand sink"):
        resize_foreign_oil_delivery(n, settings())
    pd.testing.assert_frame_equal(n.links, before)
