import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml

from scripts.add_electricity import attach_continuous_acaes, calculate_annuity

REPO_ROOT = Path(__file__).parents[1]


def _production_config():
    with (REPO_ROOT / "config" / "config.de.yaml").open(
        encoding="utf-8"
    ) as stream:
        return yaml.safe_load(stream)


def _production_options():
    return _production_config()["electricity"]["storage_options"][
        "aCAES_RESC_continuous"
    ]


def _constraint_function():
    module_path = (
        REPO_ROOT / "scripts" / "pypsa-de" / "additional_functionality.py"
    )
    spec = importlib.util.spec_from_file_location(
        "acaes_continuous_constraints", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.add_continuous_acaes_constraints


def _network_with_buses(buses, periods=1):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2035-01-01", periods=periods, freq="h"))
    n.add(
        "Bus",
        pd.Index(buses),
        carrier="AC",
        country="DE",
        x=np.arange(len(buses), dtype=float),
        y=np.arange(len(buses), dtype=float),
    )
    return n


def test_production_config_uses_approved_continuous_formulation():
    config = _production_config()
    options = _production_options()
    costs = options["costs"]

    assert (
        config["run"]["prefix"]
        == "caes_rte69_fixed_nonde_3h_2025base"
    )
    assert config["run"]["name"] == ["KN2045_Mix_FixedRenewables"]
    assert options["round_trip_efficiency"] == 0.69
    assert options["minimum_output_duration_hours"] == 8
    assert options["maximum_output_duration_hours"] == 48
    assert costs["capex_power_usd2022_per_kw"] == 1699.0
    assert costs["capex_output_energy_usd2022_per_kwh"] == 40.50
    assert costs["fom_power_usd2022_per_kw_year"] == 10.856666666666667
    assert costs["fom_output_energy_usd2022_per_kwh_year"] == 0.020
    assert costs["vom_usd2022_per_mwh_output"] == 1.05
    assert costs["capex_fit_sample_size"] == 18
    assert np.isclose(sum(options["site_output_capacities_twh"].values()), 7.6)


def test_attachment_maps_output_costs_and_geology_to_internal_units():
    options = _production_options()
    buses = pd.Index(options["site_output_capacities_twh"])
    n = _network_with_buses(buses)
    n.snapshot_weightings.loc[:, "generators"] = 8760.0

    attach_continuous_acaes(
        n=n,
        buses_i=buses,
        options=options,
        investment_year=2035,
    )

    carrier = options["carrier"]
    stores = n.stores.index[n.stores.carrier.eq(carrier)]
    chargers = n.links.index[n.links.carrier.eq(f"{carrier} charge")]
    dischargers = n.links.index[n.links.carrier.eq(f"{carrier} discharge")]
    eta = np.sqrt(0.69)

    assert len(stores) == 4
    assert len(chargers) == 4
    assert len(dischargers) == 4
    assert n.storage_units.empty
    assert np.allclose(n.links.loc[chargers, "efficiency"], eta)
    assert np.allclose(n.links.loc[dischargers, "efficiency"], eta)
    assert np.allclose(n.stores.loc[stores, "standing_loss"], 0.0)
    assert np.allclose(n.stores.loc[stores, "e_min_pu"], 0.0)

    site_output_mwh = pd.Series(options["site_output_capacities_twh"]) * 1e6
    assert np.allclose(
        n.stores.loc[stores, "e_nom_max"].to_numpy() * eta,
        site_output_mwh.to_numpy(),
    )

    conversion = options["currency_conversion"]["usd2022_to_eur2020"]
    costs = options["costs"]
    annuity = calculate_annuity(30, 0.07)
    expected_power_cost = (
        costs["capex_power_usd2022_per_kw"] * annuity
        + costs["fom_power_usd2022_per_kw_year"]
    ) * conversion
    expected_output_energy_cost = (
        costs["capex_output_energy_usd2022_per_kwh"] * annuity
        + costs["fom_output_energy_usd2022_per_kwh_year"]
    ) * conversion
    expected_output_vom = costs["vom_usd2022_per_mwh_output"] * conversion

    assert np.allclose(
        n.links.loc[chargers, "capital_cost"] / 1e3,
        expected_power_cost,
    )
    assert np.allclose(
        n.stores.loc[stores, "capital_cost"] / (1e3 * eta),
        expected_output_energy_cost,
    )
    assert np.allclose(
        n.links.loc[dischargers, "marginal_cost"] / eta,
        expected_output_vom,
    )


def test_continuous_acaes_delivers_eight_hours_at_69_percent_rte():
    options = copy.deepcopy(_production_options())
    options["geological_output_capacity_twh"] = 0.004
    options["site_output_capacities_twh"] = {"DE test": 0.004}

    n = _network_with_buses(["DE test"], periods=21)
    n.add(
        "Generator",
        "charging supply",
        bus="DE test",
        p_nom=500,
        p_max_pu=[1.0] * 13 + [0.0] * 8,
    )
    n.add(
        "Load",
        "output demand",
        bus="DE test",
        p_set=[0.0] * 13 + [500.0] * 8,
    )
    attach_continuous_acaes(
        n=n,
        buses_i=pd.Index(["DE test"]),
        options=options,
        investment_year=2035,
    )

    add_constraints = _constraint_function()

    def extra_functionality(network, snapshots):
        add_constraints(network, options)

    status, condition = n.optimize(
        solver_name="highs",
        extra_functionality=extra_functionality,
        include_objective_constant=False,
    )
    assert (status, condition) == ("ok", "optimal")

    carrier = options["carrier"]
    store = f"DE test {carrier}"
    charger = f"{store} charger"
    discharger = f"{store} discharger"
    eta = np.sqrt(0.69)

    charge_power = n.links.at[charger, "p_nom_opt"]
    output_power = eta * n.links.at[discharger, "p_nom_opt"]
    output_energy = eta * n.stores.at[store, "e_nom_opt"]
    assert np.isclose(charge_power, 500.0)
    assert np.isclose(output_power, 500.0)
    assert np.isclose(output_energy, 4000.0)
    assert np.isclose(output_energy / output_power, 8.0)

    charge_input = n.links_t.p0[charger].sum()
    discharge_output = -n.links_t.p1[discharger].sum()
    assert np.isclose(discharge_output, 4000.0)
    assert np.isclose(discharge_output / charge_input, 0.69)

    assert "Link-n_mod" not in n.model.variables
    assert "Store-n_mod" not in n.model.variables
    assert "Link-aCAES-grid-power-ratio-DE test" in n.model.constraints
    assert "Store-aCAES-minimum-output-duration-DE test" in n.model.constraints
    assert "Store-aCAES-maximum-output-duration-DE test" in n.model.constraints
    assert "Store-aCAES-geological-output-DE test" in n.model.constraints
