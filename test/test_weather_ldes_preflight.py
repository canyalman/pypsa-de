# SPDX-License-Identifier: MIT

"""Real LDES preparation must pass the unchanged fixed-fleet preflight."""

import copy
import json

import pandas as pd
import pytest

from scripts.add_existing_baseyear import add_build_year_to_new_assets
from scripts.controlled_weather import freeze_domestic_renewables
from test import test_controlled_weather as controlled_tests
from test.test_controlled_weather import calendar, fleet_pair
from test.test_weather_ldes import attach_via_sector_script, technology
from test.test_weather_ldes_pathways import scenario_config


def prepared_pair(config):
    n, reference = fleet_pair()
    year = pd.Timestamp(config["snapshots"]["start"]).year
    n.set_snapshots(calendar(year)[:3])
    n.snapshot_weightings.loc[:, :] = 3.0
    if technology(config) == "aCAES_RESC_continuous":
        options = config["electricity"]["storage_options"]["aCAES_RESC_continuous"]
        for model in (n, reference):
            model.add(
                "Bus",
                list(options["site_output_capacities_twh"]),
                carrier="AC",
                country="DE",
                x=0.0,
                y=0.0,
            )
    attach_via_sector_script(n, config)
    add_build_year_to_new_assets(n, 2035)
    return n, reference


def internal_assets(n, reference):
    buses = n.buses.index.difference(reference.buses.index)
    stores = n.stores.index[n.stores.bus.isin(buses)]
    links = n.links.index[n.links.bus0.isin(buses) | n.links.bus1.isin(buses)]
    return buses, stores, links


@pytest.mark.parametrize("power", [None, 15, 35, 55])
def test_ldes_internals_preserve_the_reference_and_endogenous_storage(power):
    config = scenario_config(power)
    n, reference = prepared_pair(config)
    buses, stores, links = internal_assets(n, reference)
    before_buses = n.buses.copy(deep=True)
    before_stores = n.stores.loc[stores].copy(deep=True)
    before_links = n.links.loc[links].copy(deep=True)
    before_reference = reference.copy()
    ledger = freeze_domestic_renewables(
        n,
        reference,
        config["controlled_weather"],
        0.05,
        electricity=config["electricity"],
    )
    assert len(buses) == (1 if technology(config) == "iron-air" else 4)
    assert len(ledger) == 1
    assert n.generators.at["DE0 solar-2035", "p_nom"] == 30
    assert not n.generators.at["DE0 solar-2035", "p_nom_extendable"]
    assert n.generators.at["FR0 solar-2035", "p_nom_extendable"]
    assert n.links.at["DE0 OCGT-2035", "p_nom_extendable"]
    pd.testing.assert_frame_equal(n.buses, before_buses)
    pd.testing.assert_frame_equal(n.stores.loc[stores], before_stores)
    pd.testing.assert_frame_equal(n.links.loc[links], before_links)
    for component in ("Bus", "Generator", "Link", "Store"):
        pd.testing.assert_frame_equal(
            reference.components[component].static,
            before_reference.components[component].static,
        )


@pytest.mark.parametrize(
    "failure",
    [
        "missing_common_bus",
        "unexpected_bus",
        "wrong_country",
        "foreign_site",
        "unknown_site",
        "wrong_carrier",
        "disabled_technology",
        "unconfigured_site",
        "missing_store",
        "duplicate_store",
        "wrong_store_carrier",
        "fixed_store",
        "missing_charger",
        "wrong_direction",
        "wrong_efficiency",
        "wrong_dynamic_efficiency",
        "fixed_discharger",
        "extra_port",
        "extra_load",
        "extra_generator",
        "extra_line",
        "extra_link",
        "wrong_name",
    ],
)
def test_ldes_exception_does_not_accept_arbitrary_topology_or_configuration(failure):
    config = scenario_config(None)
    n, reference = prepared_pair(config)
    buses, stores, links = internal_assets(n, reference)
    bus = buses[0]
    store = stores[n.stores.loc[stores, "bus"].eq(bus)][0]
    charger = links[n.links.loc[links, "bus1"].eq(bus)][0]
    discharger = links[n.links.loc[links, "bus0"].eq(bus)][0]
    site = n.buses.at[bus, "location"]
    key = technology(config)
    options = config["electricity"]["storage_options"][key]
    if failure == "missing_common_bus":
        n.remove("Bus", "FR0")
    elif failure == "unexpected_bus":
        n.add("Bus", "unexpected AC", carrier="AC", country="DE")
    elif failure == "wrong_country":
        n.buses.at[bus, "country"] = "FR"
    elif failure == "foreign_site":
        n.buses.at[bus, "location"] = "FR0"
    elif failure == "unknown_site":
        n.buses.at[bus, "location"] = "DE missing"
    elif failure == "wrong_carrier":
        n.buses.at[bus, "carrier"] = "battery"
    elif failure == "disabled_technology":
        if key == "iron-air":
            config["electricity"]["extendable_carriers"]["Store"].remove(key)
        else:
            options["enable"] = False
    elif failure == "unconfigured_site":
        if key == "iron-air":
            options["bus_carriers"] = []
        else:
            options["site_output_capacities_twh"].pop(site)
    elif failure == "missing_store":
        n.remove("Store", store)
    elif failure == "duplicate_store":
        n.add("Store", "unexpected store", bus=bus, carrier=n.buses.at[bus, "carrier"])
    elif failure == "wrong_store_carrier":
        n.stores.at[store, "carrier"] = "battery"
    elif failure == "fixed_store":
        n.stores.at[store, "e_nom_extendable"] = False
    elif failure == "missing_charger":
        n.remove("Link", charger)
    elif failure == "wrong_direction":
        n.links.loc[charger, ["bus0", "bus1"]] = [bus, site]
    elif failure == "wrong_efficiency":
        n.links.at[charger, "efficiency"] = 0.5
    elif failure == "wrong_dynamic_efficiency":
        n.links_t.efficiency[charger] = [0.5, 0.5, 0.5]
    elif failure == "fixed_discharger":
        n.links.at[discharger, "p_nom_extendable"] = False
    elif failure == "extra_port":
        n.links.at[charger, "bus2"] = "FR0"
    elif failure == "extra_load":
        n.add("Load", "unexpected load", bus=bus, p_set=1)
    elif failure == "extra_generator":
        n.add("Generator", "unexpected generator", bus=bus, p_nom=1)
    elif failure == "extra_line":
        n.add("Line", "unexpected line", bus0=site, bus1=bus, x=0.1, s_nom=1)
    elif failure == "extra_link":
        n.add("Link", "unexpected link", bus0=site, bus1=bus, p_nom=1)
    elif failure == "wrong_name":
        n.buses.rename(index={bus: bus + " unexpected"}, inplace=True)
    else:
        raise AssertionError(failure)
    original = n.generators.copy(deep=True)
    with pytest.raises(
        ValueError, match="Missing reference buses|Unexpected controlled-weather bus"
    ):
        freeze_domestic_renewables(
            n,
            reference,
            config["controlled_weather"],
            0.05,
            electricity=config["electricity"],
        )
    pd.testing.assert_frame_equal(n.generators, original)


def test_ldes_bus_exception_requires_explicit_active_configuration():
    config = scenario_config(None)
    n, reference = prepared_pair(config)
    with pytest.raises(ValueError, match="not an enabled domestic LDES carrier"):
        freeze_domestic_renewables(n, reference, config["controlled_weather"], 0.05)


@pytest.mark.parametrize("power", [None, 15, 35, 55])
def test_real_preflight_after_ldes_attachment_and_native_vintage_labelling(
    power, tmp_path, monkeypatch
):
    config = scenario_config(power)
    name = controlled_tests.SCENARIO
    if power is not None:
        name += f"_BESS_{power}GW"
    monkeypatch.setattr(controlled_tests, "SCENARIO", name)
    monkeypatch.setattr(controlled_tests, "fleet_pair", lambda: prepared_pair(config))
    controlled_tests.test_actual_preflight_script_checks_network_without_an_lp(
        tmp_path, monkeypatch, copy.deepcopy(config)
    )
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["weather_year"] == pd.Timestamp(config["snapshots"]["start"]).year
    assert report["status"] == "passed_without_solve"
    assert report["fixed_neighbor"]["materialized_assets"] == 2
    assert report["controlled_weather"]["fixed_domestic_renewable_assets"] == 1
