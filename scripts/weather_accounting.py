# SPDX-License-Identifier: MIT

"""Keep final oil accounting connectors distinct from the fixed physical fleet."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def resize_transport_oil_delivery(n):
    """Size only native, zero-CAPEX accounting connectors to final-demand peaks."""
    demand = n.get_switchable_as_dense("Load", "p_set").mul(-n.loads.sign)
    if not np.isfinite(demand.to_numpy()).all():
        raise ValueError("Non-finite final demand in weather oil accounting inputs.")
    demand = demand.T.groupby(n.loads.bus).sum().T
    links = n.links.loc[n.links.carrier.eq("land transport oil")]
    countries = links.bus1.map(n.buses.country).fillna("")
    if countries.eq("").any():
        raise ValueError("Transport oil delivery buses need explicit countries.")
    adjustments = []
    for asset, row in links.iterrows():
        if row.bus1 not in demand:
            continue
        peak = float(demand[row.bus1].clip(lower=0).max())
        previous = float(row.p_nom)
        if previous >= peak:
            continue
        other_ports = [
            str(row[key])
            for key in row.index
            if key.startswith("bus") and key[3:].isdigit() and int(key[3:]) > 2
        ]
        native_connector = (
            n.buses.at[row.bus0, "carrier"] == "oil"
            and n.buses.at[row.bus1, "carrier"] == "land transport oil"
            and row.bus2 == "co2 atmosphere"
            and not any(other_ports)
            and not row.p_nom_extendable
            and row.capital_cost == 0
            and row.length == 0
            and np.isfinite(row.marginal_cost)
            and row.marginal_cost_quadratic == 0
        )
        for attribute, expected in [
            ("efficiency", 1.0),
            ("p_min_pu", 0.0),
            ("p_max_pu", 1.0),
            ("marginal_cost_quadratic", 0.0),
        ]:
            values = n.get_switchable_as_dense("Link", attribute, inds=[asset])
            native_connector = native_connector and np.allclose(values, expected)
        if not native_connector:
            raise ValueError(
                f"Refusing to resize a non-standard oil accounting link: {asset}."
            )
        if not np.isfinite(previous) or peak > row.p_nom_max:
            raise ValueError(
                f"Oil accounting link has an invalid/physical capacity bound: {asset}."
            )
        bus_columns = [
            key for key in n.links if key.startswith("bus") and key[3:].isdigit()
        ]
        connected = n.links.index[n.links[bus_columns].eq(row.bus1).any(axis=1)]
        if set(connected) != {asset}:
            raise ValueError(
                f"Transport oil bus is not a dedicated final-demand sink: {row.bus1}."
            )
        adjustments.append(
            {
                "asset": asset,
                "country": countries.at[asset],
                "previous_nominal_mw": previous,
                "applied_nominal_mw": peak,
                "increase_mw": peak - previous,
                "reason": "zero-CAPEX final oil delivery and CO2 accounting, not a physical plant",
            }
        )
    for change in adjustments:
        n.links.at[change["asset"], "p_nom"] = change["applied_nominal_mw"]
        logger.info(
            "Resized oil accounting connector %s from %.6f to %.6f MW "
            "to accommodate unchanged target-weather demand; fuel and CO2 accounting retained.",
            change["asset"],
            change["previous_nominal_mw"],
            change["applied_nominal_mw"],
        )
    return adjustments
