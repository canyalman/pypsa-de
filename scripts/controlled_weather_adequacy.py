# SPDX-License-Identifier: MIT

"""Explicit foreign heat shortfalls and non-physical oil delivery connectors."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNSERVED_HEAT = "controlled weather unserved heat"
FINAL_HEAT = ("rural heat", "urban decentral heat", "urban central heat")
SHORTFALL_HOUR_THRESHOLD_MW = 0.001


def _bus_demand(n):
    demand = n.get_switchable_as_dense("Load", "p_set").mul(-n.loads.sign)
    if not np.isfinite(demand.to_numpy()).all():
        raise ValueError(
            "Non-finite final demand in controlled-weather adequacy inputs."
        )
    return demand.T.groupby(n.loads.bus).sum().T


def foreign_heat_demand(n, domestic_country):
    """Sum all loads, including static agriculture heat, on each final-heat bus."""
    heat = n.buses.index[n.buses.carrier.isin(FINAL_HEAT)]
    if heat.empty:
        return pd.DataFrame(index=n.snapshots)
    if "country" not in n.buses:
        raise ValueError(
            "Final-heat buses need explicit countries for foreign-only slack."
        )
    countries = n.buses.loc[heat, "country"].fillna("")
    if countries.eq("").any():
        raise ValueError(
            "Final-heat buses need explicit countries for foreign-only slack."
        )
    foreign = heat[countries.ne(domestic_country)]
    demand = _bus_demand(n).reindex(columns=foreign, fill_value=0).clip(lower=0)
    return demand.loc[:, demand.max().gt(0)]


def add_foreign_heat_shortfall(n, settings):
    enabled = settings.get("foreign_heat_unserved_energy", False)
    penalty = float(settings.get("foreign_heat_unserved_energy_cost", 100000.0))
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Foreign heat shortfall penalty must be positive and finite.")
    summary = {
        "enabled": enabled,
        "penalty_eur_per_mwh_th": penalty,
        "scope": "non-domestic final heat only",
        "asset_count": 0,
    }
    if not enabled:
        return summary
    demand = foreign_heat_demand(n, settings["domestic_country"])
    if demand.empty:
        return summary
    names = demand.columns + " " + UNSERVED_HEAT
    if names.isin(n.generators.index).any() or UNSERVED_HEAT in n.carriers.index:
        raise ValueError(
            "Controlled-weather heat shortfall asset/carrier name collision."
        )
    n.add("Carrier", UNSERVED_HEAT, nice_name="Unserved foreign heat", color="#d62728")
    peaks = demand.max()
    limits = demand.div(peaks)
    limits.columns = names
    n.add(
        "Generator",
        names,
        bus=demand.columns,
        carrier=UNSERVED_HEAT,
        p_nom=peaks.to_numpy(),
        p_nom_extendable=False,
        capital_cost=0.0,
        marginal_cost=penalty,
        p_max_pu=limits,
    )
    summary["asset_count"] = len(names)
    logger.info(
        "Enabled bounded unserved heat on %s foreign final-heat buses at %.1f EUR/MWhth; "
        "heat demand and physical capacities are unchanged.",
        len(names),
        penalty,
    )
    return summary


def resize_foreign_oil_delivery(n, settings):
    """Remove a peak-flow artefact only from zero-CAPEX final oil accounting links."""
    if not settings.get("resize_foreign_transport_oil_delivery", False):
        return []
    demand = _bus_demand(n)
    links = n.links.loc[n.links.carrier.eq("land transport oil")]
    countries = links.bus1.map(n.buses.country).fillna("")
    if countries.eq("").any():
        raise ValueError("Transport oil delivery buses need explicit countries.")
    adjustments = []
    for asset, row in links.loc[countries.ne(settings["domestic_country"])].iterrows():
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
            "Resized foreign oil accounting connector %s from %.6f to %.6f MW "
            "to accommodate unchanged target-weather demand; fuel and CO2 accounting retained.",
            change["asset"],
            change["previous_nominal_mw"],
            change["applied_nominal_mw"],
        )
    return adjustments


def foreign_heat_result_summary(n, domestic_country):
    demand = foreign_heat_demand(n, domestic_country)
    assets = n.generators.index[n.generators.carrier.eq(UNSERVED_HEAT)]
    served_buses = n.generators.loc[assets, "bus"]
    if not served_buses.isin(demand.columns).all() or served_buses.duplicated().any():
        raise ValueError(
            "Unserved-heat generators are outside the foreign final-heat scope."
        )
    p = n.generators_t.p.reindex(index=n.snapshots, columns=assets).clip(lower=0)
    if not np.isfinite(p.to_numpy()).all():
        raise ValueError("Missing/non-finite solved unserved-heat values.")
    penalty = float(
        p.mul(n.get_switchable_as_dense("Generator", "marginal_cost", inds=assets))
        .mul(n.snapshot_weightings.objective, axis=0)
        .to_numpy()
        .sum()
    )
    p.columns = served_buses.to_numpy()
    if (
        (p - demand.reindex(columns=p.columns))
        .gt(SHORTFALL_HOUR_THRESHOLD_MW)
        .any()
        .any()
    ):
        raise ValueError("Solved unserved heat exceeds actual heat demand.")
    energy = p.mul(n.snapshot_weightings.generators, axis=0).sum()
    required = demand.mul(n.snapshot_weightings.generators, axis=0).sum()
    by_bus = {}
    for bus in p:
        by_bus[bus] = {
            "country": n.buses.at[bus, "country"],
            "carrier": n.buses.at[bus, "carrier"],
            "demand_mwh_th": float(required[bus]),
            "unserved_mwh_th": float(energy[bus]),
            "unserved_share_pct": float(100 * energy[bus] / required[bus]),
            "hours": float(
                p[bus]
                .gt(SHORTFALL_HOUR_THRESHOLD_MW)
                .dot(n.snapshot_weightings.generators)
            ),
            "peak_mw_th": float(p[bus].max()),
        }
    total = p.sum(axis=1)
    denominator = float(required.sum())
    return {
        "unserved_heat_mwh_th": float(energy.sum()),
        "foreign_heat_demand_mwh_th": denominator,
        "unserved_heat_share_pct": float(100 * energy.sum() / denominator)
        if denominator
        else 0.0,
        "unserved_heat_hours": float(
            p.gt(SHORTFALL_HOUR_THRESHOLD_MW)
            .any(axis=1)
            .dot(n.snapshot_weightings.generators)
        ),
        "peak_unserved_heat_mw_th": float(total.max()),
        "unserved_heat_hour_threshold_mw_th": SHORTFALL_HOUR_THRESHOLD_MW,
        "unserved_heat_by_bus": by_bus,
        "unserved_heat_penalty_eur": penalty,
    }
