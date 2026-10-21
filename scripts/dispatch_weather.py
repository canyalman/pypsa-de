# SPDX-License-Identifier: MIT

"""Re-operate a solved fleet under new weather without changing its investments."""

import hashlib
import json
import logging

import numpy as np
import pandas as pd
from pypsa.descriptors import nominal_attrs

from scripts.controlled_weather import (
    align_calendar,
    is_input_attribute,
    validate_input_series,
)

logger = logging.getLogger(__name__)
MARKER = "weather_dispatch_was_extendable"
SHEDDING = "weather demand shedding"


def fleet_fingerprint(n):
    records = []
    for component, nominal in nominal_attrs.items():
        static = n.components[component].static
        for asset, row in static.iterrows():
            if component == "Generator" and row.carrier == SHEDDING:
                continue
            buses = [
                str(row[key])
                for key in sorted(static.columns)
                if key == "bus" or key.startswith("bus") and key[3:].isdigit()
            ]
            records.append([component, str(asset), float(row[nominal]).hex(), buses])
    return hashlib.sha256(json.dumps(sorted(records)).encode("ascii")).hexdigest()


def is_dispatch_weather(config):
    settings = config.get("controlled_weather", {})
    return settings.get("enable", False) and settings.get("mode") == "dispatch"


def fix_reference_fleet(n, nominal_tolerance=0.05):
    """Use solved nominal values, retaining static costs, vintages and locations."""
    rows = []
    for component, nominal in nominal_attrs.items():
        static = n.components[component].static
        if static.empty:
            continue
        extendable = static[nominal + "_extendable"].astype(bool)
        static[MARKER] = extendable
        targets = static[nominal].where(~extendable, static[nominal + "_opt"])
        invalid = targets.isna() | targets.lt(-nominal_tolerance)
        if invalid.any():
            raise ValueError(
                f"Invalid solved {component} capacities: {targets[invalid].to_dict()}"
            )
        if (targets.lt(0)).any():
            logger.warning(
                "Clipping negative %s nominal residuals within %.6f nominal units: %s.",
                component,
                nominal_tolerance,
                targets[targets.lt(0)].to_dict(),
            )
        static[nominal] = targets.clip(lower=0)
        static[nominal + "_extendable"] = False
        for asset, row in static.iterrows():
            capacity = float(row[nominal])
            cost = float(row.capital_cost)
            if not np.isfinite(cost) or (not np.isfinite(capacity) and cost != 0):
                raise ValueError(
                    f"Invalid fixed-fleet investment cost: {component} {asset}."
                )
            rows.append(
                {
                    "component": component,
                    "asset": asset,
                    "nominal_attribute": nominal,
                    "reference_nominal": float(targets.at[asset]),
                    "fixed_nominal": capacity,
                    "was_extendable": bool(extendable.at[asset]),
                    "capital_cost": cost,
                    "annualized_capex_eur": capacity * cost if cost else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _replace_weather_attribute(n, weather, component, attribute, assets):
    if assets.empty:
        return
    source = weather.components[component].static
    static = n.components[component].static
    missing = assets.difference(source.index)
    if len(missing):
        nominal = nominal_attrs.get(component)
        relevant = (
            missing if nominal is None else missing[static.loc[missing, nominal].gt(0)]
        )
        if len(relevant):
            raise ValueError(
                f"Missing target-weather {component}.{attribute} assets: {relevant.tolist()}"
            )
        assets = assets.intersection(source.index)
    if assets.empty:
        return
    values = weather.get_switchable_as_dense(component, attribute, inds=assets)
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite target-weather {component}.{attribute}.")
    n.components[component].dynamic[attribute].loc[:, assets] = values


def add_demand_shedding(n, penalty):
    """Slack is bounded by actual positive final demand, never an unlimited plant."""
    demand = n.get_switchable_as_dense("Load", "p_set").mul(-n.loads.sign).clip(lower=0)
    demand = demand.T.groupby(n.loads.bus).sum().T
    peaks = demand.max()
    peaks = peaks[peaks.gt(0)]
    if peaks.empty:
        return
    names = peaks.index + " " + SHEDDING
    if names.isin(n.generators.index).any():
        raise ValueError("Demand-shedding names collide with reference generators.")
    if SHEDDING not in n.carriers.index:
        n.add("Carrier", SHEDDING, nice_name="Unserved final demand", color="#d62728")
    limits = demand.loc[:, peaks.index].div(peaks)
    limits.columns = names
    n.add(
        "Generator",
        names,
        bus=peaks.index,
        carrier=SHEDDING,
        p_nom=peaks.to_numpy(),
        p_nom_extendable=False,
        marginal_cost=penalty,
        p_max_pu=limits,
    )


def build_dispatch_weather(reference, weather, settings):
    if set(reference.buses.index) != set(weather.buses.index):
        raise ValueError("Reference and target-weather bus sets differ.")
    for attribute in ["country", "carrier"]:
        if not reference.buses[attribute].equals(
            weather.buses[attribute].reindex(reference.buses.index)
        ):
            raise ValueError(
                f"Target-weather bus {attribute} assignments differ from the reference."
            )
    for attribute in ["x", "y"]:
        if not np.allclose(
            reference.buses[attribute],
            weather.buses[attribute].reindex(reference.buses.index),
            atol=1e-6,
            rtol=0,
            equal_nan=True,
        ):
            raise ValueError(
                "Target-weather bus locations differ; reuse the reference spatial clustering."
            )
    if reference.snapshots.year.unique().tolist() != [2013]:
        raise ValueError(
            "This experiment requires the agreed 2013-weather fleet reference."
        )
    recorded = reference.meta.get("wildcards", {}).get("planning_horizons")
    if recorded is not None and int(recorded) != 2035:
        raise ValueError(
            "Dispatch needs the solved 2035 fleet, not a previous horizon."
        )
    if not np.allclose(
        align_calendar(reference.snapshot_weightings, weather.snapshots),
        weather.snapshot_weightings,
    ):
        raise ValueError("Dispatch reference and weather snapshot weights differ.")
    n = reference.copy()
    ledger = fix_reference_fleet(n, settings.get("nominal_clip_tolerance", 0.05))
    inputs = {
        (c.name, a): align_calendar(v, weather.snapshots)
        for c in n.components.values()
        for a, v in c.dynamic.items()
        if not v.empty and is_input_attribute(c, a)
    }
    n.set_snapshots(weather.snapshots)
    n.snapshot_weightings = weather.snapshot_weightings.copy()
    for c in n.components.values():
        for attribute in c.dynamic:
            c.dynamic[attribute] = inputs.get(
                (c.name, attribute),
                pd.DataFrame(index=n.snapshots, columns=c.static.index[:0]),
            )

    _replace_weather_attribute(n, weather, "Load", "p_set", n.loads.index)
    profile_attributes = {
        "Generator": ["p_max_pu", "p_min_pu"],
        "Link": ["p_max_pu", "p_min_pu", "efficiency"],
        "Store": ["e_max_pu", "e_min_pu", "standing_loss"],
        "StorageUnit": ["inflow"],
        "Line": ["s_max_pu"],
    }
    profile_attributes["Link"] += [
        a for a in weather.links_t if a.startswith("efficiency") and a != "efficiency"
    ]
    for component, attributes in profile_attributes.items():
        static = n.components[component].static
        source = weather.components[component]
        for attribute in attributes:
            columns = (
                n.components[component]
                .dynamic[attribute]
                .columns.union(source.dynamic[attribute].columns)
            )
            assets = static.index.intersection(columns)
            # Renewable availability is mandatory even if a frame was omitted.
            if component == "Generator":
                renewable = static.index[
                    static.carrier.isin(settings["frozen_generator_carriers"])
                    | static.carrier.str.contains("solar thermal")
                ]
                assets = assets.union(renewable)
            if component == "Link" and attribute.startswith("efficiency"):
                port = attribute.removeprefix("efficiency")
                if not port:
                    assets = assets.union(
                        static.index[
                            static.carrier.str.contains(
                                "heat pump|heat direct utilisation"
                            )
                        ]
                    )
                else:
                    buses = static.get(
                        "bus" + port, pd.Series("", index=static.index)
                    ).fillna("")
                    assets = assets.intersection(buses.index[buses.ne("")])
            _replace_weather_attribute(n, weather, component, attribute, assets)

    if settings.get("dispatch_unserved_energy", True):
        add_demand_shedding(n, settings["dispatch_unserved_energy_cost"])
    validate_input_series(n)
    summary = {
        "reference_network": settings["reference_network"],
        "weather_year": int(n.snapshots[0].year),
        "planning_horizon": 2035,
        "reference_annualized_capex_eur": float(ledger.annualized_capex_eur.sum()),
        "reference_co2_price_eur_per_t": float(
            -reference.global_constraints.at["CO2Limit", "mu"]
        ),
        "fixed_asset_count": len(ledger),
        "fixed_fleet_fingerprint": fleet_fingerprint(n),
        "demand_shedding_enabled": settings.get("dispatch_unserved_energy", True),
        "unserved_energy_penalty": settings["dispatch_unserved_energy_cost"],
        "storage_boundaries": "Reference cyclic and initial-state inputs retained; solved SOC discarded.",
    }
    n.meta["dispatch_weather_prepared"] = summary
    validate_dispatch_weather(n)
    return n, ledger


def validate_dispatch_weather(n, model=False):
    if "dispatch_weather_prepared" not in n.meta:
        raise ValueError(
            "Missing dispatch preparation audit; do not solve the capacity-expansion network."
        )
    for component, nominal in nominal_attrs.items():
        if n.components[component].static[nominal + "_extendable"].any():
            raise ValueError(f"Dispatch has extendable {component} capacity.")
        if model and f"{component}-{nominal}" in n.model.variables:
            raise ValueError(
                f"Dispatch model contains {component}-{nominal} investment variables."
            )
    if (
        fleet_fingerprint(n)
        != n.meta["dispatch_weather_prepared"]["fixed_fleet_fingerprint"]
    ):
        raise ValueError(
            "A fixed reference capacity or location changed after dispatch preparation."
        )
    validate_input_series(n)


def dispatch_result_summary(n):
    result = dict(n.meta["dispatch_weather_prepared"])
    assets = n.generators.index[n.generators.carrier.eq(SHEDDING)]
    p = n.generators_t.p.reindex(columns=assets, fill_value=0).clip(lower=0)
    energies = p.mul(n.snapshot_weightings.generators, axis=0).sum()
    penalties = (
        p.mul(n.generators.loc[assets, "marginal_cost"], axis=1)
        .mul(n.snapshot_weightings.objective, axis=0)
        .to_numpy()
        .sum()
    )
    buses = n.generators.loc[assets, "bus"]
    groups = pd.DataFrame(
        {
            "energy": energies,
            "country": buses.map(n.buses.country),
            "carrier": buses.map(n.buses.carrier),
        }
    )
    result["unserved_energy_mwh_by_country_carrier"] = {
        f"{country}|{carrier}": float(value)
        for (country, carrier), value in groups.groupby(["country", "carrier"])
        .energy.sum()
        .items()
    }
    result["unserved_energy_penalty_eur"] = float(penalties)
    result["operating_objective_eur"] = float(n.objective)
    result["operating_objective_excluding_unserved_penalty_eur"] = float(
        n.objective - penalties
    )
    result["objective_including_reference_capex_eur"] = float(
        n.objective + result["reference_annualized_capex_eur"]
    )
    return result
