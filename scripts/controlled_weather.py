# SPDX-License-Identifier: MIT

"""Opt-in common-history weather experiments; never reuse solved operations."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calendar_keys(index):
    index = pd.DatetimeIndex(index)
    if index.empty or index.hasnans or len(index.year.unique()) != 1:
        raise ValueError("Controlled weather requires one non-empty calendar year.")
    keys = pd.MultiIndex.from_arrays(
        [index.month, index.day, index.hour, index.minute, index.second]
    )
    if not keys.is_unique:
        raise ValueError("Duplicate month/day/time in controlled-weather snapshots.")
    return keys


def align_calendar(frame, snapshots):
    """Match calendar positions, not row numbers or blindly replaced year labels."""
    source, target = calendar_keys(frame.index), calendar_keys(snapshots)
    if len(source) != len(target) or len(source.difference(target)):
        raise ValueError(
            "Snapshot calendars differ; align resolution and remove leap day explicitly."
        )
    result = frame.copy()
    result.index = source
    result = result.reindex(target)
    result.index = snapshots
    if result.isna().any().any():
        raise ValueError(
            "Missing input values after controlled-weather calendar alignment."
        )
    return result


def asset_stem(name):
    name = re.sub(r"-\d{4}$", "", str(name))
    for system in ("rural", "urban decentral"):
        for sector in ("residential", "services"):
            name = name.replace(f" {system} {sector} ", f" {system} ")
    return name


def _profile_column(source, component, attribute, asset, snapshots):
    if not source.snapshots.equals(snapshots):
        raise ValueError("Fresh weather profile network has the wrong snapshots.")
    static = source.components[component].static
    names = static.index[static.index.map(asset_stem) == asset_stem(asset)]
    if names.empty:
        raise ValueError(f"Missing fresh {component}.{attribute} profile for {asset}.")
    values = source.get_switchable_as_dense(component, attribute, inds=names)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Invalid fresh {component}.{attribute} profile for {asset}.")
    if not np.allclose(values.to_numpy(), values.iloc[:, [0]].to_numpy()):
        raise ValueError(f"Ambiguous fresh profiles after heat-bus matching: {asset}.")
    return values.iloc[:, 0]


def refresh_brownfield_weather(previous, current, vintage_weather, settings):
    """Keep old nominal capacities/ages, but replace old weather-dependent inputs."""
    year = settings["previous_horizon"]
    recorded = previous.meta.get("wildcards", {}).get("planning_horizons")
    if recorded is not None and int(recorded) != year:
        raise ValueError(f"Expected a {year} common-history network, got {recorded}.")
    before = {
        c.name: c.static.copy(deep=True)
        for c in previous.components[["Generator", "Link", "Store"]]
    }
    weights = align_calendar(previous.snapshot_weightings, current.snapshots)
    if not np.allclose(weights, current.snapshot_weightings):
        raise ValueError("Reference and weather snapshot weights differ.")
    inputs = {}
    for c in previous.components:
        for attribute, values in c.dynamic.items():
            if values.empty or "Input" not in str(c.defaults.at[attribute, "status"]):
                continue
            inputs[c.name, attribute] = align_calendar(values, current.snapshots)

    # Outputs (dispatch, energy and SOC) are not an input to a new planning year.
    previous.set_snapshots(current.snapshots)
    previous.snapshot_weightings = weights
    for c in previous.components:
        for attribute in c.dynamic:
            c.dynamic[attribute] = inputs.get(
                (c.name, attribute), pd.DataFrame(index=current.snapshots)
            )

    horizons = sorted(vintage_weather)
    changed = 0
    for component in ("Generator", "Link"):
        c = previous.components[component]
        for attribute, values in c.dynamic.items():
            if values.empty:
                continue
            for asset in values.columns:
                row = c.static.loc[asset]
                if not np.isfinite(row.lifetime):
                    continue
                if row.build_year + row.lifetime <= settings["planning_horizon"]:
                    continue
                if row.get("p_nom_extendable", False) and row.get(
                    "p_nom_opt", 0.0
                ) < settings.get("capacity_threshold", 0.0):
                    continue
                renewable = (
                    component == "Generator"
                    and attribute == "p_max_pu"
                    and (
                        row.carrier in settings["frozen_generator_carriers"]
                        or "solar thermal" in row.carrier
                    )
                )
                heat_pump = (
                    component == "Link"
                    and "heat pump" in row.carrier
                    and (attribute in {"efficiency", "efficiency2", "efficiency3"})
                )
                if not (renewable or heat_pump):
                    continue
                eligible = [h for h in horizons if h <= row.build_year]
                vintage = max(eligible) if eligible else min(horizons)
                values.loc[:, asset] = _profile_column(
                    vintage_weather[vintage],
                    component,
                    attribute,
                    asset,
                    current.snapshots,
                ).to_numpy()
                changed += 1
    for component, expected in before.items():
        pd.testing.assert_frame_equal(previous.components[component].static, expected)
    logger.info(
        "Aligned common %s history to weather %s; replaced %s vintage profiles; "
        "capacities and lifetimes unchanged, solved operations discarded.",
        year,
        current.snapshots[0].year,
        changed,
    )


def validate_input_series(n):
    for c in n.components:
        for attribute, values in c.dynamic.items():
            if values.empty or "Input" not in str(c.defaults.at[attribute, "status"]):
                continue
            if not values.index.equals(n.snapshots):
                raise ValueError(f"Wrong snapshot index: {c.name}.{attribute}.")
            if not values.columns.isin(c.static.index).all():
                raise ValueError(f"Unknown assets in {c.name}.{attribute}.")
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError(f"Missing/non-finite input: {c.name}.{attribute}.")


def freeze_domestic_renewables(n, reference, settings, tolerance):
    """Fix node/technology/vintage capacity, not dispatch or a national aggregate."""
    if set(n.buses.index) != set(reference.buses.index):
        raise ValueError("Reference and weather networks have different bus sets.")
    country = settings["domestic_country"]
    rows = []
    for component, carriers in (
        ("Generator", settings["frozen_generator_carriers"]),
        ("StorageUnit", ["hydro", "PHS"]),
    ):
        current = n.components[component].static
        old = reference.components[component].static
        current_selected = current.carrier.isin(carriers) & current.bus.map(
            n.buses.country
        ).eq(country)
        reference_selected = old.carrier.isin(carriers) & old.bus.map(
            reference.buses.country
        ).eq(country)
        targets = old.loc[reference_selected]
        missing = targets.index.difference(current.index)
        if len(missing):
            raise ValueError(
                f"Missing reference {component} assets: {missing.tolist()}."
            )
        extra = current.index[current_selected].difference(targets.index)
        for asset in extra:
            if current.at[asset, "p_nom"] > tolerance:
                raise ValueError(f"Unexpected existing renewable asset: {asset}.")
            current.loc[asset, ["p_nom", "p_nom_min", "p_nom_max"]] = 0.0
            current.at[asset, "p_nom_extendable"] = False
        for asset, row in targets.iterrows():
            value = float(row.p_nom_opt)
            if not np.isfinite(value) or value < -tolerance:
                raise ValueError(
                    f"Invalid reference nominal capacity: {asset}={value}."
                )
            value = max(value, 0.0)
            lower, upper = current.loc[asset, ["p_nom_min", "p_nom_max"]]
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"Reference renewable capacity outside native bounds: {asset}: "
                    f"{value} not in [{lower}, {upper}]."
                )
            if current.at[asset, "bus"] != row.bus:
                raise ValueError(f"Reference renewable location changed: {asset}.")
            current.at[asset, "capital_cost"] = row.capital_cost
            if "onight_cost" in current and "onight_cost" in row:
                current.at[asset, "onight_cost"] = row.onight_cost
            removed = (
                value * float(current.at[asset, "capital_cost"])
                if current.at[asset, "p_nom_extendable"]
                else 0.0
            )
            current.at[asset, "p_nom"] = value
            current.at[asset, "p_nom_extendable"] = False
            current.at[asset, "p_nom_min"] = min(lower, value)
            current.at[asset, "p_nom_max"] = max(upper, value)
            rows.append(
                {
                    "component": component,
                    "asset": asset,
                    "nominal": value,
                    "removed_annualized_capex_eur": removed,
                }
            )
    return pd.DataFrame(
        rows, columns=["component", "asset", "nominal", "removed_annualized_capex_eur"]
    )


def prepare_controlled_weather(n, snakemake):
    settings = getattr(snakemake, "config", {}).get("controlled_weather", {})
    if not settings.get("enable", False):
        return
    if int(snakemake.wildcards.planning_horizons) != settings["planning_horizon"]:
        raise ValueError("Controlled-weather branches solve only the final horizon.")
    import pypsa

    reference = pypsa.Network(snakemake.input.controlled_weather_reference)
    constraints = snakemake.params.solving["constraints"]
    fixed = constraints["fixed_neighbor_capacities"]
    if fixed.get("formulation") != "static" or not fixed.get("enable"):
        raise ValueError("Controlled weather requires static non-DE materialisation.")
    if not constraints.get("fixed_co2_price", {}).get("enable"):
        raise ValueError("Controlled weather requires the common fixed CO2 price.")
    if str(fixed["reference_networks"][2035]) != str(settings["reference_network"]):
        raise ValueError("CO2/fleet and fixed-neighbor reference networks differ.")
    price = snakemake.config["costs"]["emission_prices"]["co2"][2035]
    if not np.isclose(price, -reference.global_constraints.at["CO2Limit", "mu"]):
        raise ValueError(
            "Fixed CO2 price does not match the common reference shadow price."
        )
    validate_input_series(n)
    weights = align_calendar(reference.snapshot_weightings, n.snapshots)
    if not np.allclose(weights, n.snapshot_weightings):
        raise ValueError("Final snapshot weights differ from the reference.")
    # Within-node weather weighting can change representative connection costs.
    # The fleet is fixed, so its investment-cost constants must remain fixed too.
    foreign_summary = n._fixed_neighbor_static_summary
    foreign_path = Path(foreign_summary["capex_ledger"])
    foreign_ledger = pd.read_csv(
        foreign_path, dtype={"asset": str, "was_extendable": "boolean"}
    )
    for i, row in foreign_ledger.iterrows():
        current_static = n.components[row.component].static
        reference_static = reference.components[row.component].static
        if row.asset not in reference_static.index:
            continue  # Absent zero-capacity candidates have no cost contribution.
        cost = float(reference_static.at[row.asset, "capital_cost"])
        current_static.at[row.asset, "capital_cost"] = cost
        if "onight_cost" in current_static and "onight_cost" in reference_static:
            current_static.at[row.asset, "onight_cost"] = reference_static.at[
                row.asset, "onight_cost"
            ]
        foreign_ledger.at[i, "capital_cost"] = cost
        foreign_ledger.at[i, "removed_annualized_capex"] = (
            cost * row.applied_nominal if row.was_extendable else 0.0
        )
    foreign_ledger.to_csv(foreign_path, index=False)
    foreign_summary["removed_annualized_capex_eur"] = float(
        foreign_ledger.removed_annualized_capex.sum()
    )
    logger.info(
        "Pinned fixed-fleet capital costs to the common reference; non-DE CAPEX constant %.6f EUR/a.",
        foreign_summary["removed_annualized_capex_eur"],
    )
    ledger = freeze_domestic_renewables(
        n, reference, settings, fixed.get("bound_clip_tolerance", 0.05)
    )
    path = (
        Path(snakemake.output.network).parent.parent
        / "costs"
        / "controlled_weather_renewable_capex.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(path, index=False)
    n._controlled_weather_summary = {
        "weather_year": int(n.snapshots[0].year),
        "common_previous_network": settings["previous_network"],
        "common_fleet_reference": settings["reference_network"],
        "fixed_domestic_renewable_assets": len(ledger),
        "removed_annualized_domestic_renewable_capex_eur": float(
            ledger.removed_annualized_capex_eur.sum()
        ),
        "renewable_capex_ledger": str(path),
    }
    logger.info("Validated controlled-weather inputs and fixed German renewable fleet.")


def cached_weather_resources(directory):
    """Reuse compatible preprocessing only; never an investment-bearing network."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and "networks" not in p.relative_to(root).parts
        and p.suffix in {".nc", ".csv", ".geojson", ".tif", ".json"}
    )
