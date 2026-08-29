# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Extract non-domestic nominal capacities from solved reference networks."""

from pathlib import Path

import pandas as pd
import pypsa

COMPONENTS = {
    "Generator": ("generators", "p", ("bus",)),
    "Link": ("links", "p", ("bus0", "bus1", "bus2", "bus3")),
    "Store": ("stores", "e", ("bus",)),
    "StorageUnit": ("storage_units", "p", ("bus",)),
    "Line": ("lines", "s", ("bus0", "bus1")),
    "Transformer": ("transformers", "s", ("bus0", "bus1")),
}


def _signed_operation_requirement(operation, lower_pu, upper_pu, component):
    """Return nominal capacity required by a signed operational time series."""
    positive = operation.gt(0.0)
    negative = operation.lt(0.0)
    allowed_positive = positive & upper_pu.gt(0.0)
    allowed_negative = negative & lower_pu.lt(0.0)
    positive_requirement = operation.where(allowed_positive).div(
        upper_pu.where(allowed_positive)
    )
    negative_requirement = operation.where(allowed_negative).div(
        lower_pu.where(allowed_negative)
    )
    return (
        pd.concat(
            [positive_requirement.max(axis=0), negative_requirement.max(axis=0)], axis=1
        )
        .max(axis=1)
        .fillna(0.0)
    )


def operational_nominal_requirement(n, component, assets):
    """Return capacity needed to reproduce each asset's reference operation."""
    assets = pd.Index(assets)
    if assets.empty:
        return pd.Series(dtype=float, index=assets)

    if component == "Generator":
        operation = n.generators_t.p.reindex(columns=assets).astype(float)
        lower = n.get_switchable_as_dense("Generator", "p_min_pu", inds=assets).astype(
            float
        )
        upper = n.get_switchable_as_dense("Generator", "p_max_pu", inds=assets).astype(
            float
        )
        return _signed_operation_requirement(operation, lower, upper, component)

    if component == "Link":
        operation = n.links_t.p0.reindex(columns=assets).astype(float)
        lower = n.get_switchable_as_dense("Link", "p_min_pu", inds=assets).astype(float)
        upper = n.get_switchable_as_dense("Link", "p_max_pu", inds=assets).astype(float)
        return _signed_operation_requirement(operation, lower, upper, component)

    if component == "Store":
        operation = n.stores_t.e.reindex(columns=assets).astype(float)
        lower = n.get_switchable_as_dense("Store", "e_min_pu", inds=assets).astype(
            float
        )
        upper = n.get_switchable_as_dense("Store", "e_max_pu", inds=assets).astype(
            float
        )
        return _signed_operation_requirement(operation, lower, upper, component)

    if component == "StorageUnit":
        operation = n.storage_units_t.p.reindex(columns=assets).astype(float)
        lower = n.get_switchable_as_dense(
            "StorageUnit", "p_min_pu", inds=assets
        ).astype(float)
        upper = n.get_switchable_as_dense(
            "StorageUnit", "p_max_pu", inds=assets
        ).astype(float)
        power_requirement = _signed_operation_requirement(
            operation, lower, upper, component
        )

        state_of_charge = n.storage_units_t.state_of_charge.reindex(
            columns=assets
        ).astype(float)
        max_hours = n.storage_units.loc[assets, "max_hours"].astype(float)
        invalid = state_of_charge.gt(0.0) & max_hours.le(0.0)
        if invalid.any().any():
            raise ValueError(
                "StorageUnit: positive reference state of charge has non-positive "
                "max_hours."
            )
        energy_requirement = state_of_charge.div(max_hours.where(max_hours.gt(0.0)))
        return pd.concat(
            [power_requirement, energy_requirement.max(axis=0).fillna(0.0)], axis=1
        ).max(axis=1)

    if component in {"Line", "Transformer"}:
        list_name = "lines" if component == "Line" else "transformers"
        operation = getattr(n, f"{list_name}_t").p0.reindex(columns=assets).abs()
        upper = n.get_switchable_as_dense(component, "s_max_pu", inds=assets).astype(
            float
        )
        allowed = operation.gt(0.0) & upper.gt(0.0)
        return (
            operation.where(allowed).div(upper.where(allowed)).max(axis=0).fillna(0.0)
        )

    raise ValueError(f"Unsupported fixed-neighbor component {component!r}.")


def asset_countries(n, static, bus_columns):
    countries = {}
    for asset, row in static.iterrows():
        asset_countries = set()
        for column in bus_columns:
            if column not in row.index:
                continue
            bus = row[column]
            if not isinstance(bus, str) or bus not in n.buses.index:
                continue
            country = n.buses.at[bus, "country"]
            if pd.notna(country) and country:
                asset_countries.add(str(country))
        countries[asset] = asset_countries
    return countries


def extract_capacity_rows(network_path, year, domestic_country):
    n = pypsa.Network(network_path)
    rows = []
    for component, (list_name, nominal_attr, bus_columns) in COMPONENTS.items():
        static = getattr(n, list_name)
        active = static.get(
            "active", pd.Series(True, index=static.index, dtype=bool)
        ).fillna(False)
        countries = asset_countries(n, static, bus_columns)
        nominal_column = f"{nominal_attr}_nom"
        optimized_column = f"{nominal_column}_opt"
        nominal = static.get(optimized_column, static[nominal_column]).fillna(
            static[nominal_column]
        )

        external_assets = pd.Index(
            [
                asset
                for asset in static.index[active]
                if countries[asset]
                and any(country != domestic_country for country in countries[asset])
            ]
        )
        operational_nominal = operational_nominal_requirement(
            n, component, external_assets
        )

        for asset in external_assets:
            connected_countries = countries[asset]
            rows.append(
                {
                    "year": year,
                    "component": component,
                    "asset": str(asset),
                    "nominal_attribute": nominal_attr,
                    "nominal": nominal.at[asset],
                    "operational_nominal": operational_nominal.at[asset],
                    "countries": ";".join(sorted(connected_countries)),
                }
            )
    return rows


if "snakemake" in globals():
    workflow = globals()["snakemake"]
    rows = []
    for year, network_path in {
        2030: workflow.input.network_2030,
        2035: workflow.input.network_2035,
    }.items():
        rows.extend(
            extract_capacity_rows(network_path, year, workflow.params.domestic_country)
        )

    manifest = pd.DataFrame(rows).sort_values(["year", "component", "asset"])
    output = Path(workflow.output.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(f"Wrote {len(manifest)} fixed-neighbor capacity targets to {output}.")
