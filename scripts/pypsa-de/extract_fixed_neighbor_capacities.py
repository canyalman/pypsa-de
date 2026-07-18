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

        for asset in static.index[active]:
            connected_countries = countries[asset]
            if not connected_countries or not any(
                country != domestic_country for country in connected_countries
            ):
                continue
            rows.append(
                {
                    "year": year,
                    "component": component,
                    "asset": str(asset),
                    "nominal_attribute": nominal_attr,
                    "nominal": nominal.at[asset],
                    "countries": ";".join(sorted(connected_countries)),
                }
            )
    return rows


rows = []
for year, network_path in {
    2030: snakemake.input.network_2030,
    2035: snakemake.input.network_2035,
}.items():
    rows.extend(
        extract_capacity_rows(network_path, year, snakemake.params.domestic_country)
    )

manifest = pd.DataFrame(rows).sort_values(["year", "component", "asset"])
output = Path(snakemake.output.manifest)
output.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(output, index=False)
print(f"Wrote {len(manifest)} fixed-neighbor capacity targets to {output}.")
