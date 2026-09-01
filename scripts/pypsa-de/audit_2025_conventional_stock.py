# SPDX-License-Identifier: MIT
"""Report pre-optimisation German electricity-output capacity by technology."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pypsa

THERMAL_PATTERN = (
    r"CCGT|OCGT|CHP|coal|lignite|gas|oil|waste|biomass|biogas|nuclear"
)

TECHNOLOGY_GROUPS = {
    "gas": ["CCGT", "OCGT", "urban central gas CHP"],
    "hard_coal": ["coal", "urban central coal CHP"],
    "lignite": ["lignite", "urban central lignite CHP"],
}


def german_link_stock(n: pypsa.Network) -> pd.DataFrame:
    links = n.links.copy()
    electric_buses = n.buses.index[
        n.buses.carrier.isin(["AC", "low voltage"])
        & n.buses.index.to_series().str.startswith("DE")
    ]
    links = links[
        links.bus1.isin(electric_buses)
        & links.carrier.str.contains(THERMAL_PATTERN, case=False, na=False)
        & links.p_nom.gt(0)
    ].copy()

    links["electric_output_mw"] = links.p_nom * links.efficiency
    links["electric_output_min_mw"] = links.p_nom_min * links.efficiency
    links["build_year"] = links.build_year.fillna(0).astype(int)

    return (
        links.groupby(["carrier", "build_year"], dropna=False)
        .agg(
            assets=("carrier", "size"),
            input_capacity_mw=("p_nom", "sum"),
            electric_output_mw=("electric_output_mw", "sum"),
            electric_output_min_mw=("electric_output_min_mw", "sum"),
            extendable_assets=("p_nom_extendable", "sum"),
        )
        .reset_index()
        .sort_values(["carrier", "build_year"], ignore_index=True)
    )


def german_conventional_summary(n: pypsa.Network) -> pd.DataFrame:
    links = n.links.copy()
    bus_country = links.bus1.map(n.buses.country)
    bus_country = bus_country.fillna(links.bus1.str[:2])
    links = links.loc[bus_country.eq("DE")].copy()
    links["electric_output_mw"] = links.p_nom * links.efficiency

    rows = []
    for technology, carriers in TECHNOLOGY_GROUPS.items():
        selected = links.loc[links.carrier.isin(carriers)]
        rows.append(
            {
                "technology": technology,
                "assets": len(selected),
                "electric_output_mw": selected.electric_output_mw.sum(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("network", type=Path, help="Pre-optimisation *_final.nc file")
    parser.add_argument("--output", type=Path, help="Optional output CSV")
    args = parser.parse_args()

    network = pypsa.Network(args.network)
    stock = german_link_stock(network)
    summary = german_conventional_summary(network)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stock.to_csv(args.output, index=False)

    display = stock.copy()
    for column in [
        "input_capacity_mw",
        "electric_output_mw",
        "electric_output_min_mw",
    ]:
        display[column] = display[column].round(3)
    summary["electric_output_mw"] = summary.electric_output_mw.round(3)
    print("German conventional capacity summary (MW_el):")
    print(summary.to_string(index=False))
    print("\nCarrier and build-year detail:")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
