# SPDX-License-Identifier: MIT

"""Transfer weather to the fixed physical fleet with audited oil accounting."""

import pypsa

from scripts._helpers import configure_logging, set_scenario_config
from scripts.dispatch_weather import build_dispatch_weather

configure_logging(snakemake)  # noqa: F821
set_scenario_config(snakemake)  # noqa: F821
reference = pypsa.Network(snakemake.input.reference)  # noqa: F821
weather = pypsa.Network(snakemake.input.weather)  # noqa: F821
n, ledger = build_dispatch_weather(reference, weather, snakemake.params.settings)  # noqa: F821
summary = n.meta["dispatch_weather_prepared"]
n.meta = dict(
    snakemake.config,  # noqa: F821
    wildcards=dict(snakemake.wildcards),  # noqa: F821
    dispatch_weather_prepared=summary,
)  # noqa: F821
ledger.to_csv(snakemake.output.ledger, index=False)  # noqa: F821
n.export_to_netcdf(snakemake.output.network)  # noqa: F821
