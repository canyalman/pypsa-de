# SPDX-License-Identifier: MIT

"""Validate the actual prepared network without constructing or solving an LP."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pypsa

from scripts._helpers import configure_logging, set_scenario_config
from scripts.controlled_weather import validate_input_series
from scripts.dispatch_weather import is_dispatch_weather

configure_logging(snakemake)  # noqa: F821
set_scenario_config(snakemake)  # noqa: F821
n = pypsa.Network(snakemake.input.network)  # noqa: F821
module_path = Path("scripts/pypsa-de/additional_functionality.py")
spec = importlib.util.spec_from_file_location(
    "controlled_weather_preflight", module_path
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report_path = Path(snakemake.output[0])  # noqa: F821
audit = SimpleNamespace(
    config=snakemake.config,  # noqa: F821
    params=snakemake.params,  # noqa: F821
    input=snakemake.input,  # noqa: F821
    wildcards=snakemake.wildcards,  # noqa: F821
    output=SimpleNamespace(
        network=str(report_path.parent / "preflight" / "networks" / "audit.nc"),
        fixed_neighbor_capex=str(
            report_path.parent / "preflight" / "fixed_neighbor_capex.csv"
        ),
    ),
)
module.prepare_network_before_model(n, audit)
validate_input_series(n)
if len(module.emissions_budget_constraints(n)):
    raise ValueError("An emissions budget remains in the fixed-price preflight.")
report = {
    "status": "passed_without_solve",
    "weather_year": int(n.snapshots[0].year),
    "snapshot_count": len(n.snapshots),
    "annual_objective_hours": float(n.snapshot_weightings.objective.sum()),
    "fixed_co2_price_eur_per_t": module.active_fixed_co2_price(audit),
    "component_counts": {
        c.name: len(c.static) for c in n.components if not c.static.empty
    },
}
if is_dispatch_weather(audit.config):
    report["dispatch_weather"] = n.meta["dispatch_weather_prepared"]
else:
    report["fixed_neighbor"] = n._fixed_neighbor_static_summary
    report["controlled_weather"] = n._controlled_weather_summary
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
