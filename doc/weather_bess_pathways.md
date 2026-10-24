# BESS pathways under controlled weather (2035)

| Branch | Prefix | Weather | Active utility BESS targets |
| --- | --- | --- | --- |
| thesis-weather-bess-1996 | thesis_weather_bess_1996 | 1996 | 15 / 35 / 55 GW AC |
| thesis-weather-bess-2010 | thesis_weather_bess_2010 | 2010 | 15 / 35 / 55 GW AC |
| thesis-weather-bess-2023 | thesis_weather_bess_2023 | 2023 | 15 / 35 / 55 GW AC |

Each branch has only `KN2045_Mix_FixedRenewables_BESS_15GW`,
`KN2045_Mix_FixedRenewables_BESS_35GW` and
`KN2045_Mix_FixedRenewables_BESS_55GW` active. These are nine new final-year
optimisations in total, not dispatch-only stress tests. The endogenous parent
scenario remains defined as a template but is not an active run. No iron-air,
CAES or Existing pathway is enabled.

## Same experiment, prescribed German BESS

The parent [controlled-weather experiment](controlled_weather.md) is retained:
common 2030 investment history, common 2035 non-DE physical fleet, fixed German
renewable/hydro fleet and free dispatch/trade. Other permitted German investments
remain endogenous. All branches use the same reference directory:

```text
cluster-results/endo_4h_3h_2025base_fixedgas_2026-09-14/KN2045_Mix_FixedRenewables/networks/
```

`base_s_27__none_2030.nc` supplies history; `base_s_27__none_2035.nc` supplies the
fixed fleet and carbon-price reference. Neither file is edited. The fixed carbon
price stays 308.2489253762274 EUR/tCO2, replacing the EU and national emissions
budgets only in the 2035 solve. Do not use a different foreign reference or carbon
price for each weather year or BESS target.

The only investment assumption changed from the parent is total German utility
Li-ion capacity: 15, 35 or 55 GW of grid-facing AC output, including inherited
batteries. Germany-wide siting remains optimised. The existing four-hour fleet
equation gives 60, 140 or 220 GWh AC-equivalent rated energy. Internal Store energy
is larger because discharge efficiency is below one; it must not be independently
fixed to those AC-equivalent values. Residential storage and non-DE storage are
not part of these targets. The observed German 2025 battery fleet remains exempt
from the four-hour sizing equation, as in the established AC-output pathways.

Three-hour resolution, 2920 snapshots/8760 hours (including the established
leap-day treatment), 2013 base electricity load and nuclear availability,
weather-dependent renewable/hydro/heat/COP profiles, BEV DSM 0.20/0.35, V2G off,
costs/lifetimes and solver settings are unchanged. Only 2035 is solved; the
2025/2030 horizon entries remain for native vintage/preprocessing logic.

The parent bounded foreign final-heat shortfall and audited zero-CAPEX foreign
transport-oil accounting adjustments are retained. Report any unserved heat and
its penalty separately. For resource-cost comparisons restore the fixed fleet
CAPEX constants and exclude carbon payments and shortage penalties using the
existing `controlled_weather_result` accounting; raw objectives are not directly
comparable resource costs. No additional physical foreign investment is released.

## Cluster preparation

Use a separate idle checkout; do not switch or update an active workflow's
checkout. Place the common reference networks at the paths above. Compatible
weather preprocessing is reused from
`resources/thesis_weather_<year>/KN2045_Mix_FixedRenewables`, as in the parent
branches. Change only `controlled_weather.weather_resources_directory` if the
cache is elsewhere on the cluster. Cached investment-bearing networks are never
used. No local weather download or full-size solve is needed to prepare a branch.

Generate scenario overrides for all three active pathways:

```bash
./.pixi/envs/default/bin/snakemake build_scenarios --cores 1 \
  --forcerun build_scenarios --allowed-rules build_scenarios --retries 2
```

Run `controlled_weather_preflight` with the established Slurm submission settings.
It now expands to three separate reports, one per BESS scenario. It validates
the actual prepared weather/fleet inputs without solving an LP; the optimisation
adds the AC-output and duration constraints. Each report must say
`passed_without_solve`.

Request these final-year network targets using the usual Slurm command rather
than the default all-years plotting target (1996 example):

```text
results/thesis_weather_bess_1996/KN2045_Mix_FixedRenewables_BESS_15GW/networks/base_s_27__none_2035.nc
results/thesis_weather_bess_1996/KN2045_Mix_FixedRenewables_BESS_35GW/networks/base_s_27__none_2035.nc
results/thesis_weather_bess_1996/KN2045_Mix_FixedRenewables_BESS_55GW/networks/base_s_27__none_2035.nc
```

Replace only the prefix year for the other two branches. No 2025/2030 solve or
reference-manifest modification is required. Preserve the explicit hour units
and scheduler-status tracking documented in [Weather cluster runs](weather_cluster_runs.md).
Accept a cluster result only after checking Slurm completion/exit code, optimal
solver termination, successful Python finalisation and the exported network.

## Local regression tests

```bash
python -m pytest test/test_weather_bess_pathways.py test/test_battery_ac_pathways.py \
  test/test_controlled_weather.py test/test_controlled_weather_adequacy.py \
  test/test_weather_workflow.py test/test_controlled_weather_heat_profiles.py \
  test/test_fixed_co2_price.py test/test_fixed_neighbor_constraints.py \
  test/test_german_conventional_capacity_pathway.py -q
```

These tests use synthetic small networks and cached-file DAG fixtures, not a
full-size cluster optimisation. They check config/schema consistency, unchanged
parent assumptions, direct common-2030 routing, AC power including inherited
batteries, four-hour energy, three independent preflights and safe weather reuse.
