# Fixed-fleet 2035 weather dispatch

| Branch | Prefix |
| --- | --- |
| thesis-weather-dispatch-1996 | thesis_weather_dispatch_1996 |
| thesis-weather-dispatch-2010 | thesis_weather_dispatch_2010 |
| thesis-weather-dispatch-2023 | thesis_weather_dispatch_2023 |

These are separate from the controlled capacity-expansion branches. Every
country's complete generation, conversion, storage power/energy and network
capacity is fixed from this solved reference:

```text
cluster-results/endo_4h_3h_2025base_fixedgas_2026-09-14/KN2045_Mix_FixedRenewables/networks/base_s_27__none_2035.nc
```

No 2025/2030 optimisation and no new 2035 investment is performed. The existing
2030 reference (`base_s_27__none_2030.nc` in the same reference directory) is
also required, only to assemble compatible weather-dependent inputs
through the native brownfield preprocessing. The resulting candidate network
supplies profiles, not the dispatch run's capacities. A separate preparation
rule starts from the complete solved 2035 network and transfers those inputs.
No fixed-neighbor manifest is needed or rebuilt for dispatch.

The 3H resolution, original 4H BESS formulation, residential batteries, plant
vintages/lifetimes, cyclic/initial storage conditions, 2035 BEV availability 0.35,
fuel costs and operational policy limits are retained. The fixed carbon price
is 308.2489253762274 EUR/tCO2; emissions budgets are replaced, while physical
sequestration constraints remain. Electricity demand retains the common 2013
base shape and 2035 demand assumptions. Renewable profiles, hydro inflows,
temperature-driven heat demand, heat-pump COPs and relevant weather/calendar
profiles follow the new year. Leap day is dropped, yielding 2920 3H steps.

Reference nominal capacities and costs are copied, never inferred from target
weather candidates. Solver-negative nominal residuals no larger than 0.05 are
clipped to zero and logged, following the existing nominal bound-clip allowance;
larger negatives are rejected. No new capacity
headroom is added. The fleet fingerprint is checked after export/import and
before solving; all six nominal variable groups must be non-extendable and the
LP must have no nominal investment variables. Missing positive-capacity profile
matches, bus country/carrier/location mismatches, invalid snapshots or NaNs fail
before solving. A spatial-clustering mismatch must be corrected, not relabelled.

Solved dispatch and SOC outputs are discarded. Original economic plant vintages
remain unchanged, and native profile selection preserves inherited wind-turbine
vintages and central heat-pump updates. Formerly extendable boilers do not gain
an existing-boiler profile restriction merely because their capacity is fixed.

`noisy_costs` is disabled only because the solved reference already contains
the native seeded perturbation; adding it again would change the cost inputs.

## Unserved demand and cost reporting

An explicit diagnostic slack is enabled, with a penalty of 100000 EUR/MWh for
energy demand. It is attached only to buses with positive final demand and is
bounded by that actual demand in each snapshot. It is not unlimited new gas,
electricity, hydrogen or CO2 supply. Its generator carrier is
`weather demand shedding` and must be excluded from physical generation totals.
Any shortage must be reported, not presented as an adequately supplied system.
No stochastic outages or probabilistic LOLE are modelled.

After solving, `n.meta.dispatch_weather_result` records unserved energy by country
and bus carrier, its penalty, the operating objective, the operating objective
without that penalty, and operating cost plus the common fleet CAPEX constant.
Do not compare raw objectives to capacity-expansion objectives as total costs.
The native fixed-carbon-price metadata separately identifies CO2 payments.
Full-year perfect foresight is retained; this is an operational stress test,
not a realistic trading backtest or a multi-weather robust investment design.

## Cluster steps

Keep the reference files above in the stated paths. The existing target-weather
cache is read from `controlled_weather.weather_resources_directory`, normally
`resources/thesis_weather_<year>/KN2045_Mix_FixedRenewables`. If that cache is
in a different cluster checkout, change only this path. Compatible non-network
resources are reused; missing inputs use the normal archive workflow. No local
weather download or full-size solve was performed when preparing these branches.

Generate scenario overrides once:

```bash
./.pixi/envs/default/bin/snakemake build_scenarios --cores 1 \
  --forcerun build_scenarios --allowed-rules build_scenarios --retries 2
```

Using the established Slurm command and `--retries 2`, first request
`weather_dispatch_preflight`. It prepares the fixed-fleet network and checks
inputs without constructing the LP. Its JSON must say `passed_without_solve`.
Then request, for example:

```text
results/thesis_weather_dispatch_1996/KN2045_Mix_FixedRenewables/networks/base_s_27__none_2035.nc
```

Replace the year for the other branches. Profile/brownfield/preflight preparation
requests 64 GB; keep the usual solve memory/time settings. Do not switch a
checkout used by a running workflow. A local test or preflight is not evidence
that the real cluster optimisation completed optimally.
