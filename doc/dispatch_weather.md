# Fixed-fleet 2035 weather dispatch

For reproducible cache handling, preflight/solve commands and Slurm status
checks, see [Weather cluster runs](weather_cluster_runs.md).

| Branch | Prefix |
| --- | --- |
| thesis-weather-dispatch-1996 | thesis_weather_dispatch_1996 |
| thesis-weather-dispatch-2010 | thesis_weather_dispatch_2010 |
| thesis-weather-dispatch-2023 | thesis_weather_dispatch_2023 |

These are separate from the controlled capacity-expansion branches. Every
country's physical generation, conversion, storage power/energy and network
capacity is fixed from this solved reference. The audited non-physical oil
accounting exception is described below:

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
larger negatives are rejected. No physical capacity
headroom is added. Reference and prepared fingerprints are recorded separately
so the accounting exception remains visible. The prepared fleet fingerprint is
checked after export/import and
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

## Final oil accounting connectors

All three dispatch configs enable
`controlled_weather.resize_transport_oil_delivery: true`.
The native `land transport oil` Link delivers oil to a dedicated final-demand
bus and records its CO2 emissions. It has zero CAPEX and no physical capacity
investment trade-off. Its optimised nominal value in the reference is just the
2013-weather peak flow, not a fuel-infrastructure adequacy limit.

After transferring the unchanged target-weather demand profile, only such
zero-CAPEX, zero-length, unit-efficiency, dedicated accounting Links may have
their **fixed** nominal value raised to the new demand peak. This applies in
Germany too: unlike controlled capacity expansion, dispatch otherwise freezes
German accounting Links as well. No investment variable is reopened. A priced
investment, non-standard topology, incompatible efficiency or binding native
nominal upper limit is not silently relaxed.

Fuel consumption, marginal cost (including reference cost noise) and the CO2
output port remain intact. Reference NCs and manifests are not edited.
`oil_delivery_accounting_adjustments` in the prepared/final metadata lists every
asset, country, previous/applied capacity and reason; the fleet CSV retains both
reference and applied nominal values. Zero CAPEX leaves the fleet cost constant
unchanged. Other physical plant and network capacities remain at the reference.
Without this exception, unmet transport fuel demand could misleadingly appear
as a physical weather-related energy shortage.

## Unserved demand and cost reporting

An explicit diagnostic slack is enabled, with a penalty of 100000 EUR/MWh for
energy demand. It is attached only to buses with positive final demand and is
bounded by that actual demand in each snapshot. It is not unlimited new gas,
electricity, hydrogen or CO2 supply. Its generator carrier is
`weather demand shedding` and must be excluded from physical generation totals.
Any shortage must be reported, not presented as an adequately supplied system.
No stochastic outages or probabilistic LOLE are modelled.

This existing slack scope remains **all countries and all positive final-demand
carriers**, not just non-DE heat. No new heat-only slack or load reduction is
introduced. A finite high penalty discourages avoidable shedding but is not a
proof that every shortfall is technically unavoidable. It can affect dispatch
and prices even when excluded from reported costs.

After solving, `n.meta.dispatch_weather_result` records unserved energy by country
and bus carrier, its penalty, the operating objective, the operating objective
without that penalty, and operating cost plus the common fleet CAPEX constant.
Do not compare raw objectives to capacity-expansion objectives as total costs.
The native fixed-carbon-price metadata separately identifies CO2 payments.
The legacy `objective_including_reference_capex_eur` still includes the penalty
and CO2 payments. The new
`objective_including_reference_capex_excluding_unserved_penalty_eur` removes the
penalty, while
`objective_including_reference_capex_excluding_co2_and_unserved_penalty_eur`
additionally excludes carbon payments. Always report missing service alongside
these costs; lower costs with unserved demand are not equivalent to full service.

`unserved_energy_by_bus` and `unserved_energy_by_country_carrier` additionally
report demand, unserved energy, percentage, weighted shortage hours and peak
simultaneous shortage. Country/carrier peaks are calculated from summed hourly
shortfalls, not by summing each bus's individual peak. Keep heat, electricity
and fuel carriers separate when interpreting service adequacy.
`hours_with_any_unserved_final_demand` is the duration of any detected shortage,
not a probabilistic reliability metric. An active-hour threshold of 0.001 MW
(one kW) avoids counting solver noise; positive shortfall energy is still counted
below that threshold. Three-hour steps carry three hours. Energy/hour accounting
uses physical weights, while penalty and CO2 payments use objective weights.
Missing/non-finite solved series or shortfalls above demand fail reporting
rather than silently becoming zero.

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

For an existing checkout with ready scenarios and weather profiles, this repair
requires rerunning only `prepare_dispatch_weather_network`, the preflight and
the failed/new 2035 solve. The prepared `_dispatch.nc` must be regenerated to
include audited accounting connectors; reuse the unchanged `_2035_final.nc`
candidate and weather preprocessing. No new 2025/2030 solve, `build_scenarios`,
forced `add_brownfield` or weather-profile regeneration is required solely for
these changes. Do not update a checkout underneath active jobs.
The pre-model validator rejects an older prepared network when its recorded
accounting option differs from the current config, with an explicit instruction
to regenerate `_dispatch.nc` before any optimisation starts.
