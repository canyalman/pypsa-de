# Controlled weather-year experiment (2035)

| Branch | Prefix | Weather |
| --- | --- | --- |
| thesis-weather-controlled-1996 | thesis_weather_controlled_1996 | 1996 |
| thesis-weather-controlled-2010 | thesis_weather_controlled_2010 | 2010 |
| thesis-weather-controlled-2023 | thesis_weather_controlled_2023 | 2023 |

Only `KN2045_Mix_FixedRenewables` is active. German utility Li-ion and gas
investment remain endogenous. There are no 15/35/55 GW pathways, iron-air or
CAES additions. Three-hour resolution, the reference's original four-hour
battery formulation (not the later AC-pathway override), residential storage,
costs/lifetimes, BEV DSM 0.20/0.35 and V2G off are retained.

## Common history, fleet and carbon price

All branches use this existing reference directory:

```text
cluster-results/endo_4h_3h_2025base_fixedgas_2026-09-14/KN2045_Mix_FixedRenewables/networks/
```

`base_s_27__none_2030.nc` supplies the common investment history. The 2035
reference supplies non-DE nominal capacities and German wind/solar/ror and
hydro/PHS capacities. German renewable node/asset/technology/vintage allocation
is fixed, not just national totals. Native bounds and small numerical clipping
checks are retained; meaningful mismatches fail rather than silently changing
the reference. No old dispatch or SOC solution is fixed.

Fixed location means the model's node/resource asset, not a reconstruction of
individual power-plant coordinates. Native within-node profile aggregation is
retained, including the original single resource class per technology.

Only 2035 is optimised. The historical planning-horizon list remains intact for
native vintage/base-year logic, but the previous-horizon input directly points
to the existing 2030 file. Neither 2025 nor 2030 needs to be solved again.

The fixed price is 308.2489253762274 EUR/tCO2, from that same reference's
`-CO2Limit.mu`. Both emissions budgets are replaced only in the 2035 solve;
physical sequestration limits remain. The German additional shadow price in
the saved reference is approximately 0.000050 EUR/tCO2 (numerically negligible).
The existing 2013 result is the control; no new 2013 solve is required merely
for this conversion.

## Weather and calendar handling

Base electricity load remains sourced from 2013. Target-year wind, solar,
hydro inflow, heating demand and COP profiles are generated/reused normally.
Fresh, **unsolved** 2025/2030-vintage sector networks supply the target-weather
profiles of inherited plants, preserving their technology vintage. These are
preprocessing steps, not new 2025/2030 optimisation runs. Native updating of
central heat pumps and dynamic pit storage to 2035 is retained.

Native renewable-profile vintage selection is applied to each unsolved source
network before transfer: a 2030 wind investment uses its 2030 turbine profile
under the target weather, not the earliest turbine profile in add_electricity.
The controlled brownfield job requests 64 GB because it loads these extra
profile networks in addition to the common history.

Brownfield inputs are aligned by month/day/time, not by row position. Old
dispatch, energy and SOC output series are discarded. Nominal capacities,
build years and lifetimes are unchanged. Missing/ambiguous profile matches,
different snapshot weights, NaNs and mismatched assets fail before solving.
The existing leap-day policy gives 8760 hourly inputs / 2920 three-hour steps
in each year, including 1996. The source load's weekday/holiday pattern is
not newly realigned to the target calendar.

`controlled_weather.weather_resources_directory` points to the corresponding
original weather preprocessing directory. If it is available in the same
checkout/shared filesystem, compatible non-network resources are copied into
the new prefix with timestamps preserved. Investment-bearing networks are
never reused from those weather runs. Sources are not modified. If that cache
is elsewhere, change this single path before submission. Missing resources
are built by the normal workflow from the archived cutout. Do not delete or
force regeneration of ready profiles.

## Cluster preparation and checks

Commit/push the prepared branch, fetch it in a separate cluster checkout,
and put the common reference files at the paths above. Do not switch a
checkout underneath another running Snakemake workflow.

Generate the scenario overrides once:

```bash
./.pixi/envs/default/bin/snakemake build_scenarios --cores 1 \
  --forcerun build_scenarios --allowed-rules build_scenarios --retries 2
```

First request `controlled_weather_preflight` with the established Slurm command
and `--retries 2`. It builds the actual 2035 prepared network and validates the
static fleet, common price, snapshots and profiles **without constructing or
solving the LP**. Its JSON report must say `passed_without_solve`.

Then request, for example:

```text
results/thesis_weather_controlled_1996/KN2045_Mix_FixedRenewables/networks/base_s_27__none_2035.nc
```

Use the corresponding prefix for 2010/2023. Keep the usual resource overrides;
if renewable preprocessing is needed, allow
`--set-resources build_renewable_profiles:mem_mb=64000`. A local unit-test pass
or preflight is not evidence of an optimal large-model solution: check the
cluster solver status and resulting network after the real solve.

## Accounting and local regression tests

The existing fixed-neighbor CAPEX ledger is retained. Fixed German renewable
CAPEX removed from the variable objective has its own ledger. After solving,
`n.meta.controlled_weather_result` records the objective corrected for both
removed investment-cost constants, with a separate carbon-payment-excluded
value. Do not compare old and new raw objectives directly.

Fixed-fleet capital costs are taken from the common reference as well; native
weather-dependent within-node weighting must not change the reported CAPEX of
an unchanged fleet. Variable dispatch costs and weather-dependent efficiencies
are not overwritten by this investment-cost correction.

```bash
python -m pytest test/test_controlled_weather.py test/test_fixed_co2_price.py \
  test/test_fixed_neighbor_constraints.py test/test_battery_ac_pathways.py \
  test/test_german_conventional_capacity_pathway.py -q
```
