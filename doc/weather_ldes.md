# Controlled LDES weather runs (2035)

| Branch | Prefix | Weather | Available storage |
| --- | --- | --- | --- |
| thesis-weather-iron-air-1996 | thesis_weather_iron_air_1996 | 1996 | Li-ion + iron-air |
| thesis-weather-iron-air-2010 | thesis_weather_iron_air_2010 | 2010 | Li-ion + iron-air |
| thesis-weather-iron-air-2023 | thesis_weather_iron_air_2023 | 2023 | Li-ion + iron-air |
| thesis-weather-caes69-1996 | thesis_weather_caes69_1996 | 1996 | Li-ion + CAES69 |
| thesis-weather-caes69-2010 | thesis_weather_caes69_2010 | 2010 | Li-ion + CAES69 |
| thesis-weather-caes69-2023 | thesis_weather_caes69_2023 | 2023 | Li-ion + CAES69 |

Each branch enables four independent 2035 capacity-expansion cases:

- `KN2045_Mix_FixedRenewables`: endogenous Li-ion and the selected LDES technology.
- `KN2045_Mix_FixedRenewables_BESS_15GW`: 15 GW Li-ion, endogenous LDES.
- `KN2045_Mix_FixedRenewables_BESS_35GW`: 35 GW Li-ion, endogenous LDES.
- `KN2045_Mix_FixedRenewables_BESS_55GW`: 55 GW Li-ion, endogenous LDES.

This gives 24 solves across six branches: six endogenous cases and 18 fixed-BESS
cases. These are not dispatch-only runs. No Existing pathway or combined
iron-air-plus-CAES scenario is active. The original endogenous case is unchanged.

The fixed pathways target total German utility-scale Li-ion grid-facing AC
output power, including inherited utility batteries. Four-hour sizing implies
60/140/220 GWh AC-equivalent deliverable energy. Internal Store energy is not
independently fixed to those numbers: the native efficiency-aware sizing
constraint sets it consistently. Residential batteries are not part of the
utility target. Only the fixed pathways override the Li-ion duration exclusion
and add the two-sided AC power target; their other assumptions match the
endogenous case. The selected LDES technology remains extendable in all cases.

## Common experiment

The parent [controlled-weather experiment](controlled_weather.md) is retained.
Only 2035 is optimised, using the common solved 2030 history and 2035 fleet:
`cluster-results/endo_4h_3h_2025base_fixedgas_2026-09-14/KN2045_Mix_FixedRenewables/networks/`.
The inputs are `base_s_27__none_2030.nc` and `base_s_27__none_2035.nc`.
Neither reference is modified or re-solved.

Non-DE physical nominal capacities remain statically fixed. German renewable
and hydro/PHS assets stay fixed as in the parent; other permitted German
investment decisions and dispatch/trade remain endogenous. The existing bounded
foreign final-heat shortfall and audited zero-CAPEX foreign oil-accounting
connector exception are retained. They do not reopen physical foreign investment.

All families retain the fixed 2035 CO2 price of 308.2489253762274 EUR/tCO2,
three-hour resolution, 2920 steps/8760 hours and established leap-day handling.
The original endogenous four-hour Li-ion sizing, residential batteries, BEV DSM
0.20/0.35, V2G off, solver settings and other model assumptions are unchanged.
Base electricity load and nuclear availability use 2013; weather-sensitive
renewable, hydro, heat, COP and other native profiles use the target weather.
Common 2030 assets receive target-weather profiles at their proper vintage;
old dispatch and SOC results are not carried over. The 2025/2030 horizon entries
remain for native vintage and preprocessing logic, not additional optimisations.

## Technology settings

The existing technology assumptions are copied, not changed or reinterpreted
as part of this weather experiment.

- Iron-air: German AC buses only; 100 hours at rated AC output; charging
  efficiency 0.71 and discharging efficiency 0.60; equal grid-side charging and
  discharging power. The combined 1547.09 EUR2020/kW AC installed cost is charged
  once to the charger, with 1% FOM, 17.5-year lifetime and 7% discount rate.
  Existing source currency-year metadata is preserved. The configuration is
  copied from `iron-air-fixed-nonde-3h-2025base`; no new currency conversion,
  repowering, degradation or cost revision is introduced.
- CAES69: the existing continuous RESC model, available from 2035 at the four
  configured German sites. RTE is 0.69 with the existing symmetric efficiency
  split and 1:1 grid-side charging/discharging power ratio. Output duration is
  endogenous between 8 and 48 hours; the 7.6 TWh national electrical-output
  geological potential and regional limits are retained. Lifetime is 30 years,
  discount rate 7%; all USD2022 cost coefficients, conversion factors and source
  metadata are copied from `caes-fixed-nonde-3h-2025base-rte69`.

Additional domestic LDES internal buses need not exist in the Li-ion-only
reference. Preflight accepts them only for an enabled storage technology at an
allowed common German AC bus (and a configured CAES site), with one matching
extendable Store, the two correctly connected extendable Links and configured
efficiencies. Other connected assets, extra ports, missing reference buses and
unrecognised extra buses remain errors. Fixed-fleet capacities are not relaxed.

The source technology functions are ported narrowly into the controlled weather
code. In particular, the explicit nuclear availability year, calendar alignment,
static reference validation, heat/oil accounting and workflow/cache repairs are
not replaced with older shared code from the technology branches.

## Cluster preparation

Use a separate idle checkout. Do not update or switch a checkout underneath a
running workflow. Reuse compatible profiles from
`resources/thesis_weather_<year>/KN2045_Mix_FixedRenewables`, or change only
`controlled_weather.weather_resources_directory` if that cache is elsewhere.
Only recognised, complete non-network output groups are reused; investment-bearing
weather networks are never copied. Missing profiles use the normal archive
workflow. No local weather download or full-size solve is required for branch
preparation.

Generate scenarios once:

```bash
./.pixi/envs/default/bin/snakemake build_scenarios --cores 1 \
  --forcerun build_scenarios --allowed-rules build_scenarios --retries 2
```

Run `controlled_weather_preflight` with the usual Slurm settings first. All four
scenario reports must say `passed_without_solve`. Then request the four final-year
networks for the selected branch. For iron-air 1996:

```text
results/thesis_weather_iron_air_1996/KN2045_Mix_FixedRenewables/networks/base_s_27__none_2035.nc
results/thesis_weather_iron_air_1996/KN2045_Mix_FixedRenewables_BESS_15GW/networks/base_s_27__none_2035.nc
results/thesis_weather_iron_air_1996/KN2045_Mix_FixedRenewables_BESS_35GW/networks/base_s_27__none_2035.nc
results/thesis_weather_iron_air_1996/KN2045_Mix_FixedRenewables_BESS_55GW/networks/base_s_27__none_2035.nc
```

Use the same four scenario suffixes with `thesis_weather_caes69_1996` in its
separate CAES checkout. Replace the prefix year for 2010/2023. Each target is
an independent optimisation, not a step in a 15-to-35-to-55 GW investment chain.
Use explicit 2035 targets rather than the default all-years plotting target.
No new 2025/2030 solve, reference edit or forced regeneration of ready weather
profiles is required solely for this port.
Preserve hour units and Slurm status tracking in
[Weather cluster runs](weather_cluster_runs.md).

Compare the endogenous case with the corresponding Li-ion-only controlled
weather result. Compare each fixed-BESS case with the same-year, same-GW
Li-ion-only case in `thesis-weather-bess-<year>`. Within this branch, compare
15/35/55 GW to assess deployment dependence with the same LDES option available.
German renewable locations are pinned to the common reference in this design;
before comparing older 2013 LDES results, also check that asset-level restriction
rather than assuming equal national totals imply identical siting.
For resource costs restore fixed CAPEX and exclude CO2 payments and
unserved-heat penalties using `controlled_weather_result`; report any unserved
heat alongside those costs. A raw objective is not an aligned resource cost.
Accept a result only with scheduler completion, optimal solver termination,
successful Python finalisation and the exported network.

## Local checks

Run the common suite:

```bash
python -m pytest test/test_weather_ldes_preflight.py \
  test/test_weather_ldes_pathways.py test/test_weather_ldes.py \
  test/test_controlled_weather.py \
  test/test_controlled_weather_adequacy.py test/test_weather_workflow.py \
  test/test_controlled_weather_heat_profiles.py test/test_fixed_co2_price.py \
  test/test_fixed_neighbor_constraints.py test/test_battery_ac_pathways.py \
  test/test_german_conventional_capacity_pathway.py -q
```

CAES branches additionally include `test/test_acaes_continuous.py`. Tests check
config/schema and reference consistency, production preparation calls and solve
hooks, endogenous LDES sizing and efficiency, legacy/vintage behaviour, small
LPs and the real post-solve metadata/finalizer/NetCDF/YAML publication flow.
Pathway tests exercise the production scenario provider, verify no target leakage
into the endogenous case, solve coupled Li-ion/LDES toy LPs and build a real
four-scenario preflight/cache DAG. LDES preflight tests also run the actual script
after technology attachment, native vintage labelling and NetCDF export/import
for all four cases, and reject malformed internal-storage connections. They do
not replace full-size cluster preflight and solve validation.
