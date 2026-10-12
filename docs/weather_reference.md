# Weather-year endogenous reference runs

These branches start from commit 573ba7cb, matching the selected
endo_4h_3h_2025base_fixedgas reference assumptions.

| Branch | Results prefix | Weather cutout |
| --- | --- | --- |
| thesis-weather-1996 | thesis_weather_1996 | europe-1996-sarah3-era5 |
| thesis-weather-2010 | thesis_weather_2010 | europe-2010-sarah3-era5 |
| thesis-weather-2023 | thesis_weather_2023 | europe-2023-sarah3-era5 |

## Experiment

- Solve the complete myopic chain: 2025, 2030, 2035.
- Utility Li-ion investment remains endogenous in 2030 and 2035.
- Non-DE investment remains endogenous in these reference runs.
- Keep the original CO2 budgets, renewable targets, conventional fleet,
  technology costs and lifetimes.
- Keep three-hour resolution and the original four-hour utility battery logic.
- Keep BEV DSM availability at 0.20 in 2030 and 0.35 in 2035, with V2G off.
- Keep German gas investment fixed in 2025 and available in later years.
- In the 1996 branch, select the 2013 nuclear availability column only to let
  the intermediate add_electricity step complete; the CSV has no 1996 column.
- No iron-air, CAES or 15/35/55 GW deployment pathways are added here.

## Electricity load and weather

The historical base electricity-load input uses 2013 in every weather case.
This fixes both the initial hourly values and their annual country totals.
It does not freeze total future electricity consumption: the existing
model-year industry, mobility and sector-coupling calculations are unchanged.

The electricity-demand script first selects the source-year calendar,
including synthetic supplementation, and then assigns the target dates.
Missing country/hour data is rejected before writing the output.

Wind, solar, hydro, temperature, heating profiles, heat-pump COP and other
weather-dependent calculations use the selected weather year. The
temperature-dependent heating effects remain part of the weather sensitivity.

February 29 is removed under the existing enable.drop_leap_day setting,
including in 1996: 8760 hourly input steps, or 2920 three-hour model snapshots.
Mapping the 2013 load by month/day/hour preserves its values, but does not
realign its weekday/holiday pattern to each target calendar.

Weather-dependent resources are not shared with the 2013 baseline or the
other weather branches. All explicit renewable and line-rating cutout
references match the selected year.

## Cluster workflow

Generate scenarios again after checking out the branch:

```bash
./.pixi/envs/default/bin/snakemake build_scenarios --cores 1 \
  --forcerun build_scenarios --allowed-rules build_scenarios --retries 2
```

The 2035 target is, for example:

```text
results/thesis_weather_1996/KN2045_Mix_FixedRenewables/networks/base_s_27__none_2035.nc
```

Request that target with the established cluster submission command.
The workflow builds the 2025 and 2030 predecessors for the same prefix;
do not substitute previously solved 2013-weather networks.

The data.cutout source remains archive. The workflow retrieves the prepared
cutout instead of building ERA5/SARAH weather data locally. If compute nodes
lack internet access, retrieve the cutout on the login node before submission.

Later fixed-non-DE deployment scenarios should use the matching weather-year
reference network, not the original 2013-weather reference.

## Local checks

```bash
python -m pytest test/test_weather_reference.py test/test_nuclear_availability.py -q
```

These tests use synthetic in-memory load data, not downloaded weather files.
They cover all 15 country columns, 2013 load preservation, synthetic gap
filling from the source year, leap days, missing-data rejection, schema
validation, cutout routing and resource isolation.

Nuclear tests exercise add_electricity on small synthetic networks. They check
2013 availability in all weather calendars, missing-year errors, schema validation
and unchanged default/scalar/non-nuclear attribute behavior.

## Resuming the failed 1996 run

Pull thesis-weather-1996, then resubmit the same target with the established
cluster command and `--rerun-incomplete`. The failed add_electricity output must
be rebuilt, but existing weather profiles can be reused. Do not delete profiles,
change the prefix, or force all rules. There is no need to regenerate scenarios
for this correction because the scenario overrides have not changed.

The current config excludes nuclear from pypsa_eur.Generator. Consequently,
prepare_sector_network removes these intermediate nuclear Generators. The
existing nuclear fleet is added separately as Links without transferring this
CSV's availability factors. This fix unblocks 1996; it does not introduce a
new availability assumption into the final sector optimisation. The tests
verify that different intermediate nuclear availability factors are removed
while retained renewable Generators are unchanged.

Do not cancel or restart the 2010/2023 runs because of this correction: they
already pass add_electricity and these CSV factors do not reach their final
sector models. This correction is published only on thesis-weather-1996.

The nuclear setting is passed through the existing add_electricity conventional
params; no new workflow input or scenario-generation step is needed for this
correction. Missing requested years raise an explicit error, never use a
nearest/latest year. Non-year conventional attribute tables remain supported
when no explicit nuclear reference is requested.

Real weather-profile generation and solver optimality must still be checked
on the cluster. A local full Snakemake dry-run may also require connectivity
and trusted certificates for archive manifest checks.

