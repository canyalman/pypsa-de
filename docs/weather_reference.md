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
python -m pytest test/test_weather_reference.py -q
```

These tests use synthetic in-memory load data, not downloaded weather files.
They cover all 15 country columns, 2013 load preservation, synthetic gap
filling from the source year, leap days, missing-data rejection, schema
validation, cutout routing and resource isolation.

Real weather-profile generation and solver optimality must still be checked
on the cluster. A local full Snakemake dry-run may also require connectivity
and trusted certificates for archive manifest checks.

