# Controlled and fixed-fleet weather runs on Slurm

Use a dedicated **idle checkout** for each controller. Do not switch branches,
pull, patch, clean or cancel jobs in a checkout used by an active workflow.
The commands below are instructions; preparing the repository does not execute
them or access a cluster. Use the committed branch, not an untracked wrapper.

## One-time preparation

Select one of `thesis-weather-controlled-{1996,2010,2023}` or
`thesis-weather-dispatch-{1996,2010,2023}` in that dedicated checkout. Install its
locked Pixi environment on Linux, not in a Windows/WSL-shared `.pixi` directory.
The environment used for submission must provide the `cluster-generic` executor.
Do not modify solver/model settings as part of this workflow repair.

Both experiments require the common reference networks from:

```text
cluster-results/endo_4h_3h_2025base_fixedgas_2026-09-14/KN2045_Mix_FixedRenewables/networks/
```

Keep the paths in `config/config.de.yaml` intact. The 2030 and 2035 files are
required; controlled capacity manifests additionally list the 2025 file.
The existing target-weather preprocessing cache may be read from another
checkout via `controlled_weather.weather_resources_directory`. Do not change
the source files. All requested weather years and model assumptions remain
as configured in the branch.

The cache rule now accepts only complete native output groups. For example,
`base_network` must still build `networks/base.nc`, so its accompanying region
files are not claimed by a copy job. A complete renewable profile **and** its
class-region output can be copied together as needed, without rerunning the
expensive native profile job. Missing members cause that native output group
to be rebuilt. Investment-bearing networks and previous preflight reports are
never used as weather cache inputs.

Generate scenario overrides after pulling this repair:

```bash
./.pixi/envs/default/bin/snakemake build_scenarios --cores 1 \
  --forcerun build_scenarios --allowed-rules build_scenarios --retries 2
```

This is the sole scenario producer in these weather branches; `create_scenarios`
is inactive. Do not add a temporary Snakefile wrapper. No new scenario targets
or assumptions have been introduced; this generation step only makes a clean
checkout ready for the standard workflow.

## Preflight and Solve

Run the following in the chosen dedicated checkout. Keep the existing resource
allocation if the site's queue needs different memory/CPU limits. In particular,
**all time overrides below include `h`**: CLI `runtime=240` is not a portable
way to request four hours and was interpreted as minutes in the affected setup.

```bash
set -euo pipefail
mkdir -p logs/slurm

# Choose the branch's matching preflight target:
target=controlled_weather_preflight
# For dispatch branches instead: target=weather_dispatch_preflight

./.pixi/envs/default/bin/snakemake "$target" \
  --executor cluster-generic --jobs 10 --local-cores 1 \
  --latency-wait 120 --rerun-incomplete --rerun-triggers mtime --retries 2 \
  --cluster-generic-submit-cmd 'sbatch --parsable --cpus-per-task={threads} --mem={resources.mem_mb} --time={resources.runtime} --job-name="smk-{rule}-{wildcards}" --output=logs/slurm/%x-%j.out --chdir=$PWD --partition={resources.partition}' \
  --cluster-generic-status-cmd './.pixi/envs/default/bin/python scripts/pypsa-de/slurm_job_status.py' \
  --default-resources partition=standard mem_mb=64000 runtime=24h \
  --set-threads solve_sector_network_myopic=32 \
  --set-resources \
    build_renewable_profiles:mem_mb=64000 \
    build_renewable_profiles:runtime=4h \
    build_fixed_neighbor_capacity_manifest:mem_mb=64000 \
    build_fixed_neighbor_capacity_manifest:runtime=4h \
    add_brownfield:mem_mb=64000 \
    add_brownfield:runtime=8h \
    check_controlled_weather_network:mem_mb=64000 \
    check_controlled_weather_network:runtime=4h \
    prepare_dispatch_weather_network:mem_mb=64000 \
    prepare_dispatch_weather_network:runtime=4h \
    solve_sector_network_myopic:mem_mb=500000 \
    solve_sector_network_myopic:runtime=24h
```

Keep the controller in its own persistent terminal/session. Immediately inspect
the **actual** submitted job, rather than assuming the resource strings were
accepted correctly:

```bash
squeue --me -o "%.18i %.100j %.2t %.12M %.12l %.10m %.30R"
job=REPLACE_WITH_ACTUAL_JOB_ID
scontrol show job -o "$job"
sacct -X -j "$job" --format=JobID,JobName,State,Elapsed,Timelimit,ReqMem,ExitCode
```

Verify `TimeLimit`, memory, CPUs, partition and working directory. The expected
limits are `04:00:00`, `08:00:00` or `1-00:00:00` for the overrides above, not
`00:04:00`, `00:08:00` or `00:24:00`. Add any site-specific node exclusion to
the submit command deliberately; do not infer that it came from another profile.

The committed status helper reports `TIMEOUT`, `OUT_OF_MEMORY`, `FAILED`,
`CANCELLED` and other terminal failures as `failed`, allowing Snakemake to handle
failure/retry instead of waiting indefinitely for a success marker. Only the
parent job's `COMPLETED` **and** `0:0` produces `success`; a completed `.extern`
step is insufficient. Missing accounting data is temporarily `running`, never
assumed success. Investigate persistent accounting gaps manually.

Preflight must finish with a JSON status of `passed_without_solve`. Then rerun
the **same command** with `target` set to the matching 2035 result, for example:

```bash
target=results/thesis_weather_controlled_2023/KN2045_Mix_FixedRenewables/networks/base_s_27__none_2035.nc
# Dispatch example:
# target=results/thesis_weather_dispatch_2023/KN2045_Mix_FixedRenewables/networks/base_s_27__none_2035.nc
```

Use the branch's year in the prefix. To resume a failed brownfield/preflight,
keep its ready profiles and reference networks; do not force profile rebuilds.
Use `--rerun-incomplete --retries 2`. Do not stop unrelated running controllers.

An accepted final result needs all of: Slurm `COMPLETED`/`0:0`, a successful
solver termination, a successful Python/export log, and the expected result
network. Unit tests, a running controller, preflight, or a directory existing
are not evidence that the full model solved optimally.
