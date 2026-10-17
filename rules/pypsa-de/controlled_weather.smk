# SPDX-License-Identifier: MIT

from scripts.controlled_weather import cached_weather_resources

controlled_settings = config.get("controlled_weather", {})
if controlled_settings.get("enable", False):
    if controlled_settings.get("mode") == "dispatch":
        rule prepare_dispatch_weather_network:
            input:
                weather=resources("networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_final.nc"),
                reference=controlled_settings["reference_network"],
                helper="scripts/dispatch_weather.py",
                calendar_helper="scripts/controlled_weather.py",
            output:
                network=resources("networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}_dispatch.nc"),
                ledger=resources("dispatch_weather_fleet_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv"),
            wildcard_constraints:
                planning_horizons="2035",
            log:
                logs("prepare_dispatch_weather_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log"),
            params:
                settings=config_provider("controlled_weather"),
            threads: 1
            resources:
                mem_mb=64000,
                runtime="4h",
            script:
                scripts("pypsa-de/prepare_dispatch_weather.py")

    rule controlled_weather_preflight:
        input:
            expand(
                resources("controlled_weather_preflight_{clusters}_{opts}_{sector_opts}_2035.json"),
                run=config["run"]["name"],
                clusters=config["scenario"]["clusters"],
                opts=config["scenario"]["opts"],
                sector_opts=config["scenario"]["sector_opts"],
            ),

    rule check_controlled_weather_network:
        input:
            network=weather_solve_input,
            fixed_neighbor_capacities=fixed_neighbor_capacity_manifest,
            controlled_weather_reference=controlled_settings["reference_network"],
            helper="scripts/controlled_weather.py",
            dispatch_helper="scripts/dispatch_weather.py",
            additional="scripts/pypsa-de/additional_functionality.py",
        output:
            resources("controlled_weather_preflight_{clusters}_{opts}_{sector_opts}_{planning_horizons}.json"),
        wildcard_constraints:
            planning_horizons="2035",
        log:
            logs("controlled_weather_preflight_{clusters}_{opts}_{sector_opts}_2035.log"),
        params:
            solving=config_provider("solving"),
        threads: 1
        resources:
            mem_mb=64000,
            runtime="4h",
        script:
            scripts("pypsa-de/check_controlled_weather.py")

    if controlled_settings.get("mode") == "dispatch":
        rule weather_dispatch_preflight:
            input:
                rules.controlled_weather_preflight.input,

    source_weather_resources = Path(controlled_settings["weather_resources_directory"])
    weather_resource_names = cached_weather_resources(source_weather_resources)
    if weather_resource_names:
        import re

        rule reuse_controlled_weather_resource:
            input:
                source=lambda w: str(source_weather_resources / w.controlled_weather_resource),
            output:
                resources("{controlled_weather_resource}"),
            wildcard_constraints:
                controlled_weather_resource="(?:" + "|".join(map(re.escape, weather_resource_names)) + ")",
            message:
                "Reusing compatible weather preprocessing: {wildcards.controlled_weather_resource}",
            script:
                scripts("pypsa-de/reuse_controlled_weather_resource.py")

        localrules: reuse_controlled_weather_resource

        for existing_rule in list(workflow.rules):
            if existing_rule.name != "reuse_controlled_weather_resource":
                workflow.ruleorder("reuse_controlled_weather_resource", existing_rule.name)
