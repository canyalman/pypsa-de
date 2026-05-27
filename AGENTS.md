# Project context

This is my PyPSA-DE thesis project.

Goal:
Assess the system impacts of different battery expansion pathways in Germany towards 2035.

Main modeling idea:
- Use PyPSA-DE.
- Focus on Germany.
- Compare low / medium / high battery expansion scenarios.
- Battery scenarios should include both power capacity in GW and energy capacity in GWh.
- Ideally keep neighboring countries fixed from a baseline run.
- Allow battery siting inside Germany if possible.
- Analyze effects on gas/H2 backup dispatch, curtailment, prices, system costs, and system operation.

My skill level:
I usually run PyPSA-DE from config files and Snakemake commands.
I am not yet comfortable with deep code modifications.
Prefer config-only solutions or minimal script changes.

Important instructions:
- Do not edit files unless I explicitly ask.
- First explain which files are relevant and why.
- Focus on config, workflow, scripts, and Snakefile.
- Ignore results, resources, cutouts, data, .nc files, logs, and generated outputs.
- Do not change unrelated model assumptions.
- Before suggesting changes, explain the modeling logic.
- Before editing files, show the planned diff or describe the exact intended changes.