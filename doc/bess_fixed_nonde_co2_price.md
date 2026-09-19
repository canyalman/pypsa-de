# Fixed non-DE BESS with a fixed carbon price

Branch: `bess-fixed-nonde-co2-price`  
Run prefix: `bess_fixed_nonde_co2_price`

## Experiment

Only the 15, 35 and 55 GW German utility BESS cases are active. From 2035 their
power targets are **AC output**, with 60, 140 and 220 GWh of **AC-equivalent rated
energy** at a 4-hour aggregate fleet duration. Temporal resolution remains 3 hours.

Each scenario selects `capacity_limit_basis.Link.battery discharger.DE.2035: output`.
The power constraint is `sum(efficiency_i * p_nom_i) = AC target`, including both
carried and new active dischargers at their own efficiencies. All other countries,
years and carriers retain their previous capacity-limit basis.

The old fixed internal Store target is removed only for German utility BESS in
2035. The existing aggregate duration constraint is enabled instead:
`sum(e_nom_j) = 4 * sum(p_nom_i)`. Consequently, the effective efficiency of the
rated power mix is `eta_fleet = sum(efficiency_i * p_nom_i) / sum(p_nom_i)`, and
`eta_fleet * sum(e_nom_j) = 4 * AC target`. This is a fleet-level rated-energy
convention; it does not add per-node duration constraints or guarantee four-hour
delivery under every SOC, siting or grid condition. No single guessed efficiency
is applied across vintages. Raw Store energy is internal energy and must not be
labelled AC-deliverable energy in plots.

All three cases reuse the solved 2025 and 2030 networks from:

`cluster-results/endo_4h_3h_2025base_fixedgas_2026-09-14/KN2045_Mix_FixedRenewables/networks/`

Required files: `base_s_27__none_2025.nc`, `base_s_27__none_2030.nc`, and
`base_s_27__none_2035.nc`. Reference files are read or copied, never modified.
Only 2035 is re-optimized. The German gas calibration and BEV DSM settings match
this reference (2030: 0.20; 2035: 0.35). Renewable targets, residential batteries,
coal retirement assumptions, storage duration rules and solver settings are
not otherwise changed.

Non-DE nominal capacities use the existing static materialization method,
including its numerical headroom/native-bound handling. Foreign dispatch and
trade remain endogenous. Snakemake automatically creates the new manifest
`resources/fixed_neighbor_endo_20260914_static_capacities.csv` from the reference
2030/2035 networks. Do not reuse the older September 2 manifest.

## Carbon policy

The 2035 carbon price is **308.2489253762274 EUR/tCO2**, equal to
`-global_constraints.loc["CO2Limit", "mu"]` in the reference 2035 network.
This is a model-implied shadow price, not an observed or forecast ETS price.

`costs.emission_prices.co2` is a year mapping. This uses PyPSA's atmospheric
Store accounting without also pricing electricity-stage generators. The Store
has `marginal_cost = -price`: emissions increase its energy stock and have
negative dispatch, so emissions incur a positive cost; net removals get a credit.

For 2035, `solving.constraints.fixed_co2_price` removes `CO2Limit` before model
creation and skips national CO2 budgets. A pre-solve check rejects any remaining
emissions cap. Physical sequestration limits are retained. Old cap values remain
in the generated scenario configuration for provenance, but are not constraints
of the 2035 fixed-price optimization. The historical 2025/2030 solves are unchanged.

## Costs and outputs

`n.meta["fixed_co2_price"]` records the applied price, net atmospheric emissions,
carbon payments, and objectives with/without those payments. Fields ending in
`including_fixed_nonde_capex` add the existing static CAPEX ledger back to the
raw objective. These are objective-basis accounting metrics, not a new full-fleet
valuation of all sunk investments.

Do not directly compare the taxed objective to the old capped benchmark's raw
objective. Compare resource-cost metrics excluding carbon payments, or revalue
the benchmark's emissions at the same carbon price for a tax-inclusive comparison.

## Cluster preparation

Keep the three reference files at the relative paths above. Generate scenarios
after switching to this branch, before submitting the usual Slurm workflow:

```bash
./.pixi/envs/default/bin/snakemake build_scenarios --cores 1 \
  --forcerun build_scenarios --allowed-rules build_scenarios
```

The three final targets are:

```text
results/bess_fixed_nonde_co2_price/KN2045_Mix_FixedRenewables_BESS_15GW/networks/base_s_27__none_2035.nc
results/bess_fixed_nonde_co2_price/KN2045_Mix_FixedRenewables_BESS_35GW/networks/base_s_27__none_2035.nc
results/bess_fixed_nonde_co2_price/KN2045_Mix_FixedRenewables_BESS_55GW/networks/base_s_27__none_2035.nc
```

Before Gurobi starts, logs should confirm `Fixed CO2 price`,
`Skipping national CO2 budgets`, and `Validated price-only CO2 policy`.
Accept final networks only after the normal solver-status and physical-bound checks.
