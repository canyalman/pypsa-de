import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xarray import DataArray

from scripts.prepare_sector_network import determine_emission_sectors

logger = logging.getLogger(__name__)


FIXED_NEIGHBOR_COMPONENTS = {
    "Generator": ("generators", "p", ("bus",)),
    "Link": ("links", "p", ("bus0", "bus1", "bus2", "bus3")),
    "Store": ("stores", "e", ("bus",)),
    "StorageUnit": ("storage_units", "p", ("bus",)),
    "Line": ("lines", "s", ("bus0", "bus1")),
    "Transformer": ("transformers", "s", ("bus0", "bus1")),
}


def _asset_countries(n, static, bus_columns):
    """Return all modelled countries connected to each asset."""
    countries = {}
    for asset, row in static.iterrows():
        asset_countries = set()
        for column in bus_columns:
            if column not in row.index:
                continue
            bus = row[column]
            if not isinstance(bus, str) or bus not in n.buses.index:
                continue
            country = n.buses.at[bus, "country"]
            if pd.notna(country) and country:
                asset_countries.add(str(country))
        countries[asset] = asset_countries
    return countries


def _external_asset_index(n, static, bus_columns, domestic_country):
    countries = _asset_countries(n, static, bus_columns)
    return pd.Index(
        [
            asset
            for asset, asset_countries in countries.items()
            if any(country != domestic_country for country in asset_countries)
        ]
    )


def _indexed_values(array, assets, label):
    """Return a one-dimensional Linopy/xarray value array as a Series."""
    values = array.to_pandas()
    if not isinstance(values, pd.Series):
        raise ValueError(f"{label} is not one-dimensional.")

    missing = assets.difference(values.index)
    if not missing.empty:
        raise ValueError(
            f"{label} is missing {len(missing)} fixed-neighbor assets, e.g. "
            f"{missing[:3].tolist()}."
        )
    return values.reindex(assets).astype(float)


def _model_nominal_bound_context(n, component, nominal_attr, assets):
    """Return effective Linopy nominal bounds and their model sources."""
    variable_name = f"{component}-{nominal_attr}_nom"
    variable = n.model[variable_name]
    lower = _indexed_values(variable.lower, assets, f"{variable_name}.lower")
    upper = _indexed_values(variable.upper, assets, f"{variable_name}.upper")
    lower = lower.fillna(-np.inf)
    upper = upper.fillna(np.inf)
    bound_constraints = {}

    # PyPSA 1.x represents nominal min/max mostly as constraints rather than
    # finite Variable.lower/upper values. Include their actual model RHS here.
    for side, expected_sign in (("lower", ">="), ("upper", "<=")):
        constraint_name = f"{component}-ext-{nominal_attr}_nom-{side}"
        if constraint_name not in n.model.constraints:
            continue

        constraint = n.model.constraints[constraint_name]
        rhs = constraint.rhs.to_pandas()
        signs = constraint.sign.to_pandas()
        if not isinstance(rhs, pd.Series) or not isinstance(signs, pd.Series):
            raise ValueError(f"{constraint_name} is not one-dimensional.")

        common = assets.intersection(rhs.index)
        if common.empty:
            continue
        invalid_sign = signs.reindex(common).ne(expected_sign)
        if invalid_sign.any():
            raise ValueError(
                f"{constraint_name} has an unexpected constraint sign for "
                f"{invalid_sign.sum()} assets."
            )

        constraint_bound = rhs.reindex(common).astype(float)
        bound_constraints[side] = constraint
        if side == "lower":
            lower.loc[common] = np.maximum(lower.loc[common], constraint_bound)
        else:
            upper.loc[common] = np.minimum(upper.loc[common], constraint_bound)

    return {
        "variable_name": variable_name,
        "variable": variable,
        "lower": lower,
        "upper": upper,
        "bound_constraints": bound_constraints,
    }


def _fixed_neighbor_bound_violations(
    targets, lower_bounds, upper_bounds, component, variable_name
):
    """Return all fixed-neighbor targets outside their model bounds."""
    lower_delta = (lower_bounds - targets).where(targets < lower_bounds, 0.0)
    upper_delta = (targets - upper_bounds).where(targets > upper_bounds, 0.0)
    delta = pd.concat([lower_delta, upper_delta], axis=1).max(axis=1)
    violated = delta.gt(0)
    if not violated.any():
        return pd.DataFrame(
            columns=[
                "component",
                "asset",
                "variable",
                "reference_value",
                "lower_bound",
                "upper_bound",
                "delta",
            ]
        )

    assets = delta.index[violated]
    return pd.DataFrame(
        {
            "component": component,
            "asset": assets,
            "variable": variable_name,
            "reference_value": targets.loc[assets].to_numpy(),
            "lower_bound": lower_bounds.loc[assets].to_numpy(),
            "upper_bound": upper_bounds.loc[assets].to_numpy(),
            "delta": delta.loc[assets].to_numpy(),
        }
    )


def add_fixed_neighbor_capacity_constraints(
    n,
    investment_year,
    manifest_path,
    domestic_country,
    strict_asset_match=True,
    capacity_tolerance=0.01,
):
    """Fix non-domestic nominal capacities within a narrow reference band."""
    if capacity_tolerance < 0:
        raise ValueError("Fixed-neighbor capacity tolerance must be non-negative.")

    manifest = pd.read_csv(manifest_path, dtype={"asset": str})
    targets = manifest.loc[manifest["year"].eq(investment_year)].copy()
    if targets.empty:
        raise ValueError(
            f"No fixed-neighbor capacity targets found for {investment_year} in "
            f"{manifest_path}."
        )

    pending_constraints = []
    bound_reports = []
    validated_targets = 0
    for component, (
        list_name,
        nominal_attr,
        bus_columns,
    ) in FIXED_NEIGHBOR_COMPONENTS.items():
        component_targets = targets.loc[targets["component"].eq(component)].copy()
        if component_targets.empty:
            continue
        if component_targets["asset"].duplicated().any():
            raise ValueError(
                f"Duplicate {component} entries in fixed-neighbor manifest for "
                f"{investment_year}."
            )

        component_targets = component_targets.set_index("asset")
        static = getattr(n, list_name)
        target_index = component_targets.index
        missing = target_index.difference(static.index)
        if strict_asset_match and not missing.empty:
            raise ValueError(
                f"{component}: {len(missing)} reference assets are missing from the "
                f"current network, e.g. {missing[:3].tolist()}."
            )
        target_index = target_index.intersection(static.index)

        active = static.get(
            "active", pd.Series(True, index=static.index, dtype=bool)
        ).fillna(False)
        external_active = _external_asset_index(
            n, static, bus_columns, domestic_country
        ).intersection(static.index[active])
        unexpected = external_active.difference(target_index)
        if strict_asset_match and not unexpected.empty:
            raise ValueError(
                f"{component}: {len(unexpected)} active non-{domestic_country} assets "
                f"are absent from the reference manifest, e.g. "
                f"{unexpected[:3].tolist()}."
            )

        extendable_column = f"{nominal_attr}_nom_extendable"
        nominal_column = f"{nominal_attr}_nom"
        extendable = target_index[
            active.loc[target_index]
            & static.loc[target_index, extendable_column].fillna(False).astype(bool)
        ]
        fixed = target_index.difference(extendable)
        if strict_asset_match and not fixed.empty:
            expected = component_targets.loc[fixed, "nominal"]
            actual = static.loc[fixed, nominal_column]
            mismatch = ~np.isclose(actual, expected, rtol=1e-9, atol=1e-6)
            if mismatch.any():
                mismatched = fixed[np.asarray(mismatch)]
                raise ValueError(
                    f"{component}: {len(mismatched)} fixed non-{domestic_country} "
                    f"capacities differ from the reference, e.g. "
                    f"{mismatched[:3].tolist()}."
                )

        if extendable.empty:
            logger.info(
                "Fixed-neighbor %s: no extendable non-%s assets in %s.",
                component,
                domestic_country,
                investment_year,
            )
            continue

        target_nominal = component_targets.loc[extendable, "nominal"].astype(float)
        bound_context = _model_nominal_bound_context(
            n, component, nominal_attr, extendable
        )
        bound_reports.append(
            _fixed_neighbor_bound_violations(
                target_nominal,
                bound_context["lower"],
                bound_context["upper"],
                component,
                bound_context["variable_name"],
            )
        )
        target_tolerance = pd.Series(capacity_tolerance, index=extendable)
        effective_lower = pd.concat(
            [target_nominal - target_tolerance, bound_context["lower"]], axis=1
        ).max(axis=1)
        effective_upper = pd.concat(
            [target_nominal + target_tolerance, bound_context["upper"]], axis=1
        ).min(axis=1)
        no_overlap = effective_lower.gt(effective_upper)
        if no_overlap.any():
            invalid = extendable[no_overlap.to_numpy()]
            details = pd.DataFrame(
                {
                    "component": component,
                    "asset": invalid,
                    "variable": bound_context["variable_name"],
                    "reference_value": target_nominal.loc[invalid].to_numpy(),
                    "lower_bound": bound_context["lower"].loc[invalid].to_numpy(),
                    "upper_bound": bound_context["upper"].loc[invalid].to_numpy(),
                    "allowed_tolerance": target_tolerance.loc[invalid].to_numpy(),
                }
            ).to_string(index=False)
            raise ValueError(
                "Fixed-neighbor reference bands do not intersect the Linopy "
                f"nominal bounds for {len(invalid)} targets:\n{details}"
            )

        pending_constraints.append(
            (
                component,
                nominal_attr,
                extendable,
                bound_context,
                effective_lower,
                effective_upper,
            )
        )
        validated_targets += len(extendable)

    bound_report = (
        pd.concat(bound_reports, ignore_index=True)
        if bound_reports
        else pd.DataFrame(columns=["delta"])
    )
    if not bound_report.empty:
        for row in bound_report.itertuples(index=False):
            logger.warning(
                "Fixed-neighbor pre-solve bound violation: component=%s, "
                "asset=%s, variable=%s, reference_value=%.12g, "
                "lower_bound=%.12g, upper_bound=%.12g, delta=%.12g.",
                row.component,
                row.asset,
                row.variable,
                row.reference_value,
                row.lower_bound,
                row.upper_bound,
                row.delta,
            )

    for (
        component,
        nominal_attr,
        extendable,
        bound_context,
        effective_lower,
        effective_upper,
    ) in pending_constraints:
        nominal = n.model[bound_context["variable_name"]].loc[extendable]
        dimension = nominal.dims[0]
        lower_rhs = DataArray(
            effective_lower.to_numpy(),
            coords={dimension: extendable},
            dims=(dimension,),
        )
        upper_rhs = DataArray(
            effective_upper.to_numpy(),
            coords={dimension: extendable},
            dims=(dimension,),
        )
        n.model.add_constraints(
            nominal >= lower_rhs,
            name=f"FixedNeighbor-{component}-{nominal_attr}_nom-lower",
        )
        n.model.add_constraints(
            nominal <= upper_rhs,
            name=f"FixedNeighbor-{component}-{nominal_attr}_nom-upper",
        )
        logger.info(
            "Fixed-neighbor %s: constrained %s non-%s %s_nom capacities to "
            "narrow reference bands in %s.",
            component,
            len(extendable),
            domestic_country,
            nominal_attr,
            investment_year,
        )

    logger.info(
        "Fixed-neighbor pre-solve validation passed for %s targets in %s; "
        "%s reference targets lie just outside their original nominal bounds. "
        "Capacity bands use tolerance %.12g nominal units.",
        validated_targets,
        investment_year,
        len(bound_report),
        capacity_tolerance,
    )


def add_continuous_acaes_constraints(n, options):
    """Link continuous A-CAES power, output duration, and geology conventions."""
    if not options or not options.get("enable", False):
        return

    carrier = str(options["carrier"])
    charger_carrier = f"{carrier} charge"
    discharger_carrier = f"{carrier} discharge"
    site_output_twh = pd.Series(options["site_output_capacities_twh"], dtype=float)
    national_output_twh = float(options["geological_output_capacity_twh"])
    if not np.isclose(site_output_twh.sum(), national_output_twh):
        raise ValueError(
            "Continuous A-CAES regional geological limits do not sum to the "
            "national cap."
        )

    round_trip_efficiency = float(options["round_trip_efficiency"])
    if options["efficiency_split"] != "symmetric":
        raise ValueError("Continuous A-CAES requires a symmetric efficiency split.")
    efficiency_dispatch = np.sqrt(round_trip_efficiency)
    if not np.isclose(float(options["charge_to_discharge_power_ratio"]), 1.0):
        raise ValueError(
            "Continuous A-CAES requires a 1:1 grid-side charging-to-discharging "
            "power ratio."
        )

    minimum_duration = float(options["minimum_output_duration_hours"])
    maximum_duration = float(options["maximum_output_duration_hours"])
    if minimum_duration <= 0 or maximum_duration < minimum_duration:
        raise ValueError("Continuous A-CAES output-duration bounds are invalid.")

    acaes_stores = n.stores.index[n.stores.carrier.eq(carrier)]
    acaes_chargers = n.links.index[n.links.carrier.eq(charger_carrier)]
    acaes_dischargers = n.links.index[n.links.carrier.eq(discharger_carrier)]
    if len(acaes_stores) != len(site_output_twh):
        raise ValueError(
            "Continuous A-CAES is enabled but the expected regional Stores do "
            "not exist."
        )
    if len(acaes_chargers) != len(site_output_twh) or len(acaes_dischargers) != len(
        site_output_twh
    ):
        raise ValueError(
            "Continuous A-CAES is enabled but the expected charging/discharging "
            "Links do not exist."
        )

    link_p_nom = n.model["Link-p_nom"]
    store_e_nom = n.model["Store-e_nom"]
    for site, output_cap_twh in site_output_twh.items():
        store = f"{site} {carrier}"
        charger = f"{store} charger"
        discharger = f"{store} discharger"
        missing = [
            name
            for name, index in (
                (store, acaes_stores),
                (charger, acaes_chargers),
                (discharger, acaes_dischargers),
            )
            if name not in index
        ]
        if missing:
            raise ValueError(
                f"Continuous A-CAES components are missing at {site}: {missing}."
            )

        # Charger p_nom is grid input. Discharger p_nom is internal input, so
        # eta_d * discharger p_nom is grid output. The equality imposes the
        # approved 1:1 ratio between both grid-side power capacities.
        charge_power = link_p_nom.loc[[charger]].sum()
        discharge_power_internal = link_p_nom.loc[[discharger]].sum()
        energy_internal = store_e_nom.loc[[store]].sum()
        output_energy = efficiency_dispatch * energy_internal

        n.model.add_constraints(
            charge_power == efficiency_dispatch * discharge_power_internal,
            name=f"Link-aCAES-grid-power-ratio-{site}",
        )
        n.model.add_constraints(
            output_energy >= minimum_duration * charge_power,
            name=f"Store-aCAES-minimum-output-duration-{site}",
        )
        n.model.add_constraints(
            output_energy <= maximum_duration * charge_power,
            name=f"Store-aCAES-maximum-output-duration-{site}",
        )
        n.model.add_constraints(
            output_energy <= float(output_cap_twh) * 1e6,
            name=f"Store-aCAES-geological-output-{site}",
        )

    logger.info(
        "Constrained continuous RESC A-CAES to %.3f TWh electrical output "
        "across %d German sites with %.1f-%.1f h endogenous duration.",
        national_output_twh,
        len(site_output_twh),
        minimum_duration,
        maximum_duration,
    )


def add_capacity_limits(n, investment_year, limits_capacity, sense="maximum"):
    for c in n.iterate_components(limits_capacity):
        logger.info(f"Adding {sense} constraints for {c.list_name}")

        attr = "e" if c.name == "Store" else "p"
        units = "MWh or tCO2" if c.name == "Store" else "MW"

        for carrier in limits_capacity[c.name]:
            for ct in limits_capacity[c.name][carrier]:
                if investment_year not in limits_capacity[c.name][carrier][ct].keys():
                    continue

                limit = 1e3 * limits_capacity[c.name][carrier][ct][investment_year]

                logger.info(
                    f"Adding constraint on {c.name} {carrier} capacity in {ct} to be {sense} {limit} {units}"
                )

                valid_components = (
                    (c.static.index.str[:2] == ct)
                    & (c.static.carrier.str[: len(carrier)] == carrier)
                    & ~c.static.carrier.str.contains("thermal")
                )  # exclude solar thermal

                existing_index = c.static.index[
                    valid_components & ~c.static[attr + "_nom_extendable"]
                ]
                extendable_index = c.static.index[
                    valid_components
                    & c.static[attr + "_nom_extendable"]
                    & c.static.active
                ]

                if extendable_index.empty:
                    logger.info(
                        f"No extendable {c.name} with carrier {carrier} found in {ct}. Skipping constraint."
                    )
                    continue

                existing_capacity = c.static.loc[existing_index, attr + "_nom"].sum()

                logger.info(
                    f"Existing {c.name} {carrier} capacity in {ct}: {existing_capacity} {units}"
                )

                nom = n.model[c.name + "-" + attr + "_nom"].loc[extendable_index]

                lhs = nom.sum()

                cname = f"capacity_{sense}-{ct}-{c.name}-{carrier.replace(' ', '-')}"

                if cname in n.global_constraints.index:
                    logger.warning(
                        f"Global constraint {cname} already exists. Dropping and adding it again."
                    )
                    n.global_constraints.drop(cname, inplace=True)

                rhs = limit - existing_capacity

                if sense == "maximum":
                    if rhs <= 0:
                        logger.warning(
                            f"Existing capacity in {ct} for carrier {carrier} already exceeds the limit of {limit} MW. Limiting capacity expansion for this investment period to 0."
                        )
                        rhs = 0

                    n.model.add_constraints(
                        lhs <= rhs,
                        name=f"GlobalConstraint-{cname}",
                    )
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=rhs,
                        sense="<=",
                        type="",
                        carrier_attribute="",
                    )

                elif sense == "minimum":
                    n.model.add_constraints(
                        lhs >= rhs,
                        name=f"GlobalConstraint-{cname}",
                    )
                    n.add(
                        "GlobalConstraint",
                        cname,
                        constant=rhs,
                        sense=">=",
                        type="",
                        carrier_attribute="",
                    )
                else:
                    logger.error("sense {sense} not recognised")
                    sys.exit()


def add_power_limits(n, investment_year, limits_power_max):
    """
    " Restricts the maximum inflow/outflow of electricity from/to a country.
    """

    def add_pos_neg_aux_variables(n, idx, var_name, infix):
        """
        For every snapshot in the network `n` this functions adds auxiliary variables corresponding to the positive and negative parts of the dynamical variables of the network components specified in the index `idx`. The `infix` parameter is used to create unique names for the auxiliary variables and constraints.

        Parameters
        ----------
        n : pypsa.Network
            The PyPSA network object containing the model.
        idx : pandas.Index
            The index of the network component (e.g., lines or links) for which to create auxiliary variables.
        infix : str
            A string used to create unique names for the auxiliary variables and constraints.
        """
        var = n.model[var_name].sel({"name": idx})
        aux_pos = n.model.add_variables(
            name=f"{var_name}-{infix}-aux-pos",
            lower=0,
            coords=[n.snapshots, idx],
        )
        aux_neg = n.model.add_variables(
            name=f"{var_name}-{infix}-aux-neg",
            upper=0,
            coords=[n.snapshots, idx],
        )
        n.model.add_constraints(
            aux_pos >= var,
            name=f"{var_name}-{infix}-aux-pos-constr",
        )
        n.model.add_constraints(
            aux_neg <= var,
            name=f"{var_name}-{infix}-aux-neg-constr",
        )
        return aux_pos, aux_neg

    for ct in limits_power_max:
        if investment_year not in limits_power_max[ct].keys():
            continue

        lim = 1e3 * limits_power_max[ct][investment_year]  # in MW

        logger.info(
            f"Adding constraint on electricity import/export from/to {ct} to be < {lim} MW"
        )
        # identify interconnectors

        incoming_lines = n.lines.query(
            f"not bus0.str.startswith('{ct}') and bus1.str.startswith('{ct}') and active"
        )
        outgoing_lines = n.lines.query(
            f"bus0.str.startswith('{ct}') and not bus1.str.startswith('{ct}') and active"
        )
        incoming_links = n.links.query(
            f"not bus0.str.startswith('{ct}') and bus1.str.startswith('{ct}') and carrier == 'DC' and active"
        )
        outgoing_links = n.links.query(
            f"bus0.str.startswith('{ct}') and not bus1.str.startswith('{ct}') and carrier == 'DC' and active"
        )

        # define auxiliary variables for positive and negative parts of line and link flows

        incoming_lines_aux_pos, incoming_lines_aux_neg = add_pos_neg_aux_variables(
            n, incoming_lines.index, "Line-s", f"incoming-{ct}"
        )

        outgoing_lines_aux_pos, outgoing_lines_aux_neg = add_pos_neg_aux_variables(
            n, outgoing_lines.index, "Line-s", f"outgoing-{ct}"
        )

        incoming_links_aux_pos, incoming_links_aux_neg = add_pos_neg_aux_variables(
            n, incoming_links.index, "Link-p", f"incoming-{ct}"
        )

        outgoing_links_aux_pos, outgoing_links_aux_neg = add_pos_neg_aux_variables(
            n, outgoing_links.index, "Link-p", f"outgoing-{ct}"
        )
        # To constraint the absolute values of imports and exports, we have to sum the
        # corresponding positive and negative flows separately, using the auxiliary variables

        import_lhs = (
            incoming_links_aux_pos
            + incoming_lines_aux_pos
            - outgoing_links_aux_neg
            - outgoing_lines_aux_neg
        ).sum(dim="name") / 10

        export_lhs = (
            outgoing_links_aux_pos
            + outgoing_lines_aux_pos
            - incoming_links_aux_neg
            - incoming_lines_aux_neg
        ).sum(dim="name") / 10

        n.model.add_constraints(import_lhs <= lim / 10, name=f"Power-import-limit-{ct}")
        n.model.add_constraints(export_lhs <= lim / 10, name=f"Power-export-limit-{ct}")


def h2_import_limits(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["h2_import"]:
        limit = limits_volume_max["h2_import"][ct][investment_year] * 1e6

        logger.info(f"limiting H2 imports in {ct} to {limit / 1e6} TWh/a")
        pipeline_carrier = [
            "H2 pipeline",
            "H2 pipeline (Kernnetz)",
            "H2 pipeline retrofitted",
        ]
        incoming = n.links.index[
            (n.links.carrier.isin(pipeline_carrier))
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing = n.links.index[
            (n.links.carrier.isin(pipeline_carrier))
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        incoming_p = (
            n.model["Link-p"].loc[:, incoming] * n.snapshot_weightings.generators
        ).sum()
        outgoing_p = (
            n.model["Link-p"].loc[:, outgoing] * n.snapshot_weightings.generators
        ).sum()

        lhs = incoming_p - outgoing_p

        cname = f"H2_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )

        logger.info("Adding H2 export ban")

        cname = f"H2_export_ban-{ct}"

        n.model.add_constraints(lhs >= 0, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=0,
            sense=">=",
            type="",
            carrier_attribute="",
        )


def h2_production_limits(n, investment_year, limits_volume_min, limits_volume_max):
    for ct in limits_volume_max["electrolysis"]:
        if ct not in limits_volume_min["electrolysis"]:
            logger.warning(
                f"no lower limit for H2 electrolysis in {ct} assuming 0 TWh/a"
            )
            limit_lower = 0
        else:
            limit_lower = limits_volume_min["electrolysis"][ct][investment_year] * 1e6

        limit_upper = limits_volume_max["electrolysis"][ct][investment_year] * 1e6

        logger.info(
            f"limiting H2 electrolysis in DE between {limit_lower / 1e6} and {limit_upper / 1e6} TWh/a"
        )

        production = n.links[
            (n.links.carrier == "H2 Electrolysis") & (n.links.bus0.str.contains(ct))
        ].index
        efficiency = n.links.loc[production, "efficiency"]

        lhs = (
            n.model["Link-p"].loc[:, production]
            * n.snapshot_weightings.generators
            * efficiency
        ).sum()

        cname_upper = f"H2_production_limit_upper-{ct}"
        cname_lower = f"H2_production_limit_lower-{ct}"

        n.model.add_constraints(
            lhs <= limit_upper, name=f"GlobalConstraint-{cname_upper}"
        )

        n.model.add_constraints(
            lhs >= limit_lower, name=f"GlobalConstraint-{cname_lower}"
        )

        if cname_upper not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname_upper,
                constant=limit_upper,
                sense="<=",
                type="",
                carrier_attribute="",
            )
        if cname_lower not in n.global_constraints.index:
            n.add(
                "GlobalConstraint",
                cname_lower,
                constant=limit_lower,
                sense=">=",
                type="",
                carrier_attribute="",
            )


def electricity_import_limits(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["electricity_import"]:
        limit = limits_volume_max["electricity_import"][ct][investment_year] * 1e6

        if limit < 0:
            limit *= n.snapshot_weightings.generators.sum() / 8760

        logger.info(f"limiting electricity imports in {ct} to {limit / 1e6} TWh/a")

        incoming_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] != ct)
            & (n.lines.bus1.str[:2] == ct)
        ]
        outgoing_line = n.lines.index[
            (n.lines.carrier == "AC")
            & (n.lines.bus0.str[:2] == ct)
            & (n.lines.bus1.str[:2] != ct)
        ]

        incoming_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] != ct)
            & (n.links.bus1.str[:2] == ct)
        ]
        outgoing_link = n.links.index[
            (n.links.carrier == "DC")
            & (n.links.bus0.str[:2] == ct)
            & (n.links.bus1.str[:2] != ct)
        ]

        incoming_line_p = (
            n.model["Line-s"].loc[:, incoming_line] * n.snapshot_weightings.generators
        ).sum()
        outgoing_line_p = (
            n.model["Line-s"].loc[:, outgoing_line] * n.snapshot_weightings.generators
        ).sum()

        incoming_link_p = (
            n.model["Link-p"].loc[:, incoming_link] * n.snapshot_weightings.generators
        ).sum()
        outgoing_link_p = (
            n.model["Link-p"].loc[:, outgoing_link] * n.snapshot_weightings.generators
        ).sum()

        lhs = (incoming_link_p - outgoing_link_p) + (incoming_line_p - outgoing_line_p)

        cname = f"Electricity_import_limit-{ct}"

        n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def add_national_co2_budgets(n, snakemake, national_co2_budgets, investment_year):
    """
    Add a set of emissions limit constraints for specified countries.

    The countries and emissions limits are specified in the config file entry 'co2_budget_national'.

    Parameters
    ----------
    n : pypsa.Network
    snakemake : snakemake.io.Snakemake
    national_co2_budgets : dict
    investment_year : int

    """
    logger.info("Adding national CO2 budgets")
    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    sectors = determine_emission_sectors(n.config["sector"])
    energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=[0, 1])

    # convert MtCO2 to tCO2
    co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

    co2_total_totals = co2_totals[sectors].sum(axis=1) * nyears

    for ct in national_co2_budgets:
        if ct != "DE":
            logger.error(
                f"CO2 budget for countries other than `DE` is not yet supported. Found country {ct}. Please check the config file."
            )

        limit = co2_total_totals[ct] * national_co2_budgets[ct][investment_year]
        logger.info(
            f"Limiting emissions in country {ct} to {national_co2_budgets[ct][investment_year]:.1%} of "
            f"1990 levels, i.e. {limit:,.2f} tCO2/a",
        )

        lhs = []

        for port in [col[3:] for col in n.links if col.startswith("bus")]:
            links = n.links.index[
                (n.links.index.str[:2] == ct)
                & (n.links[f"bus{port}"] == "co2 atmosphere")
                & ~n.links.carrier.str.contains(
                    "shipping|aviation"
                )  # first exclude aviation to multiply it with a domestic factor later
            ]

            logger.info(
                f"For {ct} adding following link carriers to port {port} CO2 constraint: {n.links.loc[links, 'carrier'].unique()}"
            )

            if port == "0":
                efficiency = -1.0
            elif port == "1":
                efficiency = n.links.loc[links, "efficiency"]
            else:
                efficiency = n.links.loc[links, f"efficiency{port}"]

            lhs.append(
                (
                    n.model["Link-p"].loc[:, links]
                    * efficiency
                    * n.snapshot_weightings.generators
                ).sum()
            )

        # Aviation demand
        domestic_aviation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total domestic aviation"
        ]
        international_aviation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total international aviation"
        ]
        domestic_aviation_factor = domestic_aviation / (
            domestic_aviation + international_aviation
        )
        aviation_links = n.links[
            (n.links.index.str[:2] == ct) & (n.links.carrier == "kerosene for aviation")
        ]
        lhs.append(
            (
                n.model["Link-p"].loc[:, aviation_links.index]
                * aviation_links.efficiency2
                * n.snapshot_weightings.generators
            ).sum()
            * domestic_aviation_factor
        )
        logger.info(
            f"Adding domestic aviation emissions for {ct} with a factor of {domestic_aviation_factor}"
        )

        # Shipping oil
        domestic_navigation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total domestic navigation"
        ]
        international_navigation = energy_totals.loc[
            (ct, snakemake.params.energy_year), "total international navigation"
        ]
        domestic_navigation_factor = domestic_navigation / (
            domestic_navigation + international_navigation
        )
        shipping_links = n.links[
            (n.links.index.str[:2] == ct) & (n.links.carrier == "shipping oil")
        ]
        lhs.append(
            (
                n.model["Link-p"].loc[:, shipping_links.index]
                * shipping_links.efficiency2
                * n.snapshot_weightings.generators
            ).sum()
            * domestic_navigation_factor
        )

        # Shipping methanol
        shipping_meoh_links = n.links[
            (n.links.index.str[:2] == ct) & (n.links.carrier == "shipping methanol")
        ]
        if not shipping_meoh_links.empty:  # no shipping methanol in 2025
            lhs.append(
                (
                    n.model["Link-p"].loc[:, shipping_meoh_links.index]
                    * shipping_meoh_links.efficiency2
                    * n.snapshot_weightings.generators
                ).sum()
                * domestic_navigation_factor
            )

        logger.info(
            f"Adding domestic shipping emissions for {ct} with a factor of {domestic_navigation_factor}"
        )

        # Adding Efuel imports and exports to constraint
        incoming_oil = n.links.index[n.links.index == f"EU renewable oil -> {ct} oil"]
        outgoing_oil = n.links.index[n.links.index == f"{ct} renewable oil -> EU oil"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_oil]
                * 0.2571
                * n.snapshot_weightings.generators
            ).sum()
        )
        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_oil]
                * 0.2571
                * n.snapshot_weightings.generators
            ).sum()
        )

        incoming_methanol = n.links.index[
            n.links.index == f"EU methanol -> {ct} methanol"
        ]
        outgoing_methanol = n.links.index[
            n.links.index == f"{ct} methanol -> EU methanol"
        ]

        methanol_emissions = n.links.loc["EU industry methanol", "efficiency2"]
        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_methanol]
                * methanol_emissions
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_methanol]
                * methanol_emissions
                * n.snapshot_weightings.generators
            ).sum()
        )

        # Methane
        incoming_CH4 = n.links.index[n.links.index == f"EU renewable gas -> {ct} gas"]
        outgoing_CH4 = n.links.index[n.links.index == f"{ct} renewable gas -> EU gas"]

        lhs.append(
            (
                -1
                * n.model["Link-p"].loc[:, incoming_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs.append(
            (
                n.model["Link-p"].loc[:, outgoing_CH4]
                * 0.198
                * n.snapshot_weightings.generators
            ).sum()
        )

        lhs = sum(lhs)

        cname = f"co2_limit-{ct}"

        n.model.add_constraints(
            lhs <= limit,
            name=f"GlobalConstraint-{cname}",
        )

        if cname in n.global_constraints.index:
            logger.warning(
                f"Global constraint {cname} already exists. Dropping and adding it again."
            )
            n.global_constraints.drop(cname, inplace=True)

        n.add(
            "GlobalConstraint",
            cname,
            constant=limit,
            sense="<=",
            type="",
            carrier_attribute="",
        )


def force_boiler_profiles_existing_per_load(n):
    """
    This scales the boiler dispatch to the load profile with a factor common to
    all boilers at load.
    """

    logger.info("Forcing boiler profiles for existing ones")

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    boiler_load_index = pd.Index(boiler_loads.unique())
    boiler_load_index.name = "Load"

    # per load scaling factor
    n.model.add_variables(coords=[boiler_load_index], name="Load-profile_factor")

    # clumsy indicator matrix to map boilers to loads
    df = pd.DataFrame(index=boiler_load_index, columns=decentral_boilers, data=0.0)
    for k, v in boiler_loads.items():
        df.loc[v, k] = 1.0

    lhs = n.model["Link-p"].loc[:, decentral_boilers] - (
        boiler_profiles * DataArray(df) * n.model["Load-profile_factor"]
    ).sum("Load")

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.loads["profile_factor_opt"] = 0.0


def force_boiler_profiles_existing_per_boiler(n):
    """
    This scales each boiler dispatch to be proportional to the load profile.
    """

    logger.info(
        "Forcing each existing boiler dispatch to be proportional to the load profile"
    )

    decentral_boilers = n.links.index[
        n.links.carrier.str.contains("boiler")
        & ~n.links.carrier.str.contains("urban central")
        & ~n.links.p_nom_extendable
    ]

    if decentral_boilers.empty:
        return

    boiler_loads = n.links.loc[decentral_boilers, "bus1"]
    boiler_loads = boiler_loads[boiler_loads.isin(n.loads_t.p_set.columns)]
    decentral_boilers = boiler_loads.index
    boiler_profiles_pu = n.loads_t.p_set[boiler_loads].div(
        n.loads_t.p_set[boiler_loads].max(), axis=1
    )
    boiler_profiles_pu.columns = decentral_boilers
    boiler_profiles = DataArray(
        boiler_profiles_pu.multiply(n.links.loc[decentral_boilers, "p_nom"], axis=1)
    )

    # will be per unit
    n.model.add_variables(coords=[decentral_boilers], name="Link-fixed_profile_scaling")

    lhs = (
        (1, n.model["Link-p"].loc[:, decentral_boilers]),
        (
            -boiler_profiles,
            n.model["Link-fixed_profile_scaling"],
        ),
    )

    n.model.add_constraints(lhs, "=", 0, "Link-fixed_profile_scaling")

    # hack so that PyPSA doesn't complain there is nowhere to store the variable
    n.links["fixed_profile_scaling_opt"] = 0.0


def add_h2_derivate_limit(n, investment_year, limits_volume_max):
    for ct in limits_volume_max["h2_derivate_import"]:
        limit = limits_volume_max["h2_derivate_import"][ct][investment_year] * 1e6

        logger.info(f"limiting H2 derivate imports in {ct} to {limit / 1e6} TWh/a")

        incoming = n.links.loc[
            [
                "EU renewable oil -> DE oil",
                "EU methanol -> DE methanol",
                "EU renewable gas -> DE gas",
            ]
        ].index
        outgoing = n.links.loc[
            [
                "DE renewable oil -> EU oil",
                "DE methanol -> EU methanol",
                "DE renewable gas -> EU gas",
            ]
        ].index

        carrier_idx_dict = {
            # Every carrier should respect the limit individually
            "renewable_oil": 0,
            "methanol": 1,
            "renewable_gas": 2,
            # Exports of one carrier should not compensate for imports of another carrier
            "H2_derivate_oil_meoh": [0, 1],
            "H2_derivate_oil_gas": [0, 2],
            "H2_derivate_meoh_gas": [1, 2],
            # The sum of all carriers should respect the limit
            "H2_derivate_oil_meoh_gas": [0, 1, 2],
        }
        for carrier, idx in carrier_idx_dict.items():
            cname = f"{carrier}_import_limit-{ct}"

            incoming_p = (
                n.model["Link-p"].loc[:, incoming[idx]]
                * n.snapshot_weightings.generators
            ).sum()
            outgoing_p = (
                n.model["Link-p"].loc[:, outgoing[idx]]
                * n.snapshot_weightings.generators
            ).sum()

            lhs = incoming_p - outgoing_p

            n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

            if cname in n.global_constraints.index:
                logger.warning(
                    f"Global constraint {cname} already exists. Dropping and adding it again."
                )
                n.global_constraints.drop(cname, inplace=True)

            n.add(
                "GlobalConstraint",
                cname,
                constant=limit,
                sense="<=",
                type="",
                carrier_attribute="",
            )

    # Export bans on efuels are implemented in modify_prenetwork by restricting p_max_pu of the DE -> EU links


def adapt_nuclear_output(n):
    logger.info(
        "limiting german electricity generation from nuclear to 2020 value of 61 TWh"
    )
    limit = 61e6

    nuclear_de_index = n.links.index[
        (n.links.carrier == "nuclear") & (n.links.index.str[:2] == "DE")
    ]

    nuclear_gen = (
        n.model["Link-p"].loc[:, nuclear_de_index]
        * n.links.loc[nuclear_de_index, "efficiency"]
        * n.snapshot_weightings.generators
    ).sum()

    lhs = nuclear_gen

    cname = "Nuclear_generation_limit-DE"

    n.model.add_constraints(lhs <= limit, name=f"GlobalConstraint-{cname}")

    if cname in n.global_constraints.index:
        logger.warning(
            f"Global constraint {cname} already exists. Dropping and adding it again."
        )
        n.global_constraints.drop(cname, inplace=True)

    n.add(
        "GlobalConstraint",
        cname,
        constant=limit,
        sense="<=",
        type="",
        carrier_attribute="",
    )


def additional_functionality(n, snapshots, snakemake):
    logger.info("Adding Ariadne-specific functionality")

    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    constraints = snakemake.params.solving["constraints"]

    add_capacity_limits(
        n, investment_year, constraints["limits_capacity_min"], "minimum"
    )

    add_capacity_limits(
        n, investment_year, constraints["limits_capacity_max"], "maximum"
    )

    fixed_neighbor = constraints.get("fixed_neighbor_capacities", {})
    if fixed_neighbor.get("enable", False):
        add_fixed_neighbor_capacity_constraints(
            n,
            investment_year,
            Path(snakemake.input.fixed_neighbor_capacities),
            fixed_neighbor["domestic_country"],
            strict_asset_match=fixed_neighbor.get("strict_asset_match", True),
            capacity_tolerance=fixed_neighbor.get("capacity_tolerance", 0.01),
        )

    acaes_options = (
        snakemake.config.get("electricity", {})
        .get("storage_options", {})
        .get("aCAES_RESC_continuous", {})
    )
    if acaes_options.get("enable", False) and investment_year >= int(
        acaes_options["first_investment_year"]
    ):
        add_continuous_acaes_constraints(n, acaes_options)

    add_power_limits(n, investment_year, constraints["limits_power_max"])

    if snakemake.wildcards.clusters != "1":
        h2_import_limits(n, investment_year, constraints["limits_volume_max"])

        electricity_import_limits(n, investment_year, constraints["limits_volume_max"])

    if investment_year >= 2025:
        h2_production_limits(
            n,
            investment_year,
            constraints["limits_volume_min"],
            constraints["limits_volume_max"],
        )

    add_h2_derivate_limit(n, investment_year, constraints["limits_volume_max"])

    # force_boiler_profiles_existing_per_load(n)
    force_boiler_profiles_existing_per_boiler(n)

    if isinstance(constraints["co2_budget_national"], dict):
        add_national_co2_budgets(
            n,
            snakemake,
            constraints["co2_budget_national"],
            investment_year,
        )
    else:
        logger.warning("No national CO2 budget specified!")

    if investment_year == 2020:
        adapt_nuclear_output(n)
