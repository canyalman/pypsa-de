import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import yaml

from scripts.add_electricity import attach_modular_acaes

REPO_ROOT = Path(__file__).parents[1]
PROJECT_TABLE = REPO_ROOT / "data" / "pypsa-de" / "caes_resc_project_options.csv"


def _production_options():
    with (REPO_ROOT / "config" / "config.de.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config["electricity"]["storage_options"]["aCAES_RESC"]


def _geological_constraint_function():
    module_path = (
        REPO_ROOT / "scripts" / "pypsa-de" / "additional_functionality.py"
    )
    spec = importlib.util.spec_from_file_location("acaes_constraints", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.add_acaes_geological_output_constraints


def test_resc_project_table_contains_only_approved_options():
    projects = pd.read_csv(PROJECT_TABLE)

    assert len(projects) == 15
    assert set(projects.power_mw) == {125, 250, 500}
    assert set(projects.output_duration_hours) == {8, 12, 16, 24, 48}
    assert 10 not in set(projects.output_duration_hours)
    assert np.allclose(
        projects.rated_output_energy_mwh_per_module,
        projects.power_mw * projects.output_duration_hours,
    )


def test_production_options_add_60_integer_project_choices():
    options = _production_options()
    buses = pd.Index(options["site_output_capacities_twh"])

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2035-01-01", periods=1, freq="h"))
    n.add("Bus", buses, carrier="AC", country="DE")
    attach_modular_acaes(
        n=n,
        buses_i=buses,
        options=options,
        project_options_file=str(PROJECT_TABLE),
        investment_year=2035,
    )

    assert len(n.storage_units) == 4 * 3 * 5
    assert set(n.storage_units.acaes_module_power_mw) == {125, 250, 500}
    assert set(n.storage_units.acaes_output_duration_hours) == {8, 12, 16, 24, 48}
    assert (n.storage_units.p_nom_mod > 0).all()


def test_modular_acaes_builds_complete_500_mw_8h_projects(tmp_path):
    projects = pd.read_csv(PROJECT_TABLE).query(
        "power_mw == 500 and output_duration_hours == 8"
    )
    project_file = tmp_path / "acaes_500mw_8h.csv"
    projects.to_csv(project_file, index=False)

    options = copy.deepcopy(_production_options())
    options["module_power_mw"] = [500]
    options["output_duration_hours"] = [8]
    options["geological_output_capacity_twh"] = 0.004
    options["site_output_capacities_twh"] = {"DE test": 0.004}

    n = pypsa.Network()
    snapshots = pd.date_range("2035-01-01", periods=21, freq="h")
    n.set_snapshots(snapshots)
    n.add("Bus", "DE test", carrier="AC", country="DE")
    n.add(
        "Generator",
        "charging supply",
        bus="DE test",
        p_nom=500,
        p_max_pu=[1.0] * 13 + [0.0] * 8,
    )
    n.add("Load", "output demand", bus="DE test", p_set=[0.0] * 13 + [500.0] * 8)

    attach_modular_acaes(
        n=n,
        buses_i=pd.Index(["DE test"]),
        options=options,
        project_options_file=str(project_file),
        investment_year=2035,
    )

    name = "DE test aCAES RESC 500MW 8h"
    eta = np.sqrt(0.63)
    assert n.storage_units.at[name, "p_nom_mod"] == 500
    assert np.isclose(n.storage_units.at[name, "max_hours"], 8 / eta)

    status, condition = n.optimize(
        solver_name="highs", include_objective_constant=False
    )
    assert (status, condition) == ("ok", "optimal")
    assert np.isclose(n.storage_units.at[name, "p_nom_opt"], 500)

    dispatch = n.storage_units_t.p_dispatch[name]
    charge = n.storage_units_t.p_store[name]
    assert np.allclose(dispatch.iloc[-8:], 500)
    assert np.isclose(dispatch.sum(), 4000)
    assert np.isclose(dispatch.sum() / charge.sum(), 0.63)

    expected_objective = (
        n.storage_units.at[name, "capital_cost"] * 500
        + n.storage_units.at[name, "marginal_cost"] * dispatch.sum()
    )
    assert np.isclose(n.objective, expected_objective)


def test_geological_cap_combines_all_project_durations(tmp_path):
    projects = pd.read_csv(PROJECT_TABLE).query(
        "power_mw == 500 and output_duration_hours in [8, 12]"
    )
    project_file = tmp_path / "acaes_500mw_8h_12h.csv"
    projects.to_csv(project_file, index=False)

    options = copy.deepcopy(_production_options())
    options["module_power_mw"] = [500]
    options["output_duration_hours"] = [8, 12]
    options["geological_output_capacity_twh"] = 0.006
    options["site_output_capacities_twh"] = {"DE test": 0.006}

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2035-01-01", periods=2, freq="h"))
    n.add("Bus", "DE test", carrier="AC", country="DE")
    attach_modular_acaes(
        n=n,
        buses_i=pd.Index(["DE test"]),
        options=options,
        project_options_file=str(project_file),
        investment_year=2035,
    )

    n.optimize.create_model(include_objective_constant=False)
    _geological_constraint_function()(n, options)

    constraint_name = "StorageUnit-aCAES-geological-output-DE test"
    assert constraint_name in n.model.constraints
    assert "StorageUnit-n_mod" in n.model.variables
