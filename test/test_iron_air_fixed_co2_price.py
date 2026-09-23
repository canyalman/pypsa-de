# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Keep the iron-air comparison on the same 2035 carbon policy as BESS-only."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest
import yaml

from test.test_fixed_neighbor_constraints import MODULE

PRICE = 308.2489253762274


def mock_snakemake(year=2035):
    return SimpleNamespace(
        config={
            "costs": {
                "emission_prices": {
                    "enable": True,
                    "dynamic": False,
                    "co2": {2025: 0.0, 2030: 0.0, 2035: PRICE},
                }
            }
        },
        params=SimpleNamespace(
            solving={
                "constraints": {
                    "fixed_co2_price": {
                        "enable": True,
                        "planning_horizons": [2035],
                    },
                    "fixed_neighbor_capacities": {"enable": False},
                }
            }
        ),
        wildcards=SimpleNamespace(planning_horizons=str(year)),
    )


def co2_network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=2, freq="3h"))
    n.snapshot_weightings.loc[:, :] = 3.0
    n.add("Carrier", "AC")
    n.add("Carrier", "gas")
    n.add("Carrier", "co2", co2_emissions=-1.0)
    n.add("Bus", "DE electricity", carrier="AC")
    n.add("Bus", "EU gas", carrier="gas")
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Generator", "gas source", bus="EU gas", p_nom=100, marginal_cost=20)
    n.add(
        "Link",
        "gas plant",
        bus0="EU gas",
        bus1="DE electricity",
        bus2="co2 atmosphere",
        p_nom=100,
        efficiency=0.5,
        efficiency2=0.2,
    )
    n.add("Load", "electric demand", bus="DE electricity", p_set=1.0)
    n.add(
        "Store",
        "co2 atmosphere",
        bus="co2 atmosphere",
        carrier="co2",
        e_nom=1e6,
        e_min_pu=-1.0,
    )
    n.add("GlobalConstraint", "CO2Limit", type="co2_atmosphere", constant=1.0)
    n.add("GlobalConstraint", "co2_limit-DE", type="", constant=0.5)
    n.add("GlobalConstraint", "co2_sequestration_limit", type="", constant=-1e8)
    return n


def test_config_matches_bess_only_policy():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "config/config.de.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["run"]["name"] == [
        "KN2045_Mix_FixedRenewables",
        *[
            f"KN2045_Mix_FixedRenewables_BESS_{power}GW"
            for power in (15, 35, 55)
        ],
    ]
    assert config["costs"]["emission_prices"]["co2"] == {
        2025: 0.0,
        2030: 0.0,
        2035: PRICE,
    }
    assert config["solving"]["constraints"]["fixed_co2_price"] == {
        "enable": True,
        "planning_horizons": [2035],
    }


@pytest.mark.parametrize("year", [2025, 2030])
def test_reference_history_caps_are_untouched(year):
    n = co2_network()
    before = n.global_constraints.copy()
    MODULE.prepare_network_before_model(n, mock_snakemake(year))
    pd.testing.assert_frame_equal(n.global_constraints, before)
    assert n.stores.at["co2 atmosphere", "marginal_cost"] == 0.0


def test_2035_removes_only_emissions_caps_and_prices_once():
    n = co2_network()
    smk = mock_snakemake()
    MODULE.prepare_network_before_model(n, smk)
    assert n.global_constraints.index.tolist() == ["co2_sequestration_limit"]
    assert n.stores.at["co2 atmosphere", "marginal_cost"] == -PRICE
    status, condition = n.optimize(solver_name="highs")
    assert (status, condition) == ("ok", "optimal")
    MODULE.validate_fixed_co2_price_model(n, smk)
    assert n.objective == pytest.approx(240 + 2.4 * PRICE)
    n.meta["fixed_neighbor_static"] = {"removed_annualized_capex_eur": 123.0}
    MODULE.finalize_network_after_solve(n, smk)
    assert n.meta["fixed_co2_price"]["co2_payments_in_objective_eur"] == pytest.approx(
        2.4 * PRICE
    )
    assert n.meta["fixed_co2_price"]["objective_including_fixed_nonde_capex_excluding_co2_payments_eur"] == pytest.approx(363)


def test_reintroduced_emissions_cap_is_rejected():
    n = co2_network()
    smk = mock_snakemake()
    MODULE.prepare_network_before_model(n, smk)
    n.optimize.create_model()
    n.model.add_constraints(
        n.model["Store-e"].sum() <= 1, name="GlobalConstraint-CO2Limit"
    )
    with pytest.raises(ValueError, match="still has emissions caps"):
        MODULE.validate_fixed_co2_price_model(n, smk)
