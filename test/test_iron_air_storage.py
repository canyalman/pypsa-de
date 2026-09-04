# SPDX-FileCopyrightText: 2026 Can Yalman
#
# SPDX-License-Identifier: MIT

"""Focused tests for the fixed-duration iron-air storage formulation."""

import numpy as np
import pypsa

from scripts.solve_network import add_iron_air_constraints


def build_iron_air_model():
    n = pypsa.Network()
    n.set_snapshots([0])
    n.config = {
        "electricity": {
            "storage_options": {
                "iron-air": {
                    "duration_hours_at_rated_ac_output": 100.0,
                    "discharging_efficiency": 0.60,
                }
            }
        }
    }
    n.add("Bus", "DE AC", carrier="AC", country="DE")
    n.add("Bus", "DE AC iron-air", carrier="iron-air", country="DE")
    n.add(
        "Store",
        "DE AC iron-air",
        bus="DE AC iron-air",
        carrier="iron-air",
        e_nom_extendable=True,
        e_nom_min=100.0,
        e_cyclic=True,
    )
    n.add(
        "Link",
        "DE AC iron-air charger",
        bus0="DE AC",
        bus1="DE AC iron-air",
        carrier="iron-air charger",
        efficiency=0.71,
        p_nom_extendable=True,
        capital_cost=1.0,
    )
    n.add(
        "Link",
        "DE AC iron-air discharger",
        bus0="DE AC iron-air",
        bus1="DE AC",
        carrier="iron-air discharger",
        efficiency=0.60,
        p_nom_extendable=True,
    )
    n.optimize.create_model()
    return n


def test_iron_air_has_100_hours_at_rated_ac_output():
    n = build_iron_air_model()

    add_iron_air_constraints(n)
    status, condition = n.optimize.solve_model(solver_name="highs")

    assert (status, condition) == ("ok", "optimal")
    store = n.stores.at["DE AC iron-air", "e_nom_opt"]
    charger = n.links.at["DE AC iron-air charger", "p_nom_opt"]
    discharger_internal = n.links.at["DE AC iron-air discharger", "p_nom_opt"]
    rated_ac_output = 0.60 * discharger_internal

    np.testing.assert_allclose(store, 100.0 * discharger_internal)
    np.testing.assert_allclose(charger, rated_ac_output)
    np.testing.assert_allclose(store / discharger_internal, 100.0)


def test_iron_air_constraint_is_inactive_without_configuration():
    n = build_iron_air_model()
    n.config = {}

    add_iron_air_constraints(n)

    assert not any(name.startswith("Iron-air-") for name in n.model.constraints)
