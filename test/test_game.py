import numpy as np
import pandas as pd
import pytest

from src.game import attacker_success, owner_cost, solve_stackelberg


def test_success_transform_and_bounds():
    assert attacker_success(0.5) == 0
    assert attacker_success(0.75) == 0.5
    with pytest.raises(ValueError):
        attacker_success(float("nan"))
    with pytest.raises(ValueError):
        attacker_success(1.1)


def test_attacker_and_owner_payoff_calculations():
    rows = pd.DataFrame([
        {"lambda": 0.0, "budget": 1, "mia_auc": .7, "mia_advantage": .4, "utility_q": .9},
        {"lambda": 0.0, "budget": 2, "mia_auc": .7, "mia_advantage": .4, "utility_q": .9},
        {"lambda": 1.0, "budget": 1, "mia_auc": .6, "mia_advantage": .2, "utility_q": .8},
        {"lambda": 1.0, "budget": 2, "mia_auc": .6, "mia_advantage": .2, "utility_q": .8},
    ])
    out = solve_stackelberg(rows, [2.0], owner_cost_scale=.1, attacker_cost_per_batch=.05)
    lam0 = out[out["lambda"] == 0].iloc[0]
    lam1 = out[out["lambda"] == 1].iloc[0]
    assert lam0.best_budget == 1  # attacker payoff .4 - .05*b; tie resolves to lower b
    assert lam0.attacker_payoff == pytest.approx(.35)
    assert lam0.owner_payoff == pytest.approx(.9 - 2 * .4)
    assert lam1.best_budget == 1
    assert lam1.owner_cost == pytest.approx(.05)
    assert lam1.owner_payoff == pytest.approx(.8 - 2 * .2 - .05)
    assert out.loc[out.owner_selected, "lambda"].iloc[0] == 1


def test_best_response_and_owner_selection_known_example():
    rows = pd.DataFrame([
        {"lambda": 0., "budget": 1, "mia_auc": .8, "utility_q": .9},
        {"lambda": 0., "budget": 5, "mia_auc": .9, "utility_q": .9},
        {"lambda": 1., "budget": 1, "mia_auc": .55, "utility_q": .8},
        {"lambda": 1., "budget": 5, "mia_auc": .56, "utility_q": .8},
    ])
    out = solve_stackelberg(rows, [2.], owner_cost_scale=0., attacker_cost_per_batch=.02)
    assert out.set_index("lambda").best_budget.to_dict() == {0.0: 5, 1.0: 1}
    assert out.loc[out.owner_selected, "lambda"].iloc[0] == 1.
    assert set(out.solution_scope) == {"finite-grid Stackelberg solution"}


def test_tied_responses_select_lower_budget_and_lambda():
    rows = pd.DataFrame([
        {"lambda": lam, "budget": budget, "mia_auc": .5, "mia_advantage": 0., "utility_q": .5}
        for lam in [0., 1.] for budget in [1, 5]
    ])
    out = solve_stackelberg(rows, [1.], attacker_cost_per_batch=0., owner_cost_scale=0.)
    assert set(out.best_budget) == {1}
    assert out.loc[out.owner_selected, "lambda"].iloc[0] == 0
    assert set(out.attacker_tied_best_responses) == {2}


@pytest.mark.parametrize("kwargs", [
    {"alphas": []}, {"alphas": [-1]}, {"alphas": [float("nan")]},
    {"attacker_cost_per_batch": -0.1}, {"owner_cost_scale": float("inf")},
])
def test_invalid_grid_or_payoff_parameters(kwargs):
    rows = pd.DataFrame([{"lambda": 0., "budget": 1, "mia_auc": .5, "utility_q": .5}])
    with pytest.raises(ValueError):
        solve_stackelberg(rows, **kwargs)


def test_empty_grid_invalid_budget_and_incomplete_grid_are_rejected():
    rows = pd.DataFrame([{"lambda": 0., "budget": 1, "mia_auc": .5, "utility_q": .5}])
    with pytest.raises(ValueError):
        solve_stackelberg(rows, lambdas=[])
    with pytest.raises(ValueError):
        solve_stackelberg(pd.DataFrame([{"lambda": 0., "budget": 0, "mia_auc": .5, "utility_q": .5}]))
    incomplete = pd.concat([rows, pd.DataFrame([{"lambda": 1., "budget": 1, "mia_auc": .5, "utility_q": .5},
                                                {"lambda": 1., "budget": 2, "mia_auc": .5, "utility_q": .5}])])
    with pytest.raises(ValueError, match="Incomplete strategy grid"):
        solve_stackelberg(incomplete)


def test_owner_cost_function_validates_and_matches_model():
    assert owner_cost(1.0, .02) == pytest.approx(.01)
    with pytest.raises(ValueError):
        owner_cost(-1, .02)
    with pytest.raises(ValueError):
        owner_cost(1, -0.1)
