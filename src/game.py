"""Finite-grid Stackelberg game for privacy-utility strategy selection."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def attacker_success(auc: float) -> float:
    """Map ROC-AUC to a [0, 1] leakage proxy with random guessing at zero."""
    value = float(auc)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("AUC must be a finite value in [0, 1].")
    return 2.0 * max(0.0, value - 0.5)


def _grid(values: Iterable[float], name: str, *, positive: bool = False) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty numeric strategy grid.")
    try:
        grid = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty numeric strategy grid.") from exc
    if not grid or any(not math.isfinite(value) for value in grid):
        raise ValueError(f"{name} must contain finite values and cannot be empty.")
    if any(value < 0 or (positive and value <= 0) for value in grid):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} values must be {qualifier}.")
    return sorted(set(grid))


def owner_cost(lam: float, cost_scale: float) -> float:
    """Configured owner cost C_O(lambda) = scale * lambda / (1 + lambda)."""
    lam, cost_scale = float(lam), float(cost_scale)
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be finite and non-negative.")
    if not math.isfinite(cost_scale) or cost_scale < 0:
        raise ValueError("owner_cost_scale must be finite and non-negative.")
    return cost_scale * lam / (1.0 + lam)


def solve_stackelberg(
    results: pd.DataFrame,
    alphas=(0.25, 0.5, 1.0, 2.0, 5.0),
    owner_cost_scale: float = 0.01,
    attacker_cost_per_batch: float = 0.01,
    lambdas: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Solve the leader-follower game over the observed finite strategy grid.

    The leader chooses privacy weight ``lambda``. The follower chooses the
    number of independent release batches ``b`` after observing lambda.
    ``P`` is measured attack advantage (TPR-FPR at the best threshold); older
    result files without that metric use ``2 * max(0, AUC - .5)``.
    """
    required = {"lambda", "budget", "mia_auc", "utility_q"}
    if not isinstance(results, pd.DataFrame) or not required.issubset(results.columns):
        missing = sorted(required - set(results.columns if isinstance(results, pd.DataFrame) else []))
        raise ValueError(f"Missing game columns: {missing}")
    alpha_grid = _grid(alphas, "alphas")
    if not math.isfinite(float(owner_cost_scale)) or float(owner_cost_scale) < 0:
        raise ValueError("owner_cost_scale must be finite and non-negative.")
    if not math.isfinite(float(attacker_cost_per_batch)) or float(attacker_cost_per_batch) < 0:
        raise ValueError("attacker_cost_per_batch must be finite and non-negative.")

    frame = results.copy()
    for column in ("lambda", "budget", "mia_auc", "utility_q"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Game results must be non-empty and contain finite values.")
    if not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("Game results must contain only finite values.")
    if (frame["lambda"] < 0).any() or (frame["budget"] <= 0).any():
        raise ValueError("Observed lambdas must be non-negative and budgets positive.")
    if not np.allclose(frame["budget"], np.round(frame["budget"])):
        raise ValueError("Release batch budgets must be whole numbers.")
    if not frame["mia_auc"].between(0, 1).all() or not frame["utility_q"].between(0, 1).all():
        raise ValueError("mia_auc and utility_q must be in [0, 1].")

    observed_lambdas = _grid(frame["lambda"].unique(), "observed lambdas")
    lambda_grid = observed_lambdas if lambdas is None else _grid(lambdas, "lambdas")
    if not set(lambda_grid).issubset(set(observed_lambdas)):
        raise ValueError("Every requested lambda must have observed experiment results.")
    budget_values = frame["budget"].astype(int).unique()
    budget_grid = [int(value) for value in _grid(budget_values, "budgets", positive=True)]
    frame = frame[frame["lambda"].isin(lambda_grid)].copy()
    if "mia_advantage" in frame:
        frame["mia_advantage"] = pd.to_numeric(frame["mia_advantage"], errors="coerce")
        if frame["mia_advantage"].isna().any() or not np.isfinite(frame["mia_advantage"]).all():
            raise ValueError("mia_advantage must contain finite values.")
        if not frame["mia_advantage"].between(0, 1).all():
            raise ValueError("mia_advantage must be in [0, 1].")
        frame["privacy_p"] = frame["mia_advantage"]
    else:
        frame["privacy_p"] = frame["mia_auc"].map(attacker_success)

    aggregate_columns = {"mia_auc": ("mia_auc", "mean"), "utility_q": ("utility_q", "mean"), "privacy_p": ("privacy_p", "mean")}
    if "mia_advantage" in frame:
        aggregate_columns["mia_advantage"] = ("mia_advantage", "mean")
    agg = frame.groupby(["lambda", "budget"], as_index=False).agg(**aggregate_columns)
    expected = {(lam, budget) for lam in lambda_grid for budget in budget_grid}
    observed = set(zip(agg["lambda"].astype(float), agg["budget"].astype(int)))
    missing_pairs = sorted(expected - observed)
    if missing_pairs:
        raise ValueError(f"Incomplete strategy grid; missing lambda/budget pairs: {missing_pairs}")

    agg["attacker_payoff"] = agg["privacy_p"] - float(attacker_cost_per_batch) * agg["budget"]
    rows: list[dict] = []
    for alpha in alpha_grid:
        for lam in lambda_grid:
            candidate = agg[agg["lambda"] == lam].copy()
            top = float(candidate["attacker_payoff"].max())
            tied = candidate[np.isclose(candidate["attacker_payoff"], top, rtol=1e-10, atol=1e-12)]
            # Deterministic follower tie-break: spend the smaller release budget.
            response = tied.sort_values("budget", kind="stable").iloc[0]
            cost = owner_cost(lam, float(owner_cost_scale))
            advantage = float(response["privacy_p"])
            rows.append({
                "alpha": alpha,
                "lambda": lam,
                "best_budget": int(response["budget"]),
                "utility_q": float(response["utility_q"]),
                "mia_auc": float(response["mia_auc"]),
                "mia_advantage": float(response.get("mia_advantage", advantage)),
                "privacy_p": advantage,
                "owner_cost": cost,
                "owner_payoff": float(response["utility_q"] - alpha * advantage - cost),
                "attacker_payoff": float(response["attacker_payoff"]),
                "attacker_tied_best_responses": int(len(tied)),
                "owner_selected": False,
            })
    out = pd.DataFrame(rows)
    for _, indices in out.groupby("alpha", sort=False).groups.items():
        group = out.loc[indices]
        maximum = float(group["owner_payoff"].max())
        tied = group[np.isclose(group["owner_payoff"], maximum, rtol=1e-10, atol=1e-12)]
        # Deterministic leader tie-break: choose the smaller privacy weight.
        selected_index = tied.sort_values("lambda", kind="stable").index[0]
        out.loc[selected_index, "owner_selected"] = True
    out["attacker_cost_per_batch"] = float(attacker_cost_per_batch)
    out["owner_cost_scale"] = float(owner_cost_scale)
    out["lambda_grid_size"] = len(lambda_grid)
    out["budget_grid_size"] = len(budget_grid)
    out["solution_scope"] = "finite-grid Stackelberg solution"
    return out.sort_values(["alpha", "lambda"], kind="stable").reset_index(drop=True)
