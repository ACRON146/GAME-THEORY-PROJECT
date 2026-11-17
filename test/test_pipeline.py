import json

import numpy as np
import pandas as pd
import pytest
import yaml
import torch

from src.data import TabularEncoder, split_data
from src.evaluation import utility_metrics
from src.experiments import run_experiments, run_one
from src.gan import TabularGAN, privacy_gap_loss


def sample_frame(n=240, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "row_id": np.arange(n),
        "age": rng.integers(18, 75, size=n),
        "education-num": rng.integers(1, 17, size=n),
        "hours-per-week": rng.integers(15, 70, size=n),
        "workclass": rng.choice(["A", "B", "C"], size=n),
        "education": rng.choice(["X", "Y", "Z"], size=n),
        "marital-status": rng.choice(["M", "S"], size=n),
        "occupation": rng.choice(["O", "P", "Q"], size=n),
        "relationship": rng.choice(["R", "T"], size=n),
        "income": np.where(np.arange(n) % 2, ">50K", "<=50K"),
    })


def _small_config(tmp_path, shadow_models=0):
    root = tmp_path
    return {
        "dataset": {"raw_dir": str(root / "raw"), "processed_dir": str(root / "processed"),
                    "sample_rows": 0, "features": ["age", "education-num", "hours-per-week", "workclass",
                    "education", "marital-status", "occupation", "relationship"], "target": "income"},
        "split": {"member_fraction": .5, "reference_fraction": .1, "nonmember_fraction": .2, "test_fraction": .2},
        "training": {"latent_dim": 4, "hidden_dim": 8, "batch_size": 32, "epochs": 1,
                     "learning_rate": .0002, "seeds": [11], "smoke_epochs": 1, "smoke_rows": 120},
        "privacy": {"lambdas": [0., 1.]},
        "attack": {"budgets": [1, 2], "smoke_budgets": [1, 2], "max_candidates": 25,
                   "bootstrap_reps": 5, "shadow_models": shadow_models},
        "game": {"alphas": [0., 1.], "selection_alpha": 1., "fixed_lambda": 1.,
                 "owner_cost_scale": .01, "attacker_cost_per_batch": .01,
                 "attacker_cost_sensitivity": [0., .01], "lambda_grid_sensitivity": [[0., 1.]]},
        "paths": {"experiment_id": "test", "raw_results": str(root / "raw.csv"),
                  "failures": str(root / "failures.csv"), "baseline_results": str(root / "baseline.csv"),
                  "run_config": str(root / "run_config.json"), "summary": str(root / "summary.csv"),
                  "game": str(root / "game.csv"), "strategy_comparison": str(root / "conditions.csv"),
                  "sensitivity": str(root / "sensitivity.csv"), "mia_feasibility": str(root / "mia.csv"),
                  "synthetic_preview": str(root / "preview.csv"), "figures": str(root / "figures"),
                  "report": str(root / "report.md")},
    }


def test_disjoint_splits_and_train_fitted_encoding():
    frame = sample_frame()
    member, reference, nonmember, test, _ = split_data(frame, 17, .5, .1, .2, .2)
    sets = [set(part.row_id) for part in [member, reference, nonmember, test]]
    assert all(not (sets[i] & sets[j]) for i in range(4) for j in range(i + 1, 4))
    assert [len(member), len(reference), len(nonmember), len(test)] == [120, 24, 48, 48]
    categorical = ["workclass", "education", "marital-status", "occupation", "relationship"]
    numeric = ["age", "education-num", "hours-per-week"]
    encoder = TabularEncoder(categorical, numeric).fit(member)
    x = encoder.transform(member)
    back = encoder.inverse(x)
    assert x.shape[1] == sum(encoder.cat_sizes) + len(encoder.numeric)
    assert list(back.columns) == encoder.categorical + encoder.numeric


def test_privacy_loss_gradient_and_training_diagnostics():
    member_scores = torch.tensor([[0.2], [0.4]], requires_grad=True)
    reference_scores = torch.tensor([[0.8], [0.6]], requires_grad=True)
    loss = privacy_gap_loss(member_scores, reference_scores)
    loss.backward()
    assert member_scores.grad is not None and torch.any(member_scores.grad != 0)
    assert reference_scores.grad is not None and torch.any(reference_scores.grad != 0)

    x = np.random.default_rng(2).random((40, 6), dtype=np.float32)
    baseline = TabularGAN(6, latent_dim=4, hidden_dim=8, seed=4)
    regularized = TabularGAN(6, latent_dim=4, hidden_dim=8, seed=4)
    baseline.fit(x, x[::-1].copy(), 0., epochs=1, batch_size=20)
    regularized.fit(x, x[::-1].copy(), 2., epochs=1, batch_size=20)
    assert baseline.training_history[0]["weighted_privacy_loss"] == 0
    assert regularized.training_history[0]["privacy_gap_loss"] >= 0
    assert regularized.training_history[0]["weighted_privacy_loss"] == pytest.approx(
        2 * regularized.training_history[0]["privacy_gap_loss"])
    assert np.max(np.abs(baseline.sample(3) - regularized.sample(3))) > 0
    assert baseline.sample(3).shape == (3, 6)
    with pytest.raises(ValueError):
        baseline.fit(x, x, -1, epochs=1, batch_size=10)
    with pytest.raises(ValueError):
        baseline.fit(np.full_like(x, np.nan), x, 1, epochs=1, batch_size=10)


def test_known_toy_distribution_metrics():
    x = np.array([[1., 0., 0.], [0., 1., 1.], [1., 0., 0.], [0., 1., 1.]])
    y = np.array([0, 1, 0, 1])
    metrics = utility_metrics(x, y, x, y, x.copy(), y.copy(), [2], ["cat_a", "cat_b", "numeric"])
    assert metrics["utility_q"] == pytest.approx(1.)
    assert metrics["categorical_similarity"] == pytest.approx(1.)
    assert metrics["numerical_similarity"] == pytest.approx(1.)
    assert metrics["correlation_similarity"] == pytest.approx(1.)
    assert metrics["per_column_distribution_similarity"] == pytest.approx(1.)
    assert json.loads(metrics["per_column_similarity_json"])["numeric"] == pytest.approx(1.)


def test_fixed_seed_experiment_is_reproducible(tmp_path):
    cfg = _small_config(tmp_path)
    frame = sample_frame(240)
    first = run_one(frame, cfg, 11, 0.0, [1], 1)[0]
    second = run_one(frame, cfg, 11, 0.0, [1], 1)[0]
    for field in ["utility_q", "mia_auc", "mia_advantage", "distance_mia_auc", "synthetic_test_f1"]:
        assert first[field] == pytest.approx(second[field], abs=1e-10)
    assert first["split_id"] == second["split_id"]


def test_failed_runs_are_saved(tmp_path, monkeypatch):
    cfg = _small_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr("src.experiments.ensure_data", lambda _cfg: sample_frame(120))
    monkeypatch.setattr("src.experiments.run_one", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unstable model")))
    with pytest.raises(RuntimeError, match="Every experiment failed"):
        run_experiments(config_path, mode="full")
    failures = pd.read_csv(cfg["paths"]["failures"])
    assert failures.iloc[0]["status"] == "failed"
    assert failures.iloc[0]["error"] == "unstable model"
