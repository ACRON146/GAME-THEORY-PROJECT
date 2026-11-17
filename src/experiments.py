from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .attacks import mia_distance_attack, mia_shadow_attack
from .data import TabularEncoder, load_dataset, prepare_data, split_data
from .evaluation import utility_metrics
from .game import solve_stackelberg
from .gan import TabularGAN


def read_config(path="configs/default.yaml"):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def _validate_config(cfg):
    lambdas = cfg["privacy"]["lambdas"]
    if not lambdas or any(not math.isfinite(float(value)) or float(value) < 0 for value in lambdas):
        raise ValueError("privacy.lambdas must be a non-empty grid of finite, non-negative values.")
    if len(set(map(float, lambdas))) != len(lambdas):
        raise ValueError("privacy.lambdas cannot contain duplicates.")
    if 0.0 not in set(map(float, lambdas)):
        raise ValueError("privacy.lambdas must include 0 for the ordinary GAN baseline.")
    budgets = cfg["attack"]["budgets"]
    if not budgets or any(int(value) <= 0 or int(value) != float(value) for value in budgets):
        raise ValueError("attack.budgets must be a non-empty grid of positive whole numbers.")
    if len(set(map(int, budgets))) != len(budgets):
        raise ValueError("attack.budgets cannot contain duplicates.")
    seeds = cfg["training"]["seeds"]
    if not seeds or len(set(map(int, seeds))) != len(seeds):
        raise ValueError("training.seeds must be a non-empty set of unique integers.")
    split = cfg["split"]
    fractions = [float(split[key]) for key in ("member_fraction", "reference_fraction", "nonmember_fraction", "test_fraction")]
    if any(not math.isfinite(value) or value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0, atol=1e-8):
        raise ValueError("Configured member, reference, nonmember and test fractions must be positive and sum to 1.")
    if int(cfg["training"]["epochs"]) <= 0 or int(cfg["training"]["batch_size"]) <= 0:
        raise ValueError("Training epochs and batch size must be positive.")
    for key in ("latent_dim", "hidden_dim"):
        if int(cfg["training"][key]) <= 0:
            raise ValueError(f"training.{key} must be positive.")
    if not math.isfinite(float(cfg["training"]["learning_rate"])) or float(cfg["training"]["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be finite and positive.")
    if int(cfg["attack"].get("shadow_models", 0)) < 0:
        raise ValueError("attack.shadow_models cannot be negative.")
    if int(cfg["attack"].get("max_candidates", 0)) <= 0 or int(cfg["attack"].get("bootstrap_reps", 0)) <= 0:
        raise ValueError("Attack candidate and bootstrap counts must be positive.")
    alpha_values = [float(value) for value in cfg["game"]["alphas"]]
    if not alpha_values or any(not math.isfinite(value) or value < 0 for value in alpha_values):
        raise ValueError("game.alphas must be a non-empty, finite, non-negative grid.")
    fixed_lambda = float(cfg["game"].get("fixed_lambda", 1.0))
    if fixed_lambda not in set(map(float, lambdas)) or fixed_lambda <= 0:
        raise ValueError("game.fixed_lambda must be positive and included in privacy.lambdas.")
    for key in ("owner_cost_scale", "attacker_cost_per_batch"):
        value = float(cfg["game"][key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"game.{key} must be finite and non-negative.")


def _paths(cfg):
    return cfg["dataset"]["raw_dir"], Path(cfg["dataset"]["processed_dir"]) / "adult_clean.csv"


def ensure_data(cfg):
    raw, processed = _paths(cfg)
    if not processed.exists():
        prepare_data(raw, processed, cfg["dataset"]["features"], cfg["dataset"]["target"])
    return load_dataset(processed, cfg["dataset"].get("sample_rows", 0), cfg["training"]["seeds"][0])


def _target_values(frame, target):
    values = frame[target].astype(str).str.strip().str.rstrip(".")
    return (values == ">50K").astype(int).to_numpy()


def _encoded(member, reference, nonmember, test, cfg):
    features = cfg["dataset"]["features"]
    target = cfg["dataset"]["target"]
    categorical = [column for column in features if not pd.api.types.is_numeric_dtype(member[column])]
    numeric = [column for column in features if column not in categorical]
    encoder = TabularEncoder(categorical, numeric).fit(member)
    mx, rx = encoder.transform(member), encoder.transform(reference)
    nx, tx = encoder.transform(nonmember), encoder.transform(test)
    my, ry = _target_values(member, target), _target_values(reference, target)
    ny, ty = _target_values(nonmember, target), _target_values(test, target)
    # The label is an additional bounded GAN output; it is included in the
    # membership candidate and release rows because it is part of the record.
    return encoder, np.c_[mx, my], np.c_[rx, ry], np.c_[nx, ny], mx, nx, tx, my, ny, ty


def _postprocess_gan_rows(matrix, encoder):
    rows = np.asarray(matrix, dtype=np.float32).copy()
    feature_width = len(encoder.columns)
    offset = 0
    for size in encoder.cat_sizes:
        indices = np.argmax(rows[:, offset:offset + size], axis=1)
        rows[:, offset:offset + size] = 0.0
        rows[np.arange(len(rows)), offset + indices] = 1.0
        offset += size
    # Keep numeric columns bounded and threshold the binary income output.
    rows[:, offset:feature_width] = np.clip(rows[:, offset:feature_width], 0.0, 1.0)
    rows[:, feature_width] = (rows[:, feature_width] >= 0.5).astype(np.float32)
    return rows


def _make_shadow_sets(reference_g, encoder, cfg, seed, lam, max_budget, epochs):
    count = int(cfg["attack"].get("shadow_models", 0))
    if count == 0:
        return [], 0.0, []
    order = np.random.default_rng(int(seed) + 81001).permutation(len(reference_g))
    groups = [group for group in np.array_split(order, count) if len(group) >= 8]
    if not groups:
        raise ValueError("Not enough reference records to create a shadow-model attack.")
    shadows, total_seconds, histories = [], 0.0, []
    batch_size = int(cfg["training"]["batch_size"])
    latent_dim = int(cfg["training"]["latent_dim"])
    hidden_dim = int(cfg["training"]["hidden_dim"])
    learning_rate = float(cfg["training"]["learning_rate"])
    for index, group in enumerate(groups):
        local = np.random.default_rng(int(seed) + 82001 + index).permutation(group)
        n = len(local)
        member_count = max(2, min(n - 4, int(round(n * 0.6))))
        reference_count = max(2, min(n - member_count - 2, int(round(n * 0.2))))
        if n - member_count - reference_count < 2:
            raise ValueError("A shadow split needs at least two member, reference and non-member records.")
        shadow_members = reference_g[local[:member_count]]
        shadow_reference = reference_g[local[member_count:member_count + reference_count]]
        shadow_nonmembers = reference_g[local[member_count + reference_count:]]
        shadow_seed = int(seed) + 90000 + index
        model = TabularGAN(reference_g.shape[1], latent_dim, hidden_dim, learning_rate, shadow_seed)
        total_seconds += model.fit(shadow_members, shadow_reference, lam, epochs, batch_size)
        histories.append({"shadow_index": index, "seed": shadow_seed, "epochs": model.training_history})
        release_rows = max(1, int(math.ceil(len(shadow_members) / max_budget)))
        releases = _release_batches(model, max_budget, release_rows, encoder)
        shadows.append((shadow_members, shadow_nonmembers, releases))
    return shadows, total_seconds, histories


def _release_batches(model, count, size, encoder):
    return [_postprocess_gan_rows(model.sample(size), encoder) for _ in range(int(count))]


def run_one(frame, cfg, seed, lam, budgets, epochs):
    split = cfg["split"]
    member, reference, nonmember, test, split_id = split_data(
        frame, seed, split["member_fraction"], split["reference_fraction"],
        split["nonmember_fraction"], split["test_fraction"])
    enc, member_g, reference_g, nonmember_g, mx, nx, tx, my, ny, ty = _encoded(
        member, reference, nonmember, test, cfg)
    gan = TabularGAN(member_g.shape[1], cfg["training"]["latent_dim"], cfg["training"]["hidden_dim"],
                     cfg["training"]["learning_rate"], seed)
    target_train_seconds = gan.fit(member_g, reference_g, lam, epochs, cfg["training"]["batch_size"])
    inference_start = time.perf_counter()
    synth_rows = _postprocess_gan_rows(gan.sample(len(member)), enc)
    synthetic_release_seconds = time.perf_counter() - inference_start
    synthetic_x, synthetic_y = synth_rows[:, :-1], synth_rows[:, -1].astype(int)
    utility = utility_metrics(mx, my, tx, ty, synthetic_x, synthetic_y, enc.cat_sizes, enc.columns)

    preview = enc.inverse(synthetic_x[:min(100, len(synthetic_x))])
    preview[cfg["dataset"]["target"]] = np.where(synthetic_y[:len(preview)] == 1, ">50K", "<=50K")
    preview = preview[cfg["dataset"]["features"] + [cfg["dataset"]["target"]]]
    preview_path = Path(cfg["paths"].get("synthetic_preview", "results/processed/synthetic_preview.csv"))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(preview_path, index=False)

    max_budget = max(map(int, budgets))
    release_rows = max(1, int(math.ceil(len(member_g) / max_budget)))
    release_start = time.perf_counter()
    target_releases = _release_batches(gan, max_budget, release_rows, enc)
    synthetic_release_seconds += time.perf_counter() - release_start
    shadows, shadow_train_seconds, shadow_histories = _make_shadow_sets(reference_g, enc, cfg, seed, lam, max_budget, epochs)

    candidate_members, candidate_nonmembers = member_g[:max(1, len(member_g) // 2)], nonmember_g[:max(1, len(nonmember_g) // 2)]
    max_candidates = int(cfg["attack"]["max_candidates"])
    bootstrap = int(cfg["attack"].get("bootstrap_reps", 100))
    rows = []
    for budget in budgets:
        budget = int(budget)
        distance = mia_distance_attack(candidate_members, candidate_nonmembers, target_releases[:budget],
                                       max_candidates, enc.cat_sizes, seed + budget, bootstrap)
        if shadows:
            shadow_results = mia_shadow_attack(
                candidate_members, candidate_nonmembers, target_releases[:budget],
                [(m, n, releases[:budget]) for m, n, releases in shadows],
                max_candidates, enc.cat_sizes, seed + 1000 + budget, bootstrap)
            shadow_auc = shadow_results["mia_auc"]
            shadow_advantage = shadow_results["mia_advantage"]
        else:
            shadow_results = {"mia_auc": float("nan"), "mia_advantage": float("nan"),
                              "mia_auc_ci_low": float("nan"), "mia_auc_ci_high": float("nan"),
                              "mia_advantage_ci_low": float("nan"), "mia_advantage_ci_high": float("nan"),
                              "mia_accuracy": float("nan"), "mia_precision": float("nan"),
                              "mia_recall": float("nan"), "attack_seconds": 0.0}
            shadow_auc, shadow_advantage = float("nan"), float("nan")
        auc_candidates = [distance["mia_auc"]] + ([shadow_auc] if math.isfinite(shadow_auc) else [])
        advantage_candidates = [distance["mia_advantage"]] + ([shadow_advantage] if math.isfinite(shadow_advantage) else [])
        best_attack_index = int(np.argmax(auc_candidates))
        best_advantage_index = int(np.argmax(advantage_candidates))
        combined = {
            "mia_auc": float(max(auc_candidates)),
            "mia_advantage": float(max(advantage_candidates)),
            "mia_auc_ci_low": distance["mia_auc_ci_low"] if best_attack_index == 0 else shadow_results["mia_auc_ci_low"],
            "mia_auc_ci_high": distance["mia_auc_ci_high"] if best_attack_index == 0 else shadow_results["mia_auc_ci_high"],
            "mia_advantage_ci_low": distance["mia_advantage_ci_low"] if best_advantage_index == 0
                else shadow_results["mia_advantage_ci_low"],
            "mia_advantage_ci_high": distance["mia_advantage_ci_high"] if best_advantage_index == 0
                else shadow_results["mia_advantage_ci_high"],
        }
        rows.append({
            "experiment_id": cfg["paths"].get("experiment_id", "adult_tabular_gan"),
            "seed": int(seed), "split_id": split_id, "lambda": float(lam), "budget": budget,
            "epochs": int(epochs), "status": "ok", "member_rows": len(member_g),
            "reference_rows": len(reference_g), "nonmember_rows": len(nonmember_g), "test_rows": len(tx),
            "train_seconds": float(target_train_seconds), "shadow_train_seconds": float(shadow_train_seconds),
            "synthesis_seconds": float(synthetic_release_seconds), "distance_attack_seconds": float(distance["attack_seconds"]),
            "shadow_attack_seconds": float(shadow_results["attack_seconds"]),
            "distance_mia_auc": float(distance["mia_auc"]), "distance_mia_advantage": float(distance["mia_advantage"]),
            "shadow_mia_auc": float(shadow_auc), "shadow_mia_advantage": float(shadow_advantage),
            "shadow_accuracy": float(shadow_results["mia_accuracy"]),
            "random_auc_baseline": 0.5, "random_advantage_baseline": 0.0,
            "target_d_adversarial_loss": float(gan.training_history[-1]["discriminator_adversarial_loss"]),
            "target_privacy_gap_loss": float(gan.training_history[-1]["privacy_gap_loss"]),
            "target_weighted_privacy_loss": float(gan.training_history[-1]["weighted_privacy_loss"]),
            "target_generator_loss": float(gan.training_history[-1]["generator_loss"]),
            "target_training_history_json": json.dumps(gan.training_history, separators=(",", ":")),
            "shadow_training_history_json": json.dumps(shadow_histories, separators=(",", ":")),
            **combined, **utility,
        })
    return rows


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_experiments(config_path="configs/default.yaml", mode="full"):
    cfg = read_config(config_path)
    _validate_config(cfg)
    frame = ensure_data(cfg)
    if mode == "smoke":
        first_seed = int(cfg["training"]["seeds"][0])
        frame = frame.sample(min(len(frame), int(cfg["training"]["smoke_rows"])), random_state=first_seed).reset_index(drop=True)
        seeds = [int(seed) for seed in cfg["training"].get("smoke_seeds", [first_seed])]
        if not set(seeds).issubset(set(map(int, cfg["training"]["seeds"]))):
            raise ValueError("training.smoke_seeds must be a subset of training.seeds.")
        fixed_lambda = float(cfg["game"].get("fixed_lambda", 1.0))
        lambdas = sorted(set([0.0, fixed_lambda]))
        budgets = cfg["attack"].get("smoke_budgets", [1, 2])
        epochs = int(cfg["training"]["smoke_epochs"])
    elif mode == "full":
        seeds, lambdas, budgets = cfg["training"]["seeds"], cfg["privacy"]["lambdas"], cfg["attack"]["budgets"]
        epochs = int(cfg["training"]["epochs"])
    else:
        raise ValueError("mode must be 'smoke' or 'full'")
    if len(frame) < 20:
        raise ValueError("The loaded data sample is too small for the configured split and evaluation.")

    config_fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    _write_json(cfg["paths"].get("run_config", "results/run_config.json"),
                {"mode": mode, "config_fingerprint": config_fingerprint, "config": cfg,
                 "seeds": list(map(int, seeds)), "lambdas": list(map(float, lambdas)),
                 "budgets": list(map(int, budgets)), "epochs": epochs})
    all_rows, failures = [], []
    for seed in seeds:
        for lam in lambdas:
            print(f"training seed={seed} lambda={lam}")
            try:
                all_rows.extend(run_one(frame, cfg, int(seed), float(lam), budgets, epochs))
            except Exception as exc:
                failures.append({"seed": int(seed), "lambda": float(lam), "status": "failed",
                                 "error_type": type(exc).__name__, "error": str(exc)})
    raw_path = Path(cfg["paths"]["raw_results"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(all_rows)
    data.to_csv(raw_path, index=False)
    baseline_path = Path(cfg["paths"].get("baseline_results", "results/raw/baseline.csv"))
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    if not data.empty and "lambda" in data:
        data[data["lambda"] == 0.0].to_csv(baseline_path, index=False)
    else:
        pd.DataFrame(columns=data.columns).to_csv(baseline_path, index=False)
    failure_path = Path(cfg["paths"].get("failures", "results/raw/failures.csv"))
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["seed", "lambda", "status", "error_type", "error"]).to_csv(failure_path, index=False)
    if not data.empty:
        make_summary(data, cfg)
    if data.empty:
        raise RuntimeError(f"Every experiment failed; inspect {failure_path}.")
    print(f"Saved {len(data)} result rows and {len(failures)} failure records.")
    return data


def make_summary(data, cfg):
    out = Path(cfg["paths"]["summary"])
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["utility_q", "synthetic_test_f1", "real_synthetic_f1_similarity", "mia_auc", "mia_advantage", "distance_mia_auc",
               "shadow_mia_auc", "shadow_mia_advantage", "per_column_distribution_similarity",
               "correlation_similarity", "categorical_similarity", "numerical_similarity", "train_seconds",
               "shadow_train_seconds", "synthesis_seconds"]
    metrics = [metric for metric in metrics if metric in data.columns]
    summary = data.groupby(["lambda", "budget"], dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(part) for part in col if part != "") if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(out, index=False)
    return summary


def _selected_alpha(game, configured_alpha):
    values = game["alpha"].unique()
    return float(min(values, key=lambda value: (abs(float(value) - float(configured_alpha)), float(value))))


def _build_strategy_comparison(data, game, cfg):
    alpha = _selected_alpha(game, cfg["game"].get("selection_alpha", 2.0))
    candidates = game[game["alpha"] == alpha].copy()
    if candidates.empty:
        raise ValueError("No game solution rows are available for comparison.")
    fixed_lam = float(cfg["game"].get("fixed_lambda", 1.0))
    if fixed_lam not in set(candidates["lambda"].astype(float)):
        raise ValueError(f"Configured fixed_lambda={fixed_lam} is absent from the evaluated lambda grid.")
    q_means = data.groupby("lambda")["utility_q"].mean()
    utility_only_lam = float(sorted(q_means.items(), key=lambda pair: (-float(pair[1]), float(pair[0])))[0][0])
    game_lam = float(candidates.loc[candidates["owner_selected"], "lambda"].iloc[0])
    conditions = [
        ("A_ordinary_gan", 0.0, "lambda=0; no strategy optimization"),
        ("B_fixed_regularized_gan", fixed_lam, "configured fixed lambda; attacker response measured"),
        ("C_game_selected_gan", game_lam, f"Stackelberg solution at alpha={alpha:g}"),
        ("D_utility_only_selection", utility_only_lam, "lambda maximizing mean utility_q without attacker payoff"),
    ]
    rows = []
    for condition, lam, rule in conditions:
        if lam not in set(candidates["lambda"].astype(float)):
            raise ValueError(f"Comparison lambda {lam} is missing from the observed game solution.")
        solution = candidates[candidates["lambda"] == lam].iloc[0]
        raw = data[(data["lambda"] == lam) & (data["budget"] == int(solution["best_budget"]))]
        rows.append({"condition": condition, "selection_rule": rule, "alpha": alpha,
                     "lambda": lam, "best_budget": int(solution["best_budget"]),
                     "utility_q": float(solution["utility_q"]), "mia_auc": float(solution["mia_auc"]),
                     "mia_advantage": float(solution["mia_advantage"]),
                     "distance_mia_auc": float(raw["distance_mia_auc"].mean()),
                     "shadow_mia_auc": float(raw["shadow_mia_auc"].mean()),
                     "attacker_payoff": float(solution["attacker_payoff"]),
                     "owner_payoff": float(solution["owner_payoff"]),
                     "train_seconds_mean": float(raw["train_seconds"].mean()),
                     "shadow_train_seconds_mean": float(raw["shadow_train_seconds"].mean()),
                     "runs": int(raw["seed"].nunique())})
    return pd.DataFrame(rows)


def _write_sensitivity(data, cfg):
    rows = []
    base_lambdas = sorted(data["lambda"].astype(float).unique())
    base_alpha = cfg["game"]["alphas"]
    owner_scale = float(cfg["game"]["owner_cost_scale"])
    for cost in cfg["game"].get("attacker_cost_sensitivity", [cfg["game"]["attacker_cost_per_batch"]]):
        result = solve_stackelberg(data, base_alpha, owner_scale, float(cost), base_lambdas)
        for alpha, group in result.groupby("alpha"):
            selected = group[group["owner_selected"]].iloc[0]
            rows.append({"scenario": "attacker_cost", "scenario_value": float(cost), "alpha": float(alpha),
                         "lambda": float(selected["lambda"]), "best_budget": int(selected["best_budget"]),
                         "owner_payoff": float(selected["owner_payoff"]), "attacker_payoff": float(selected["attacker_payoff"]),
                         "grid": json.dumps(base_lambdas)})
    for candidate_grid in cfg["game"].get("lambda_grid_sensitivity", []):
        # A smoke run deliberately trains only a small lambda subset. Restrict
        # sensitivity grids to observed strategies instead of inventing data.
        grid = sorted(set(map(float, candidate_grid)) & set(base_lambdas))
        if not grid:
            continue
        result = solve_stackelberg(data, base_alpha, owner_scale,
                                   float(cfg["game"]["attacker_cost_per_batch"]), grid)
        label = json.dumps(grid)
        for alpha, group in result.groupby("alpha"):
            selected = group[group["owner_selected"]].iloc[0]
            rows.append({"scenario": "lambda_grid", "scenario_value": label, "alpha": float(alpha),
                         "lambda": float(selected["lambda"]), "best_budget": int(selected["best_budget"]),
                         "owner_payoff": float(selected["owner_payoff"]), "attacker_payoff": float(selected["attacker_payoff"]),
                         "grid": label})
    out = Path(cfg["paths"].get("sensitivity", "results/tables/sensitivity.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def solve(config_path="configs/default.yaml"):
    cfg = read_config(config_path)
    _validate_config(cfg)
    raw_path = Path(cfg["paths"]["raw_results"])
    if not raw_path.exists():
        raise FileNotFoundError("Experiment results missing. Run run-experiments first.")
    data = pd.read_csv(raw_path)
    if data.empty or "status" in data.columns and not data["status"].eq("ok").all():
        data = data[data["status"].eq("ok")] if "status" in data.columns else data
    solution = solve_stackelberg(data, cfg["game"]["alphas"], cfg["game"]["owner_cost_scale"],
                                 cfg["game"]["attacker_cost_per_batch"])
    dest = Path(cfg["paths"]["game"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    solution.to_csv(dest, index=False)
    comparison = _build_strategy_comparison(data, solution, cfg)
    comparison_path = Path(cfg["paths"].get("strategy_comparison", "results/tables/strategy_comparison.csv"))
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    _write_sensitivity(data, cfg)
    return solution


def analyze(config_path="configs/default.yaml"):
    cfg = read_config(config_path)
    _validate_config(cfg)
    source = Path(cfg["paths"]["raw_results"])
    if not source.exists():
        raise FileNotFoundError("Experiment results missing. Run experiments first.")
    data = pd.read_csv(source)
    if data.empty:
        raise ValueError("Experiment results contain no successful rows.")
    if "status" in data.columns:
        data = data[data["status"] == "ok"].copy()
    game = solve(config_path)
    comparison_path = Path(cfg["paths"].get("strategy_comparison", "results/tables/strategy_comparison.csv"))
    comparison = pd.read_csv(comparison_path)
    figures = Path(cfg["paths"]["figures"])
    figures.mkdir(parents=True, exist_ok=True)
    alpha = _selected_alpha(game, cfg["game"].get("selection_alpha", 2.0))
    selected_game = game[game["alpha"] == alpha].sort_values("lambda")

    summary = data.groupby(["lambda", "budget"], as_index=False).agg(
        utility_q=("utility_q", "mean"), utility_sd=("utility_q", "std"),
        mia_auc=("mia_auc", "mean"), mia_auc_sd=("mia_auc", "std"),
        mia_auc_ci_low=("mia_auc_ci_low", "mean"), mia_auc_ci_high=("mia_auc_ci_high", "mean"),
        mia_advantage=("mia_advantage", "mean"), mia_advantage_sd=("mia_advantage", "std"),
        mia_advantage_ci_low=("mia_advantage_ci_low", "mean"),
        mia_advantage_ci_high=("mia_advantage_ci_high", "mean"),
        train_seconds=("train_seconds", "mean"), shadow_train_seconds=("shadow_train_seconds", "mean"))
    min_budget = int(summary["budget"].min())
    at_minimum_budget = summary[summary["budget"] == min_budget]
    plots = [
        ("utility_vs_lambda", "utility_q", "utility_sd", "Downstream utility Q", "Held-out utility by lambda"),
        ("mia_auc_vs_lambda", "mia_auc", "mia_auc_sd", "Best measured attack ROC-AUC", "Membership attack AUC by lambda"),
        ("mia_advantage_vs_lambda", "mia_advantage", "mia_advantage_sd", "Attack advantage", "Membership attack advantage by lambda"),
        ("runtime_vs_lambda", "train_seconds", None, "Target training seconds", "Target GAN training time"),
    ]
    for name, metric, error, ylabel, title in plots:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.errorbar(at_minimum_budget["lambda"], at_minimum_budget[metric],
                    yerr=at_minimum_budget[error] if error else None, marker="o", capsize=3)
        ax.set(xlabel="Privacy regularization strength λ", ylabel=ylabel,
               title=f"{title} (b={min_budget})")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(summary["utility_q"], summary["mia_advantage"], c=summary["lambda"], cmap="viridis", s=48)
    for _, row in summary.iterrows():
        ax.annotate(f"λ={row['lambda']:g}, b={row['budget']:g}", (row.utility_q, row.mia_advantage), fontsize=7)
    ax.set(xlabel="Downstream utility Q", ylabel="Worst measured attack advantage",
           title="Privacy-utility measurements by strategy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "privacy_utility.png", dpi=180)
    fig.savefig(figures / "pareto.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for alpha_value, group in game.groupby("alpha"):
        ax.plot(group["lambda"], group["owner_payoff"], marker="o", label=f"α={alpha_value:g}")
    ax.set(xlabel="Privacy regularization strength λ", ylabel="Owner payoff",
           title="Owner choice sensitivity to privacy importance α")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "owner_alpha.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(selected_game["lambda"], selected_game["best_budget"], marker="o")
    ax.set(xlabel="Privacy regularization strength λ", ylabel="Attacker best-response batches b",
           title=f"Attacker best response at α={alpha:g}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "attacker_best_response.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(comparison["condition"], comparison["utility_q"], color="#5183b8")
    ax.set_ylabel("Downstream utility Q")
    ax.set_title("Four strategy conditions; privacy is reported separately")
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax2 = ax.twinx()
    ax2.plot(comparison["condition"], comparison["mia_advantage"], color="#a43d32", marker="o")
    ax2.set_ylabel("Worst measured attack advantage")
    fig.tight_layout()
    fig.savefig(figures / "strategy_conditions.png", dpi=180)
    fig.savefig(figures / "baseline_comparison.png", dpi=180)
    plt.close(fig)

    payoff_data = data.groupby(["lambda", "budget"], as_index=False).agg(
        privacy_p=("mia_advantage", "mean"))
    payoff_data["attacker_payoff"] = payoff_data["privacy_p"] - float(cfg["game"]["attacker_cost_per_batch"]) * payoff_data["budget"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for lam, group in payoff_data.groupby("lambda"):
        ax.plot(group["budget"], group["attacker_payoff"], marker="o", label=f"λ={lam:g}")
    ax.set(xlabel="Independent release batches b", ylabel="Attacker payoff",
           title="Attacker payoff across release budgets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "attacker_payoff_budget.png", dpi=180)
    plt.close(fig)

    make_summary(data, cfg)
    run_config_path = Path(cfg["paths"].get("run_config", "results/run_config.json"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    mode = run_config.get("mode", "unknown")
    report = Path(cfg["paths"]["report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    summary_text = at_minimum_budget[["lambda", "utility_q", "utility_sd", "mia_auc", "mia_auc_sd",
                                      "mia_auc_ci_low", "mia_auc_ci_high", "mia_advantage", "mia_advantage_sd",
                                      "mia_advantage_ci_low", "mia_advantage_ci_high", "train_seconds"]].to_string(index=False)
    condition_text = comparison[["condition", "lambda", "best_budget", "utility_q", "mia_auc", "mia_advantage"]].to_string(index=False)
    chosen = comparison[comparison["condition"] == "C_game_selected_gan"].iloc[0]
    ordinary = comparison[comparison["condition"] == "A_ordinary_gan"].iloc[0]
    fixed = comparison[comparison["condition"] == "B_fixed_regularized_gan"].iloc[0]
    observed_interpretation = (
        f"At α={alpha:g}, the saved runs selected λ={chosen['lambda']:g} with best-response budget "
        f"b={int(chosen['best_budget'])}; utility Q={chosen['utility_q']:.3f} and strongest measured attack "
        f"advantage={chosen['mia_advantage']:.3f}. Compared with the ordinary GAN, the fixed regularized "
        f"condition changed Q by {fixed['utility_q'] - ordinary['utility_q']:+.3f} and attack advantage by "
        f"{fixed['mia_advantage'] - ordinary['mia_advantage']:+.3f}. These values describe the saved experimental output only."
    )
    if mode == "smoke":
        observed_interpretation += f" This {data['seed'].nunique()}-seed smoke run is a pipeline pilot and provides no confirmatory evidence that game selection improves on simpler strategies."
    failures_path = Path(cfg["paths"].get("failures", "results/raw/failures.csv"))
    failed_runs = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    report.write_text(
        "# Game-Theoretic GAN Privacy: Experimental Report\n\n"
        f"## Study status\n\nThis report was regenerated from the saved experiment rows. The recorded run mode is **{mode}**. "
        f"It contains {data['seed'].nunique()} independent seed(s), {data['lambda'].nunique()} lambda values, "
        f"and {len(data)} successful seed/lambda/budget rows. "
        + ("A smoke run is a pipeline check, not confirmatory evidence. Run the configured full study before making research claims.\n\n"
           if mode == "smoke" else "The reported evidence is limited to the configured seeds, model, data and attacks.\n\n")
        + "## Research question and contribution\n\n"
        "The project studies how a data owner can select a privacy-regularization strength for a tabular GAN when an attacker can query multiple synthetic releases. "
        "It evaluates whether a finite-grid Stackelberg strategy adds value over an ordinary GAN, a fixed regularized GAN, and a utility-only strategy choice.\n\n"
        "## Model and threat model\n\n"
        "The generator and discriminator are compact multilayer perceptrons trained on a train-fitted one-hot/min-max representation. The discriminator minimizes real/fake binary cross-entropy plus λ times the squared difference between mean discriminator scores on GAN members and a separate regularizer-reference set. λ=0 recovers the ordinary GAN loss. The privacy-gap loss and adversarial losses are logged separately. The reference set is used during training and is therefore not used as an untouched membership-attack non-member set.\n\n"
        "The evaluation attacker receives independent synthetic release batches. Its baseline scores a candidate by nearest mixed-feature distance to a released row. A shadow-model attacker trains a logistic classifier on release-distance features from disjoint shadow records; the classifier and its threshold are fitted on shadow membership labels, not on target evaluation labels. The evaluation also reports ROC-AUC, maximum ROC TPR-FPR advantage, confidence intervals, and the random baselines (AUC 0.5 and advantage 0). These are empirical attacks and do not establish formal privacy.\n\n"
        "## Stackelberg game\n\n"
        "The leader is the data owner and chooses λ from the configured finite grid. The follower observes λ and chooses b, the number of independent synthetic release batches. For each candidate λ, the attacker chooses a best response that maximizes `U_A(λ,b) = P(λ,b) - c_A b`, where P is the strongest measured attack advantage and c_A is the configured cost per release batch. The owner then chooses λ that maximizes `U_O(λ,b*(λ)) = Q(λ) - α P(λ,b*(λ)) - C_O(λ)`. Q is synthetic-trained downstream F1 evaluated on held-out real data; the real-trained reference F1 is reported separately. `C_O(λ) = c_O λ/(1+λ)`. Ties resolve to lower b for the attacker and lower λ for the owner. Results are finite-grid Stackelberg solutions, not continuous global equilibria.\n\n"
        "## Data and reproducibility\n\n"
        f"Dataset: UCI Adult, using the configured features `{', '.join(cfg['dataset']['features'])}` and target `{cfg['dataset']['target']}`. "
        f"Per seed, the disjoint split assigns {cfg['split']['member_fraction']:.0%} to GAN members, "
        f"{cfg['split']['reference_fraction']:.0%} to regularizer reference, {cfg['split']['nonmember_fraction']:.0%} to untouched attack non-members, "
        f"and {cfg['split']['test_fraction']:.0%} to held-out utility evaluation. Encoders are fitted on members only. "
        f"The recorded experiment configuration is in `{run_config_path.as_posix()}`; raw rows are in `{cfg['paths']['raw_results']}` and failures in `{failures_path.as_posix()}` ({len(failed_runs)} failed configuration(s)).\n\n"
        "## Results by lambda\n\n"
        f"The following table summarizes the smallest configured attack budget, b={min_budget}. Means and sample standard deviations are across independent seeds; standard deviation is undefined for single-seed runs. Attack confidence-interval endpoints are averaged from within-run candidate bootstraps.\n\n"
        "```text\n" + summary_text + "\n```\n\n"
        "## Four strategy conditions\n\n"
        f"The comparison uses α={alpha:g}. All conditions are evaluated at the attacker's measured best-response budget. The utility-only condition chooses λ by mean Q without considering attacker payoff.\n\n"
        "```text\n" + condition_text + "\n```\n\n"
        "## Interpretation of saved measurements\n\n" + observed_interpretation + "\n\n"
        "## Figures\n\n"
        "![Privacy-utility measurements](../results/figures/privacy_utility.png)\n\n"
        "![Attacker best response](../results/figures/attacker_best_response.png)\n\n"
        "![Four strategy conditions](../results/figures/strategy_conditions.png)\n\n"
        "Privacy, utility and runtime remain separate measurements. Aggregated metrics are saved in `results/tables/summary.csv`. The game table stores each λ's attacker best response and owner payoff for every configured α. Sensitivity results vary attacker cost and the available λ grid.\n\n"
        "## Limitations and interpretation\n\n"
        "The score-gap penalty is a hypothesis about one discriminator signal; it does not guarantee that membership information is removed. The shadow attack uses distance summaries and logistic regression, so other attacks may perform better. Attack confidence intervals describe candidate resampling within a run; between-seed variation is shown separately. The compact GAN and Adult benchmark do not establish safety on other datasets or architectures. No differential privacy guarantee or DP baseline is implemented. If this report comes from a smoke run, its measurements are only pipeline diagnostics.\n\n"
        "## Reproduction\n\n"
        "From the project directory, install `requirements.txt`, run `python -m src.cli prepare-data`, then `python -m src.cli run-experiments --mode full`, followed by `python -m src.cli analyze`. This regenerates raw rows, aggregates, game solutions, sensitivity tables, comparison tables, figures and this report from the saved outputs.\n\n"
        "## References\n\n"
        "- Xu et al. (2019), *Modeling Tabular Data using Conditional GAN*, NeurIPS. https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan\n"
        "- Shokri et al. (2017), *Membership Inference Attacks against Machine Learning Models*, IEEE S&P. https://doi.org/10.1109/SP.2017.41\n"
        "- Hyeong et al. (2022), *An Empirical Study on the Membership Inference Attack against Tabular Data Synthesis Models*. https://arxiv.org/abs/2208.08114\n"
        "- UCI Machine Learning Repository, Adult dataset, DOI 10.24432/C5XW20. https://archive.ics.uci.edu/dataset/2/adult\n",
        encoding="utf-8")
    return figures


def mia_check(config_path="configs/default.yaml"):
    cfg = read_config(config_path)
    _validate_config(cfg)
    frame = ensure_data(cfg)
    seed = int(cfg["training"]["seeds"][0])
    frame = frame.sample(min(len(frame), int(cfg["training"]["smoke_rows"])), random_state=seed).reset_index(drop=True)
    rows = run_one(frame, cfg, seed, 0.0, cfg["attack"].get("smoke_budgets", [1, 2])[:1],
                   int(cfg["training"]["smoke_epochs"]))
    out = Path(cfg["paths"].get("mia_feasibility", "results/tables/mia_feasibility.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


def run_baseline(config_path="configs/default.yaml", mode="full"):
    cfg = read_config(config_path)
    _validate_config(cfg)
    frame = ensure_data(cfg)
    if mode == "smoke":
        seed = int(cfg["training"]["seeds"][0])
        frame = frame.sample(min(len(frame), int(cfg["training"]["smoke_rows"])), random_state=seed).reset_index(drop=True)
        seeds, budgets, epochs = [seed], cfg["attack"].get("smoke_budgets", [1, 2]), int(cfg["training"]["smoke_epochs"])
    else:
        seeds, budgets, epochs = cfg["training"]["seeds"], cfg["attack"]["budgets"], int(cfg["training"]["epochs"])
    rows = []
    for seed in seeds:
        rows.extend(run_one(frame, cfg, int(seed), 0.0, budgets, epochs))
    dest = Path(cfg["paths"].get("baseline_results", "results/raw/baseline.csv"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    return pd.DataFrame(rows)
