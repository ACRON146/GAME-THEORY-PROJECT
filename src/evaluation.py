from __future__ import annotations

import json

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _downstream(train_x, train_y, test_x, test_y):
    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=int)
    test_x = np.asarray(test_x, dtype=float)
    test_y = np.asarray(test_y, dtype=int)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("Downstream features must be 2D arrays with matching columns.")
    if len(np.unique(test_y)) != 2:
        raise ValueError("Downstream utility requires both target classes in held-out data.")
    if len(np.unique(train_y)) == 1:
        constant = int(train_y[0])
        pred = np.full(len(test_y), constant, dtype=int)
        return {"accuracy": float(accuracy_score(test_y, pred)),
                "f1": float(f1_score(test_y, pred, zero_division=0)),
                "roc_auc": float("nan")}
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=0)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    prob = model.predict_proba(test_x)[:, 1]
    return {"accuracy": float(accuracy_score(test_y, pred)),
            "f1": float(f1_score(test_y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_y, prob))}


def _correlation(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("Correlation similarity requires at least two rows and one feature.")
    x = np.nan_to_num(x)
    sd = x.std(axis=0)
    z = np.divide(x - x.mean(axis=0), sd, out=np.zeros_like(x), where=sd > 1e-12)
    return (z.T @ z) / max(len(x), 1)


def utility_metrics(member_x, member_y, test_x, test_y, synthetic_x, synthetic_y, cat_sizes,
                    feature_names=None):
    """Return separate held-out utility and feature distribution metrics."""
    member_x = np.asarray(member_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    synthetic_x = np.asarray(synthetic_x, dtype=float)
    synthetic_y = np.asarray(synthetic_y, dtype=int)
    if member_x.ndim != 2 or test_x.ndim != 2 or synthetic_x.ndim != 2:
        raise ValueError("Utility feature matrices must be two-dimensional.")
    if member_x.shape[1] != test_x.shape[1] or member_x.shape[1] != synthetic_x.shape[1]:
        raise ValueError("Real and synthetic feature matrices must have matching columns.")
    cat_sizes = [int(size) for size in cat_sizes]
    if any(size <= 1 for size in cat_sizes) or sum(cat_sizes) > member_x.shape[1]:
        raise ValueError("cat_sizes must describe valid categorical one-hot columns.")
    if len(synthetic_y) != len(synthetic_x):
        raise ValueError("synthetic_y must have one label for each synthetic row.")

    real = _downstream(member_x, member_y, test_x, test_y)
    syn = _downstream(synthetic_x, synthetic_y, test_x, test_y)
    # Q is downstream F1 on held-out real data when trained on synthetic data.
    utility_q = max(0.0, min(1.0, syn["f1"]))
    f1_similarity = max(0.0, min(1.0, 1.0 - abs(real["f1"] - syn["f1"])))

    corr_real = _correlation(member_x)
    corr_syn = _correlation(synthetic_x)
    correlation_similarity = max(0.0, min(1.0, 1.0 - float(np.mean(np.abs(corr_real - corr_syn))) / 2.0))
    offset, cat_scores, per_column = 0, [], {}
    names = list(feature_names or [f"feature_{i}" for i in range(member_x.shape[1])])
    if len(names) != member_x.shape[1]:
        raise ValueError("feature_names must match the number of encoded input columns.")
    for group_index, size in enumerate(cat_sizes):
        real_prob = member_x[:, offset:offset + size].mean(axis=0)
        synthetic_prob = synthetic_x[:, offset:offset + size].mean(axis=0)
        tv = 0.5 * float(np.abs(real_prob - synthetic_prob).sum())
        similarity = max(0.0, min(1.0, 1.0 - tv))
        cat_scores.append(similarity)
        per_column[f"categorical_group_{group_index}"] = similarity
        offset += size
    categorical_similarity = float(np.mean(cat_scores)) if cat_scores else 1.0

    numeric_scores = []
    for column_index in range(offset, member_x.shape[1]):
        ks = float(ks_2samp(member_x[:, column_index], synthetic_x[:, column_index]).statistic)
        similarity = max(0.0, min(1.0, 1.0 - ks))
        numeric_scores.append(similarity)
        per_column[names[column_index]] = similarity
    numerical_similarity = float(np.mean(numeric_scores)) if numeric_scores else 1.0
    distribution_similarity = float(np.mean(list(per_column.values()))) if per_column else 1.0
    return {
        "real_test_accuracy": real["accuracy"],
        "real_test_f1": real["f1"],
        "real_test_auc": real["roc_auc"],
        "synthetic_test_accuracy": syn["accuracy"],
        "synthetic_test_f1": syn["f1"],
        "synthetic_test_auc": syn["roc_auc"],
        "utility_q": utility_q,
        "real_synthetic_f1_similarity": f1_similarity,
        "per_column_distribution_similarity": distribution_similarity,
        "per_column_similarity_json": json.dumps(per_column, sort_keys=True),
        "correlation_similarity": correlation_similarity,
        "categorical_similarity": categorical_similarity,
        "numerical_similarity": numerical_similarity,
    }
