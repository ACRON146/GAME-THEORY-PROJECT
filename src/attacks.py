from __future__ import annotations

import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _validate_matrix(values, name: str, columns: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or (columns is not None and array.shape[1] != columns):
        raise ValueError(f"{name} must be a 2D matrix with {columns or 'a consistent number of'} columns.")
    if array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values and at least one feature.")
    return array


def _validate_cat_sizes(cat_sizes, width: int) -> list[int]:
    sizes = [int(size) for size in (cat_sizes or [])]
    if any(size <= 1 for size in sizes) or sum(sizes) > width:
        raise ValueError("cat_sizes must describe valid one-hot groups within the feature width.")
    return sizes


def _mixed_pairwise_distances(left: np.ndarray, right: np.ndarray, cat_sizes: list[int]) -> np.ndarray:
    """RMS distance over logical fields: categorical mismatch + scaled numeric."""
    n_fields = len(cat_sizes) + (left.shape[1] - sum(cat_sizes))
    out = np.zeros((len(left), len(right)), dtype=np.float64)
    offset = 0
    for size in cat_sizes:
        lcat = np.argmax(left[:, offset:offset + size], axis=1)
        rcat = np.argmax(right[:, offset:offset + size], axis=1)
        out += (lcat[:, None] != rcat[None, :])
        offset += size
    if offset < left.shape[1]:
        # Numeric columns are min-max scaled by the train-fitted encoder.
        for j in range(offset, left.shape[1]):
            out += (left[:, j, None] - right[None, :, j]) ** 2
    return np.sqrt(out / max(n_fields, 1))


def _distance_features(candidates, release_batches, cat_sizes=(), max_candidates=1000, chunk_size=128):
    if int(max_candidates) <= 0 or int(chunk_size) <= 0:
        raise ValueError("max_candidates and chunk_size must be positive.")
    candidates = _validate_matrix(candidates, "candidate records")
    releases = [_validate_matrix(batch, "release batch", candidates.shape[1]) for batch in release_batches]
    if not releases or sum(map(len, releases)) == 0:
        raise ValueError("At least one non-empty synthetic release batch is required.")
    released = np.concatenate(releases, axis=0)
    groups = _validate_cat_sizes(cat_sizes, candidates.shape[1])
    candidates = candidates[:int(max_candidates)]
    if len(candidates) == 0:
        raise ValueError("At least one candidate record is required.")
    feature_rows = []
    for start in range(0, len(candidates), int(chunk_size)):
        distances = _mixed_pairwise_distances(candidates[start:start + chunk_size], released, groups)
        k = min(5, distances.shape[1])
        nearest_k = np.partition(distances, k - 1, axis=1)[:, :k]
        # Lower values mean closer to a released record.
        feature_rows.append(np.column_stack([
            distances.min(axis=1),
            nearest_k.mean(axis=1),
            np.median(nearest_k, axis=1),
        ]))
    return np.concatenate(feature_rows, axis=0)


def _advantage(labels, scores) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(tpr - fpr))


def _metric_summary(member_scores, nonmember_scores, *, seed=11, bootstrap=100) -> dict[str, float]:
    if len(member_scores) == 0 or len(nonmember_scores) == 0 or int(bootstrap) <= 0:
        raise ValueError("Attack metrics require non-empty classes and positive bootstrap repetitions.")
    labels = np.r_[np.ones(len(member_scores), dtype=int), np.zeros(len(nonmember_scores), dtype=int)]
    scores = np.r_[member_scores, nonmember_scores].astype(float)
    auc = float(roc_auc_score(labels, scores))
    advantage = _advantage(labels, scores)
    rng = np.random.default_rng(int(seed))
    auc_samples, advantage_samples = [], []
    for _ in range(int(bootstrap)):
        m = rng.choice(member_scores, size=len(member_scores), replace=True)
        n = rng.choice(nonmember_scores, size=len(nonmember_scores), replace=True)
        boot_scores = np.r_[m, n]
        boot_labels = np.r_[np.ones(len(m), dtype=int), np.zeros(len(n), dtype=int)]
        auc_samples.append(roc_auc_score(boot_labels, boot_scores))
        advantage_samples.append(_advantage(boot_labels, boot_scores))
    return {
        "mia_auc": auc,
        "mia_auc_ci_low": float(np.percentile(auc_samples, 2.5)),
        "mia_auc_ci_high": float(np.percentile(auc_samples, 97.5)),
        "mia_advantage": advantage,
        "mia_advantage_ci_low": float(np.percentile(advantage_samples, 2.5)),
        "mia_advantage_ci_high": float(np.percentile(advantage_samples, 97.5)),
        "random_auc_baseline": 0.5,
        "random_advantage_baseline": 0.0,
    }


def mia_distance_attack(member_x, nonmember_x, release_batches, max_candidates=1000,
                        cat_sizes=(), seed=11, bootstrap=100):
    """Black-box baseline: score candidates by nearest released-row distance.

    Candidate labels are used only to calculate final ROC metrics. No attack
    threshold is selected using the evaluation candidate labels.
    """
    start = time.perf_counter()
    member_x = _validate_matrix(member_x, "member candidates")
    nonmember_x = _validate_matrix(nonmember_x, "non-member candidates", member_x.shape[1])
    member_features = _distance_features(member_x, release_batches, cat_sizes, max_candidates)
    nonmember_features = _distance_features(nonmember_x, release_batches, cat_sizes, max_candidates)
    result = _metric_summary(-member_features[:, 0], -nonmember_features[:, 0], seed=seed, bootstrap=bootstrap)
    result["attack_seconds"] = float(time.perf_counter() - start)
    result["attack_name"] = "nearest-release-row"
    return result


def mia_shadow_attack(member_x, nonmember_x, target_releases, shadow_sets,
                      max_candidates=1000, cat_sizes=(), seed=11, bootstrap=100):
    """Shadow-model MIA trained only on shadow-model membership labels.

    Each shadow set is ``(shadow_members, shadow_nonmembers, release_batches)``.
    The target member/non-member candidates are held apart from the records used
    to fit the shadow attack. The classifier and its 0.5 threshold are fitted on
    shadow data; target labels are used only for final metrics.
    """
    start = time.perf_counter()
    member_x = _validate_matrix(member_x, "target member candidates")
    nonmember_x = _validate_matrix(nonmember_x, "target non-member candidates", member_x.shape[1])
    if not shadow_sets:
        raise ValueError("At least one shadow model is required for the shadow attack.")
    attack_features, attack_labels = [], []
    for shadow_member, shadow_nonmember, shadow_releases in shadow_sets:
        positive = _distance_features(shadow_member, shadow_releases, cat_sizes, max_candidates)
        negative = _distance_features(shadow_nonmember, shadow_releases, cat_sizes, max_candidates)
        attack_features.extend([positive, negative])
        attack_labels.extend([np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)])
    train_x = np.vstack(attack_features)
    train_y = np.concatenate(attack_labels)
    if len(np.unique(train_y)) != 2:
        raise ValueError("Shadow attack training data must contain both membership classes.")
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=int(seed)))
    classifier.fit(train_x, train_y)

    target_member_features = _distance_features(member_x, target_releases, cat_sizes, max_candidates)
    target_nonmember_features = _distance_features(nonmember_x, target_releases, cat_sizes, max_candidates)
    member_scores = classifier.predict_proba(target_member_features)[:, 1]
    nonmember_scores = classifier.predict_proba(target_nonmember_features)[:, 1]
    result = _metric_summary(member_scores, nonmember_scores, seed=seed + 1, bootstrap=bootstrap)
    y = np.r_[np.ones(len(member_scores), dtype=int), np.zeros(len(nonmember_scores), dtype=int)]
    predicted = classifier.predict(np.vstack([target_member_features, target_nonmember_features]))
    result.update({
        "mia_accuracy": float(accuracy_score(y, predicted)),
        "mia_precision": float(precision_score(y, predicted, zero_division=0)),
        "mia_recall": float(recall_score(y, predicted, zero_division=0)),
        "attack_seconds": float(time.perf_counter() - start),
        "attack_name": "shadow-model-logistic",
    })
    return result
