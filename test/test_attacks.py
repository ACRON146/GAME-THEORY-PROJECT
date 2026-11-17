import numpy as np
import pytest

from src.attacks import _mixed_pairwise_distances, mia_distance_attack, mia_shadow_attack


def test_distance_attack_separates_near_members_and_reports_chance_baseline():
    releases = [np.array([[0.0], [1.0]])]
    members = np.array([[0.0], [1.0]])
    nonmembers = np.array([[10.0], [11.0]])
    result = mia_distance_attack(members, nonmembers, releases, seed=4, bootstrap=20)
    assert result["mia_auc"] == pytest.approx(1.0)
    assert result["mia_advantage"] == pytest.approx(1.0)
    assert result["random_auc_baseline"] == .5
    assert result["random_advantage_baseline"] == 0
    assert result["mia_auc_ci_low"] == pytest.approx(1.0)


def test_shadow_attack_fits_classifier_on_shadow_records_only():
    shadow_members = np.array([[0.], [1.], [2.], [3.]])
    shadow_nonmembers = np.array([[10.], [11.], [12.], [13.]])
    releases = [shadow_members.copy()]
    target_members = np.array([[.1], [1.1], [2.1]])
    target_nonmembers = np.array([[20.], [21.], [22.]])
    result = mia_shadow_attack(target_members, target_nonmembers, releases,
                               [(shadow_members, shadow_nonmembers, releases)],
                               seed=8, bootstrap=20)
    assert result["mia_auc"] > .99
    assert result["mia_advantage"] > .99
    assert result["mia_accuracy"] == 1.0


def test_mixed_distance_counts_categorical_mismatch_as_one_field():
    left = np.array([[1., 0., .0]])
    right = np.array([[0., 1., 1.]])
    # One categorical mismatch and one numeric unit difference, averaged over two fields.
    assert _mixed_pairwise_distances(left, right, [2])[0, 0] == pytest.approx(1.)


def test_attack_rejects_invalid_release_shapes():
    with pytest.raises(ValueError):
        mia_distance_attack(np.array([[0.]]), np.array([[1.]]), [np.array([[0., 1.]])])
    with pytest.raises(ValueError):
        mia_distance_attack(np.empty((0, 1)), np.array([[1.]]), [np.array([[0.]])])
