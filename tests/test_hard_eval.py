"""Tests for the stricter evaluation helpers."""
from __future__ import annotations

import numpy as np

from src.evaluate_hard import add_wrong_symptoms, near_copy_mask


def test_near_copy_mask_flags_subsets_and_supersets_of_same_disease_only() -> None:
    X_train = np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=np.float32)
    y_train = np.array([0, 1])
    X_test = np.array([
        [1, 1, 0, 0],   # subset of train row 0, same disease  -> near copy
        [1, 1, 1, 1],   # superset of train row 0, same disease -> near copy
        [0, 1, 0, 1],   # overlaps but is neither -> novel
        [1, 1, 0, 0],   # same vector as the first but a different disease -> novel
    ], dtype=np.float32)
    y_test = np.array([0, 0, 1, 1])
    assert near_copy_mask(X_test, y_test, X_train, y_train).tolist() == [True, True, False, False]


def test_add_wrong_symptoms_adds_exactly_k_new_symptoms() -> None:
    rng = np.random.default_rng(0)
    X = np.zeros((6, 12), dtype=np.float32)
    X[:, :3] = 1.0
    out = add_wrong_symptoms(X, np.zeros(6), 2, rng)
    assert (out.sum(axis=1) == 5).all()
    assert (out[:, :3] == 1).all()                          # real symptoms are kept
    assert np.array_equal(add_wrong_symptoms(X, np.zeros(6), 0, rng), X)
