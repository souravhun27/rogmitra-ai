"""Tests for cleaning, de-duplication, splitting, encoding and augmentation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocess import (
    build_symptom_matrix, clean_symptom, count_subset_rows, deduplicate, encode_symptoms,
    random_symptom_dropout, remove_fraction, stratified_split,
)


def test_clean_symptom_normalises_whitespace_case_and_underscores() -> None:
    assert clean_symptom(" dischromic _patches") == "discolored patches"  # typo dictionary applied
    assert clean_symptom("High_Fever") == "high fever"
    assert clean_symptom("  skin   rash ") == "skin rash"


def test_build_symptom_matrix_is_binary_sorted_and_collapses_repeats() -> None:
    df = pd.DataFrame({
        "Disease": ["Flu", "Malaria"],
        "Symptom_1": ["fever", "chills"],
        "Symptom_2": [" cough", "fever"],
        "Symptom_3": [None, "fever_"],  # the same symptom twice must count once
    })
    X, y, vocab = build_symptom_matrix(df)
    assert vocab == sorted(vocab)
    assert set(np.unique(X)) <= {0, 1}
    assert X.shape == (2, len(vocab))
    assert X[1, vocab.index("fever")] == 1
    assert list(y) == ["Flu", "Malaria"]


def test_deduplicate_removes_exact_duplicates_only_within_same_disease() -> None:
    X = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.uint8)
    y = np.array(["A", "A", "B", "B"])
    Xu, yu, report = deduplicate(X, y)
    assert report.n_raw == 4
    assert report.n_exact_duplicates == 1  # only the repeated (row, "A")
    assert len(yu) == 3
    assert report.n_conflicting_vectors == 1  # [1, 0] appears under A and B


def test_deduplicate_can_drop_conflicting_rows() -> None:
    X = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.uint8)
    y = np.array(["A", "B", "B"])
    _, yu, report = deduplicate(X, y, drop_conflicts=True)
    assert len(yu) == 1 and report.n_conflict_rows_dropped == 2


def test_count_subset_rows_detects_near_copies() -> None:
    X = np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.uint8)
    y = np.array(["A", "A", "B"])
    assert count_subset_rows(X, y) == 1  # row 0 is a subset of row 1 (same disease)


def test_stratified_split_has_no_overlap_and_keeps_every_class() -> None:
    labels = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 10)
    tr, va, te = stratified_split(labels, seed=config.SEED)
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == len(labels)
    for split in (tr, va, te):
        assert set(labels[split]) == {"a", "b", "c"}


def test_stratified_split_is_reproducible_with_fixed_seed() -> None:
    labels = np.array(["a"] * 15 + ["b"] * 15)
    first = stratified_split(labels, seed=1)
    second = stratified_split(labels, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_stratified_split_rejects_fractions_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        stratified_split(np.array(["a"] * 10), 0.5, 0.2, 0.2)


def test_encode_symptoms_reports_unknown_names() -> None:
    vec, unknown = encode_symptoms(["Fever", "not_a_symptom"], ["cough", "fever"])
    assert vec.tolist() == [0.0, 1.0]
    assert unknown == ["not_a_symptom"]


def test_random_symptom_dropout_drops_but_never_empties_a_row() -> None:
    rng = np.random.default_rng(0)
    X = np.ones((200, 30), dtype=np.float32)
    out = random_symptom_dropout(X, rng)
    assert out.sum() < X.sum()
    assert (out.sum(axis=1) >= 1).all()
    assert set(np.unique(out)) <= {0.0, 1.0}


def test_remove_fraction_removes_exact_share_and_keeps_one_symptom() -> None:
    rng = np.random.default_rng(0)
    X = np.zeros((5, 10), dtype=np.float32)
    X[:, :4] = 1.0
    out = remove_fraction(X, 0.5, rng)
    assert (out.sum(axis=1) == 2).all()
    assert (remove_fraction(X, 1.0, rng).sum(axis=1) == 1).all()
