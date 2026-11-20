"""Tests for model output shape, probabilities, calibration maths and uncertainty."""
from __future__ import annotations

import numpy as np
import torch

from src.models import build_mlp, mlp_logits, mlp_proba, softmax
from src.uncertainty import assess_confidence, entropy, mc_dropout_samples, summarize_mc


def test_mlp_output_shape_and_probabilities_sum_to_one() -> None:
    torch.manual_seed(0)
    model = build_mlp(131, 41)
    X = np.random.default_rng(0).integers(0, 2, size=(7, 131)).astype(np.float32)
    proba = mlp_proba(model, X)
    assert proba.shape == (7, 41)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba >= 0).all()


def test_eval_mode_prediction_is_deterministic() -> None:
    torch.manual_seed(0)
    model = build_mlp(20, 5)
    X = np.ones((3, 20), dtype=np.float32)
    assert np.allclose(mlp_logits(model, X), mlp_logits(model, X))


def test_higher_temperature_softens_probabilities() -> None:
    logits = np.array([[4.0, 1.0, 0.0]])
    assert softmax(logits, 0.5).max() > softmax(logits, 1.0).max() > softmax(logits, 3.0).max()
    assert np.allclose(softmax(logits, 2.0).sum(), 1.0)


def test_mc_dropout_returns_expected_shape_and_leaves_model_in_eval_mode() -> None:
    torch.manual_seed(0)
    model = build_mlp(20, 5)
    X = np.ones((2, 20), dtype=np.float32)
    samples = mc_dropout_samples(model, X, passes=30)
    assert samples.shape == (30, 2, 5)
    assert not model.training
    assert np.allclose(samples.sum(axis=2), 1.0, atol=1e-5)
    summary = summarize_mc(samples[:, 0, :])  # one patient: (passes, classes)
    assert summary.mean.shape == (5,)
    assert (summary.low <= summary.mean + 1e-9).all() and (summary.mean <= summary.high + 1e-9).all()


def test_mc_dropout_is_repeatable_for_the_same_seed() -> None:
    torch.manual_seed(0)
    model = build_mlp(20, 5)
    X = np.ones((1, 20), dtype=np.float32)
    assert np.allclose(mc_dropout_samples(model, X), mc_dropout_samples(model, X))


def test_entropy_is_zero_for_certain_and_maximal_for_uniform() -> None:
    assert entropy(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0
    assert abs(entropy(np.full(4, 0.25)) - 2.0) < 1e-9  # log2(4) bits


def test_assess_confidence_flags_low_probability_close_gap_and_few_symptoms() -> None:
    confident = assess_confidence(np.array([0.9, 0.05, 0.05]), n_symptoms=5)
    assert not confident.is_uncertain and confident.message is None
    assert assess_confidence(np.array([0.4, 0.3, 0.3]), n_symptoms=5).is_uncertain      # low top-1
    assert assess_confidence(np.array([0.52, 0.45, 0.03]), n_symptoms=5).is_uncertain    # small gap
    assert assess_confidence(np.array([0.9, 0.05, 0.05]), n_symptoms=1).is_uncertain     # too few symptoms
