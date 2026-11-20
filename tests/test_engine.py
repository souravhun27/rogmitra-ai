"""End-to-end test of the real trained model. Skipped automatically if training has not been run."""
from __future__ import annotations

import pytest

from src import config


@pytest.fixture()
def engine(trained_files_exist: bool):
    if not trained_files_exist:
        pytest.skip("trained model not found; run the training scripts first")
    from src.engine import DiagnosisEngine

    return DiagnosisEngine()


def test_analyze_returns_three_calibrated_probabilities_in_order(engine) -> None:
    picks = engine.symptoms[:4]
    result = engine.analyze(picks)
    assert len(result.top) == config.TOP_K_DISPLAY
    probs = [d.prob for d in result.top]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= d.low <= d.prob <= d.high <= 1.0 for d in result.top)
    assert result.explanation_text and result.counterfactuals is not None


def test_empty_input_raises_a_clear_error(engine) -> None:
    with pytest.raises(ValueError):
        engine.analyze([])


def test_red_flag_overrides_even_without_a_model_symptom(engine) -> None:
    result = engine.analyze([], free_text="severe chest pain")
    assert result.emergency and result.top == []


def test_single_symptom_is_flagged_as_unsure(engine) -> None:
    assert engine.analyze(engine.symptoms[:1]).is_uncertain
