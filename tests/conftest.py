"""Shared pytest fixtures. Tests use small synthetic data, so they run without the Kaggle files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``import src...`` work when pytest is started from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


@pytest.fixture()
def toy_world() -> dict:
    """A tiny 4-disease, 8-symptom world with hand-made, well-separated patterns."""
    symptoms = ["fever", "chills", "cough", "headache", "rash", "itching", "vomiting", "fatigue"]
    classes = ["Flu", "Malaria", "Allergy", "Migraine"]
    patterns = {  # disease -> symptom indices that define it
        0: [0, 2, 7],
        1: [0, 1, 6],
        2: [4, 5],
        3: [3, 6],
    }
    X = np.zeros((len(classes), len(symptoms)), dtype=np.float32)
    for c, idx in patterns.items():
        X[c, idx] = 1.0

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        """Deterministic stand-in model: softmax of overlap with each disease pattern."""
        batch = np.atleast_2d(batch)
        scores = batch @ X.T * 3.0
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    freq = X.copy() * 0.9  # P(symptom | disease) for the follow-up suggester
    return {"symptoms": symptoms, "classes": classes, "patterns": X, "predict": predict_fn, "freq": freq}


@pytest.fixture()
def trained_files_exist() -> bool:
    """True when the real trained model and processed data are present on disk."""
    needed = [config.MODELS_DIR / "mlp.pt", config.SYMPTOMS_JSON, config.CLASSES_JSON, config.DISEASE_INFO_JSON,
              config.DISEASE_SYMPTOM_MAP_JSON]
    return all(p.exists() for p in needed)
