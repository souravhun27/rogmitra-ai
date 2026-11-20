"""Inference side of the optional free-text model (DistilBERT fine-tuned on Symptom2Disease).

Kept apart from training so the Streamlit app can import it cheaply. Everything
is lazy: if ``transformers`` is not installed or no model was trained,
``text_model_available()`` simply returns False and the app hides the feature.

The text model is a SEPARATE predictor with its own (smaller, 24-disease) label
set. ``match_structured`` maps its labels onto the structured model's classes so
the app can show whether the two models agree.
"""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import numpy as np

from src import config
from src.models import softmax
from src.preprocess import clean_disease
from src.utils import load_json


def text_model_available(model_dir: Path = config.TEXT_MODEL_DIR) -> bool:
    """True when a trained text model exists and transformers + torch can be imported."""
    return (
        (model_dir / "explaindx_text_meta.json").exists()
        and importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("torch") is not None
    )


def match_structured(label: str, classes: list[str]) -> str | None:
    """Map a text-dataset label (e.g. 'Common Cold') to a structured class name, or None."""
    lookup = {c.lower(): c for c in classes}
    return lookup.get(clean_disease(label).lower())


class TextPredictor:
    """Loads the fine-tuned model once and predicts diseases from free text."""

    def __init__(self, model_dir: Path = config.TEXT_MODEL_DIR) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._lock = threading.Lock()
        self.meta = load_json(model_dir / "explaindx_text_meta.json")
        self.labels: list[str] = self.meta["labels"]
        self.temperature: float = float(self.meta["temperature"])
        self.max_len: int = int(self.meta["max_len"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    def predict_proba(self, text: str) -> np.ndarray:
        """Calibrated probabilities over the text model's labels."""
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        with self._lock, self._torch.no_grad():
            logits = self.model(**enc).logits.numpy()
        return softmax(logits, self.temperature)[0]

    def predict(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Top-k ``(label, probability)`` pairs for a free-text description."""
        probs = self.predict_proba(text)
        order = np.argsort(-probs)[:top_k]
        return [(self.labels[i], float(probs[i])) for i in order]
