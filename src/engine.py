"""DiagnosisEngine: one call that produces everything the app needs.

``engine.analyze(symptoms, free_text)`` returns an :class:`AnalysisResult` with
predictions, uncertainty, red flags, explanation, counterfactuals and follow-up
questions. The Streamlit app (Stage 4) and the PDF report only talk to this
class, so all the logic stays testable without a UI.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import numpy as np

from src import config
from src.explain import (
    Contribution, counterfactual_table, integrated_gradients, plain_language_explanation, split_contributions,
)
from src.followup import suggest_followups
from src.models import load_mlp, mlp_proba
from src.preprocess import encode_symptoms, load_vocab
from src.redflags import RedFlag, check_red_flags
from src.uncertainty import assess_confidence, mc_dropout_samples, summarize_mc
from src.utils import load_json

logger = logging.getLogger(__name__)


@dataclass
class DiseaseScore:
    """A disease with its calibrated probability and MC-dropout range."""

    disease: str
    prob: float
    low: float
    high: float


@dataclass
class AnalysisResult:
    """Everything shown on the Results page."""

    selected: list[str]
    unknown: list[str]
    top: list[DiseaseScore]
    entropy: float = 0.0
    mutual_information: float = 0.0
    is_uncertain: bool = False
    uncertainty_reasons: list[str] = field(default_factory=list)
    uncertainty_message: str | None = None
    red_flags: list[RedFlag] = field(default_factory=list)
    supports: list[Contribution] = field(default_factory=list)
    against: list[Contribution] = field(default_factory=list)
    explanation_text: str = ""
    explanation_footnote: str = ""
    counterfactuals: list[dict] = field(default_factory=list)
    followups: list[dict] = field(default_factory=list)
    description: str = ""
    precautions: list[str] = field(default_factory=list)

    @property
    def emergency(self) -> bool:
        """True when any red flag fired; the UI must then show the emergency banner first."""
        return bool(self.red_flags)


class DiagnosisEngine:
    """Loads the trained model and reference data once, then answers requests."""

    def __init__(self) -> None:
        # Streamlit serves sessions from several threads that share ONE model. MC-dropout briefly
        # switches dropout on, so every use of the model is serialised with this lock.
        self._lock = threading.RLock()
        self.symptoms, self.classes = load_vocab()
        self.model, self.temperature, self.train_info = load_mlp()
        self.disease_info: dict = load_json(config.DISEASE_INFO_JSON)
        known = load_json(config.DISEASE_SYMPTOM_MAP_JSON)
        s_index = {s: i for i, s in enumerate(self.symptoms)}
        self.freq = np.zeros((len(self.classes), len(self.symptoms)))
        for c, disease in enumerate(self.classes):
            for s, f in known.get(disease, {}).items():
                self.freq[c, s_index[s]] = f

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated, deterministic probabilities for rows of ``X``."""
        with self._lock:
            return mlp_proba(self.model, np.atleast_2d(X), self.temperature)

    def analyze(
        self, selected: list[str], free_text: str = "", top_k: int = config.TOP_K_DISPLAY, explain: bool = True,
    ) -> AnalysisResult:
        """Run the full pipeline for one patient.

        Red flags are checked FIRST and are independent of the model. If no
        symptom is selected the model is not run: we return only the red-flag
        result when there is one, otherwise raise ``ValueError``.
        """
        with self._lock:
            return self._analyze(selected, free_text, top_k, explain)

    def _analyze(self, selected: list[str], free_text: str, top_k: int, explain: bool) -> AnalysisResult:
        """Implementation of :meth:`analyze` (called with the model lock held)."""
        x, unknown = encode_symptoms(selected, self.symptoms)
        flags = check_red_flags(selected, free_text)
        names = [self.symptoms[j] for j in np.flatnonzero(x > 0)]
        if x.sum() == 0:
            if flags:
                return AnalysisResult(selected=[], unknown=unknown, top=[], red_flags=flags)
            raise ValueError("Select at least one symptom.")

        probs = self.predict_proba(x)[0]
        mc = summarize_mc(mc_dropout_samples(self.model, x[None], temperature=self.temperature))
        order = np.argsort(-probs)[:top_k]
        top = [
            DiseaseScore(
                self.classes[i], float(probs[i]),
                float(min(mc.low[i], probs[i])), float(max(mc.high[i], probs[i])),  # keep point estimate inside its range
            )
            for i in order
        ]
        conf = assess_confidence(probs, len(names))
        top_idx = int(order[0])
        result = AnalysisResult(
            selected=names, unknown=unknown, top=top, entropy=mc.predictive_entropy,
            mutual_information=mc.mutual_information, is_uncertain=conf.is_uncertain,
            uncertainty_reasons=conf.reasons, uncertainty_message=conf.message, red_flags=flags,
        )
        info = self.disease_info.get(self.classes[top_idx], {})
        result.description = info.get("description", "")
        result.precautions = info.get("precautions", [])

        if explain:
            attr = integrated_gradients(self.model, self.temperature, x, top_idx)
            result.supports, result.against = split_contributions(attr, x, self.symptoms)
            result.explanation_text, result.explanation_footnote = plain_language_explanation(
                self.classes[top_idx], result.supports, result.against)
            result.counterfactuals = counterfactual_table(self.predict_proba, x, self.symptoms, self.classes, top_idx)
            result.followups = suggest_followups(self.predict_proba, x, self.symptoms, self.classes, self.freq)
        return result
