"""Uncertainty estimation: Monte-Carlo Dropout plus the "model is unsure" rules.

Monte-Carlo Dropout in one paragraph: we normally switch dropout OFF at
prediction time. If we leave it ON and predict 30 times, each pass uses a
slightly different sub-network. If the 30 answers agree, the model is stable;
if they scatter, the model is unsure. The spread gives a confidence RANGE.

Important implementation detail: only the Dropout layers are put in train mode.
BatchNorm must stay in eval mode, otherwise a batch of one patient would be
normalised with its own statistics and the predictions would be garbage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src import config
from src.models import MLP


def entropy(p: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy (bits by default) of a probability vector."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum() / np.log(base))


def enable_mc_dropout(model: MLP) -> None:
    """Eval mode everywhere, then switch ONLY dropout layers back to train mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_samples(
    model: MLP, X: np.ndarray, passes: int = config.MC_DROPOUT_PASSES,
    temperature: float = 1.0, seed: int = config.SEED,
) -> np.ndarray:
    """Return probabilities of shape ``(passes, n_samples, n_classes)``.

    Uses a forked RNG so the same patient always gets the same range (stable UI)
    without disturbing the global random state.
    """
    X_t = torch.as_tensor(np.asarray(X, dtype=np.float32))
    out = []
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        enable_mc_dropout(model)
        for _ in range(passes):
            out.append(torch.softmax(model(X_t) / temperature, dim=1).numpy())
    model.eval()  # always leave the model in normal inference mode
    return np.stack(out)


@dataclass
class MCSummary:
    """Summary statistics of MC-dropout samples for ONE patient."""

    mean: np.ndarray  # (n_classes,) mean probability
    low: np.ndarray  # (n_classes,) lower percentile
    high: np.ndarray  # (n_classes,) upper percentile
    predictive_entropy: float  # total uncertainty (bits)
    mutual_information: float  # model (epistemic) uncertainty: entropy of the mean minus mean entropy


def summarize_mc(samples: np.ndarray, percentiles: tuple[float, float] = config.MC_RANGE_PERCENTILES) -> MCSummary:
    """Collapse ``(passes, 1, C)`` or ``(passes, C)`` samples into an :class:`MCSummary`."""
    s = samples.reshape(samples.shape[0], -1)
    mean = s.mean(0)
    mi = entropy(mean) - float(np.mean([entropy(row) for row in s]))
    return MCSummary(
        mean=mean,
        low=np.percentile(s, percentiles[0], axis=0),
        high=np.percentile(s, percentiles[1], axis=0),
        predictive_entropy=entropy(mean),
        mutual_information=max(mi, 0.0),
    )


@dataclass
class ConfidenceAssessment:
    """Outcome of the 'is the model sure enough to show a confident answer?' check."""

    is_uncertain: bool
    reasons: list[str]
    message: str | None


def assess_confidence(
    probs: np.ndarray,
    n_symptoms: int,
    conf_threshold: float = config.CONFIDENCE_THRESHOLD,
    gap_threshold: float = config.TOP2_GAP_THRESHOLD,
    min_symptoms: int = config.MIN_SYMPTOMS_FOR_CONFIDENCE,
) -> ConfidenceAssessment:
    """Decide whether to show 'The model is unsure' instead of a confident answer.

    Triggers: top-1 probability below ``conf_threshold``; top-1 minus top-2 below
    ``gap_threshold``; or fewer than ``min_symptoms`` reported symptoms (a very
    short symptom list cannot support a confident answer).
    """
    order = np.argsort(-probs)
    p1, p2 = float(probs[order[0]]), float(probs[order[1]]) if len(probs) > 1 else 0.0
    reasons: list[str] = []
    if p1 < conf_threshold:
        reasons.append(f"The top prediction has only {p1:.0%} probability (threshold {conf_threshold:.0%}).")
    if p1 - p2 < gap_threshold:
        reasons.append(f"The top two diseases are very close ({p1:.0%} vs {p2:.0%}).")
    if n_symptoms < min_symptoms:
        reasons.append(f"Only {n_symptoms} symptom(s) were provided; at least {min_symptoms} are needed for a confident answer.")
    return ConfidenceAssessment(bool(reasons), reasons, config.UNSURE_MESSAGE if reasons else None)
