"""Follow-up question suggester: which symptom should we ask about next?

Idea: among the top candidate diseases, find the symptom whose answer would
reduce our uncertainty the most. For a symptom s that the user has NOT reported:

* P(yes) = sum over candidate diseases d of  P(d) * P(s | d), where P(s | d) is
  the share of that disease's dataset rows containing s.
* If the answer is "yes" we re-run the model with s added and measure the
  entropy of the (renormalised) candidate distribution.
* If the answer is "no" the input is unchanged (the model has no way to encode
  an explicit "absent"), so nothing is gained.

Expected entropy reduction = P(yes) * (H_before - H_after_yes), in bits. We rank
symptoms by it. Limitation: P(s | d) comes from the dataset, not real-world
prevalence.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from src import config
from src.uncertainty import entropy

PredictFn = Callable[[np.ndarray], np.ndarray]


def _renormalise(p: np.ndarray) -> np.ndarray:
    total = p.sum(axis=-1, keepdims=True)
    return p / np.maximum(total, 1e-12)


def suggest_followups(
    predict_fn: PredictFn,
    x: np.ndarray,
    symptoms: list[str],
    classes: list[str],
    freq: np.ndarray,
    k: int = config.FOLLOWUP_TOP_N,
    n_candidates: int = config.FOLLOWUP_CANDIDATES,
    exclude: Iterable[str] = (),
) -> list[dict]:
    """Return up to ``k`` suggested symptoms, best first.

    ``freq`` is a ``(n_classes, n_symptoms)`` matrix of P(symptom | disease).
    Each result has: ``symptom``, ``expected_entropy_reduction`` (bits),
    ``p_yes``, ``top_if_yes`` (disease that would lead) and ``favours`` (the
    candidate diseases where the symptom is most typical, with frequencies).
    """
    excluded = set(exclude)
    p0 = predict_fn(x[None])[0]
    cand = np.argsort(-p0)[:n_candidates]
    q0 = _renormalise(p0[cand])
    h0 = entropy(q0)

    options = [
        j for j in range(len(symptoms))
        if x[j] == 0 and symptoms[j] not in excluded and freq[cand, j].max() > 0
    ]
    if not options:
        return []
    X_yes = np.repeat(x[None], len(options), axis=0)
    X_yes[np.arange(len(options)), options] = 1.0
    q_yes = _renormalise(predict_fn(X_yes)[:, cand])

    results: list[dict] = []
    for row, j in enumerate(options):
        p_yes = float(np.clip((q0 * freq[cand, j]).sum(), 0.0, 1.0))
        gain = p_yes * (h0 - entropy(q_yes[row]))
        order = np.argsort(-freq[cand, j])[:2]
        results.append(
            {
                "symptom": symptoms[j],
                "expected_entropy_reduction": float(gain),
                "p_yes": p_yes,
                "top_if_yes": classes[cand[int(np.argmax(q_yes[row]))]],
                "favours": [(classes[cand[o]], float(freq[cand[o], j])) for o in order if freq[cand[o], j] > 0],
            }
        )
    results.sort(key=lambda r: -r["expected_entropy_reduction"])
    return results[:k]
