"""Explainability: why did the model predict this disease?

Local (one patient)
    * Integrated Gradients (Captum) on the MLP's calibrated probability of the
      predicted disease. Baseline = "no symptoms". Because inputs are 0/1 and the
      baseline is all zeros, only symptoms the patient HAS can receive credit or
      blame, and the contributions add up exactly to
      ``p(patient) - p(no symptoms)`` (the "completeness" property).
    * Template-based plain-language sentence (no LLM).
    * Counterfactuals: drop each reported symptom, re-predict, report the change.

Global (whole dataset)
    * SHAP values for every model (GradientExplainer for the MLP, TreeExplainer
      for Random Forest / XGBoost, exact linear SHAP for Logistic Regression) and
      mean-|SHAP| bar plots.

Run ``python -m src.explain`` to create the global plots and one example local
explanation under ``reports/figures``.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.models import BASELINE_LABELS, BASELINE_NAMES, MLP, BaselineClassifier, load_mlp
from src.preprocess import load_split, load_vocab
from src.train_baselines import BASELINE_DIR
from src.utils import ensure_dir, set_seed, setup_logging

logger = logging.getLogger(__name__)
GREEN, RED = "#1A9850", "#D73027"
BLUE = "#0072B2"


# --------------------------------------------------------------------------- #
# Local explanations
# --------------------------------------------------------------------------- #
class _ProbWrapper(nn.Module):
    """MLP followed by temperature-scaled softmax, so attributions are in PROBABILITY units.

    Explaining probabilities (not raw logits) makes the numbers intuitive:
    +0.31 means "this symptom raised the disease probability by 31 points".
    """

    def __init__(self, model: MLP, temperature: float) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return calibrated class probabilities."""
        return torch.softmax(self.model(x) / self.temperature, dim=1)


@dataclass
class Contribution:
    """One symptom's signed effect on the probability of the explained disease."""

    symptom: str
    value: float


def integrated_gradients(
    model: MLP, temperature: float, x: np.ndarray, target: int, n_steps: int = config.IG_STEPS
) -> np.ndarray:
    """Per-symptom Integrated Gradients attribution for class ``target``.

    Returns a vector of length n_symptoms (zero for symptoms the patient lacks).
    The model must be in eval mode (dropout off, BatchNorm frozen).
    """
    from captum.attr import IntegratedGradients

    model.eval()
    wrapper = _ProbWrapper(model, temperature).eval()
    x_t = torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, -1)
    attr = IntegratedGradients(wrapper).attribute(
        x_t, baselines=torch.zeros_like(x_t), target=int(target), n_steps=n_steps, method="gausslegendre"
    )
    return attr.detach().numpy().ravel()


def split_contributions(
    attr: np.ndarray, x: np.ndarray, symptoms: list[str], eps: float = 1e-6
) -> tuple[list[Contribution], list[Contribution]]:
    """Split attributions of PRESENT symptoms into (supporting, opposing), strongest first."""
    present = np.flatnonzero(np.asarray(x) > 0)
    supports = sorted((Contribution(symptoms[j], float(attr[j])) for j in present if attr[j] > eps),
                      key=lambda c: -c.value)
    against = sorted((Contribution(symptoms[j], float(attr[j])) for j in present if attr[j] < -eps),
                     key=lambda c: c.value)
    return supports, against


def _join(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def plain_language_explanation(
    disease: str, supports: list[Contribution], against: list[Contribution],
    k_support: int = 3, k_against: int = 2,
) -> tuple[str, str]:
    """Template-based explanation; returns ``(text, footnote)``.

    Rules: name up to ``k_support`` supporting symptoms with their values, then
    up to ``k_against`` opposing ones. "slightly" is used when every opposing
    value is below 0.05 (5 percentage points).
    """
    if supports:
        parts = [f"{c.symptom} ({c.value:+.2f})" for c in supports[:k_support]]
        text = f"Your top supporting symptoms for {disease} were {_join(parts)}."
    else:
        text = f"No single reported symptom clearly supported {disease}."
    if against:
        shown = against[:k_against]
        parts = [f"{c.symptom} ({c.value:+.2f})" for c in shown]
        verb = "slightly reduced" if max(abs(c.value) for c in shown) < 0.05 else "reduced"
        listed = _join(parts)
        text += f" {listed[0].upper() + listed[1:]} {verb} its likelihood."
    footnote = (
        "Each number shows how much a symptom pushed this disease's probability up (+) or down (-), compared with "
        "reporting no symptoms at all (0.10 is roughly 10 percentage points). The numbers add up to the total change; "
        "when symptoms strongly interact, a single value can be larger than 1 in size."
    )
    return text, footnote


def plot_local_explanation(
    supports: list[Contribution], against: list[Contribution], disease: str, max_each: int = 6
) -> plt.Figure:
    """Horizontal bar chart: green bars support the disease, red bars argue against it."""
    items = supports[:max_each] + against[:max_each]
    fig, ax = plt.subplots(figsize=(6.5, max(2.2, 0.42 * len(items) + 1.2)))
    if not items:
        ax.text(0.5, 0.5, "No symptom had a measurable effect", ha="center", va="center")
        ax.axis("off")
        return fig
    items = sorted(items, key=lambda c: c.value)  # most negative at the bottom
    values = [c.value for c in items]
    bars = ax.barh([c.symptom for c in items], values, color=[GREEN if v > 0 else RED for v in values])
    span = max(abs(v) for v in values) or 1.0
    for bar, v in zip(bars, values):  # direct value labels so colour is never the only cue
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlim(min(0, min(values)) - 0.25 * span, max(0, max(values)) + 0.25 * span)
    ax.set_xlabel("Change in probability (green = supports, red = argues against)")
    ax.set_title(f"Why {disease}?", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def counterfactual_table(
    predict_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, symptoms: list[str],
    classes: list[str], target: int,
) -> list[dict]:
    """Remove each reported symptom in turn and describe what changes.

    For every symptom we report the new top disease, its probability, the
    original disease's probability before/after, and a ready-made sentence.
    Sorted by how much the original disease's probability moves.
    """
    p0 = predict_fn(x[None])[0]
    rows: list[dict] = []
    for j in np.flatnonzero(x > 0):
        name = symptoms[j]
        if x.sum() == 1:
            rows.append({"symptom": name, "new_top": None, "new_top_prob": None, "target_before": float(p0[target]),
                         "target_after": None, "delta": 0.0, "changes_prediction": False,
                         "sentence": f"If you did not have {name}, no symptoms would remain, so no prediction is possible."})
            continue
        x2 = x.copy()
        x2[j] = 0.0
        p = predict_fn(x2[None])[0]
        new = int(p.argmax())
        before, after = float(p0[target]), float(p[target])
        if new != target:
            sentence = (f"If you did not have {name}, the top prediction would change to {classes[new]} "
                        f"({p[new]:.0%}); {classes[target]} would fall from {before:.0%} to {after:.0%}.")
        else:
            if abs(after - before) < 0.005:  # avoid "would rise from 100% to 100%"
                sentence = (f"If you did not have {name}, the top prediction would stay {classes[target]} "
                            f"with almost the same confidence ({after:.0%}).")
            else:
                direction = "drop" if after < before else "rise"
                sentence = (f"If you did not have {name}, the top prediction would stay {classes[target]} "
                            f"but its confidence would {direction} from {before:.0%} to {after:.0%}.")
        rows.append({"symptom": name, "new_top": classes[new], "new_top_prob": float(p[new]),
                     "target_before": before, "target_after": after, "delta": after - before,
                     "changes_prediction": new != target, "sentence": sentence})
    rows.sort(key=lambda r: -abs(r["delta"]))
    return rows


# --------------------------------------------------------------------------- #
# Global explanations (SHAP)
# --------------------------------------------------------------------------- #
def _to_nfc(values, n: int, d: int) -> np.ndarray:
    """Normalise the different shapes SHAP versions return into ``(n, d, n_outputs)``."""
    if isinstance(values, list):  # old API: list of (n, d) arrays, one per output
        return np.stack([np.asarray(v) for v in values], axis=-1)
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.shape[0] == n and arr.shape[1] == d:
        return arr
    if arr.shape[1] == n and arr.shape[2] == d:  # (outputs, n, d)
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Unexpected SHAP output shape {arr.shape}")


def shap_mlp(model: MLP, temperature: float, X_bg: np.ndarray, X: np.ndarray) -> np.ndarray:
    """SHAP values ``(n, d, n_classes)`` for the MLP's calibrated probabilities."""
    import shap

    model.eval()
    wrapper = _ProbWrapper(model, temperature).eval()
    explainer = shap.GradientExplainer(wrapper, torch.as_tensor(X_bg, dtype=torch.float32))
    values = explainer.shap_values(torch.as_tensor(X, dtype=torch.float32), nsamples=config.SHAP_NSAMPLES)
    return _to_nfc(values, len(X), X.shape[1])


def shap_tree(clf: BaselineClassifier, X: np.ndarray) -> np.ndarray:
    """SHAP values ``(n, d, n_classes)`` for a Random Forest or XGBoost baseline."""
    import shap

    values = shap.TreeExplainer(clf.est).shap_values(X, check_additivity=False)
    compact = _to_nfc(values, len(X), X.shape[1])
    out = np.zeros((len(X), X.shape[1], clf.n_classes))
    out[:, :, clf.classes_] = compact  # classes unseen in training get zeros
    return out


def shap_linear(clf: BaselineClassifier, X_bg: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Exact SHAP for a linear model in logit units: ``coef * (x - mean(background))``."""
    coef = clf.est.coef_  # (n_seen_classes, d)
    contrib = (X - X_bg.mean(0))[:, :, None] * coef.T[None, :, :]
    out = np.zeros((len(X), X.shape[1], clf.n_classes))
    out[:, :, clf.classes_] = contrib
    return out


def global_importance(sv: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per symptom, taken on the output of each row's TRUE disease."""
    return np.abs(sv[np.arange(len(y)), :, y]).mean(0)


def class_symptom_effect(sv: np.ndarray, X: np.ndarray, min_support: int = 2) -> np.ndarray:
    """``(n_classes, d)`` matrix: mean SHAP toward each disease when a symptom is PRESENT.

    NaN where fewer than ``min_support`` rows contain the symptom. Used by the
    sanity check to ask "which symptoms does the model treat as evidence for
    disease c?".
    """
    present = (X > 0).astype(float)  # (n, d)
    support = present.sum(0)  # (d,)
    summed = np.einsum("nd,ndc->cd", present, sv)
    with np.errstate(invalid="ignore", divide="ignore"):
        effect = summed / support[None, :]
    effect[:, support < min_support] = np.nan
    return effect


def plot_global_bar(importance: np.ndarray, symptoms: list[str], title: str, xlabel: str, path: Path, top: int = 15) -> Path:
    """Bar chart of the ``top`` most important symptoms."""
    order = np.argsort(-importance)[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * top + 1.4))
    ax.barh([symptoms[i] for i in order], importance[order], color=BLUE)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E5E5E5")
    ax.set_axisbelow(True)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", path)
    return path


def compute_all_shap(
    X_bg: np.ndarray, X: np.ndarray, seed: int = config.SEED
) -> dict[str, tuple[np.ndarray, str, np.ndarray]]:
    """SHAP matrices for every available model.

    Returns ``{name: (shap_values, unit, row_indices)}`` where ``row_indices`` says
    which rows of ``X`` were explained (all rows, except for the slow Random
    Forest explainer, which uses a random subset). A model whose explainer fails
    is skipped with a warning, so one incompatible library version cannot break
    the whole report.
    """
    results: dict[str, tuple[np.ndarray, str, np.ndarray]] = {}
    all_rows = np.arange(len(X))
    sub_rows = np.sort(np.random.default_rng(seed).permutation(len(X))[: config.SHAP_TREE_MAX_ROWS])
    model, temperature, _ = load_mlp()
    try:
        results["mlp"] = (shap_mlp(model, temperature, X_bg, X), "probability", all_rows)
    except Exception as err:  # noqa: BLE001 - fall back to Integrated Gradients
        logger.warning("SHAP GradientExplainer failed (%s); using Integrated Gradients for the MLP.", err)
        results["mlp"] = (_ig_all_classes(model, temperature, X), "probability (Integrated Gradients)", all_rows)
    for name in BASELINE_NAMES:
        path = BASELINE_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        clf = joblib.load(path)
        try:
            if name == "logreg":
                results[name] = (shap_linear(clf, X_bg, X), "logit", all_rows)
            elif name == "random_forest":
                results[name] = (shap_tree(clf, X[sub_rows]), "probability", sub_rows)
            else:
                results[name] = (shap_tree(clf, X), "log-odds", all_rows)
        except Exception as err:  # noqa: BLE001
            logger.warning("SHAP failed for %s: %s", name, err)
    return results


def _ig_all_classes(model: MLP, temperature: float, X: np.ndarray) -> np.ndarray:
    """Fallback: Integrated Gradients for every class (slower, but no SHAP dependency)."""
    n_classes = model.net[-1].out_features
    out = np.zeros((len(X), X.shape[1], n_classes), dtype=np.float32)
    for i, x in enumerate(X):
        for c in range(n_classes):
            out[i, :, c] = integrated_gradients(model, temperature, x, c, n_steps=32)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: global SHAP plots for every model and one example local explanation."""
    parser = argparse.ArgumentParser(description="Generate explainability figures.")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)
    setup_logging()
    set_seed(args.seed)

    symptoms, classes = load_vocab()
    X_tr, _ = load_split("train")
    X, y = load_split("full")
    rng = np.random.default_rng(args.seed)
    bg = X_tr[rng.choice(len(X_tr), size=min(config.SHAP_BACKGROUND_SIZE, len(X_tr)), replace=False)]

    all_shap = compute_all_shap(bg, X)
    for name, (sv, unit, rows) in all_shap.items():
        imp = global_importance(sv, y[rows])
        plot_global_bar(imp, symptoms, f"Most important symptoms: {BASELINE_LABELS[name]}",
                        f"Mean |SHAP value| ({unit} units)", config.FIGURES_DIR / f"shap_global_{name}.png")
        # The web app redraws these bars in its own light/dark theme, so every model's top symptoms are saved as data.
        top = np.argsort(-imp)[:20]
        pd.DataFrame({"symptom": [symptoms[i] for i in top], "mean_abs_shap": imp[top]}).to_csv(
            config.REPORTS_DIR / f"shap_{name}_top_symptoms.csv", index=False)

    # Example local explanation on the first test patient.
    model, temperature, _ = load_mlp()
    X_te, y_te = load_split("test")
    x = X_te[0]
    p = torch.softmax(model(torch.as_tensor(x)[None]) / temperature, dim=1).detach().numpy()[0]
    top = int(p.argmax())
    attr = integrated_gradients(model, temperature, x, top)
    supports, against = split_contributions(attr, x, symptoms)
    text, _ = plain_language_explanation(classes[top], supports, against)
    logger.info("Example: %s", text)
    fig = plot_local_explanation(supports, against, classes[top])
    out = config.FIGURES_DIR / "local_explanation_example.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
