"""Theme-aware charts for the Model Performance page.

The training scripts save light-background PNGs for the README. The web app instead redraws the key figures
here from the saved model and report data, with a transparent background and colours that follow the
light / dark switch, so they look right inside the cards. Every function returns PNG bytes and is cached
per (mode, report file timestamp).
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from src import config  # noqa: E402
from src.calibration import reliability_bins  # noqa: E402
from src.models import BASELINE_LABELS, load_mlp, mlp_logits, softmax  # noqa: E402
from src.preprocess import load_split, load_vocab  # noqa: E402

# Colours must be hex (matplotlib does not read CSS rgba()).
CHART = {
    "dark": {"fg": "#f4ece6", "muted": "#b9aba2", "grid": "#ffffff1f", "axis": "#ffffff44", "accent": "#ec7a3c",
             "blue": "#5bb6ea", "green": "#3ccf76", "red": "#f0605d"},
    "light": {"fg": "#2b1d17", "muted": "#6b5a50", "grid": "#0000001a", "axis": "#00000055", "accent": "#d9602a",
              "blue": "#1f7fb8", "green": "#1d9a52", "red": "#d23b3b"},
}
# Same model -> same colour on every chart; the MLP always carries the accent colour.
MODEL_COLOURS = {
    "dark": {"logreg": "#5bb6ea", "random_forest": "#b79cf0", "xgboost": "#3ccf76", "mlp": "#ec7a3c"},
    "light": {"logreg": "#1f7fb8", "random_forest": "#7b5cc4", "xgboost": "#1d9a52", "mlp": "#d9602a"},
}
MODEL_MARKERS = {"logreg": "o", "random_forest": "s", "xgboost": "^", "mlp": "D"}


def _rc(mode: str) -> dict:
    c = CHART[mode]
    return {"figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
            "text.color": c["fg"], "axes.labelcolor": c["muted"], "axes.edgecolor": c["axis"],
            "xtick.color": c["muted"], "ytick.color": c["muted"], "axes.grid": True, "grid.color": c["grid"],
            "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
            "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "legend.labelcolor": c["fg"],
            "font.family": "DejaVu Sans"}


def _png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def _stamp() -> float:
    """Changes whenever evaluate.py rewrites its report, so cached charts refresh after re-training."""
    total = 0.0
    for name in ("metrics.json", "hard_eval.csv"):  # either report changing must refresh cached charts
        p = config.REPORTS_DIR / name
        total += p.stat().st_mtime if p.exists() else 0.0
    return total


@st.cache_resource(show_spinner=False)
def _mlp_eval(stamp: float) -> dict:
    """Test-split logits of the saved MLP (tiny: a few dozen rows)."""
    model, temperature, _ = load_mlp()
    X, y = load_split("test")
    return {"logits": mlp_logits(model, X), "y": y, "T": float(temperature), "classes": load_vocab()[1]}


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def confusion_png(mode: str, stamp: float, labels: bool = False) -> bytes:
    """Row-normalised confusion matrix of the MLP; ``labels=True`` writes disease names on the axes."""
    d = _mlp_eval(stamp)
    classes, n = d["classes"], len(d["classes"])
    pred = softmax(d["logits"], d["T"]).argmax(1)
    cm = confusion_matrix(d["y"], pred, labels=range(n)).astype(float)
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("edx", ["#2a1d1b" if mode == "dark" else "#fbeee3", "#a8431f", CHART[mode]["accent"], "#ffd9a5"])
    with plt.rc_context(_rc(mode)):
        fig, ax = plt.subplots(figsize=(9.5, 8.4) if labels else (4.6, 4.2))
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        if labels:
            ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
            ax.set_yticks(range(n), classes, fontsize=7)
        else:
            step = max(1, n // 10)
            ax.set_xticks(range(0, n, step), range(0, n, step), fontsize=7)
            ax.set_yticks(range(0, n, step), range(0, n, step), fontsize=7)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7)
        return _png(fig)


@st.cache_data(show_spinner=False)
def reliability_png(mode: str, stamp: float) -> bytes:
    """Reliability diagram: accuracy per confidence bin, before and after temperature scaling."""
    d = _mlp_eval(stamp)
    c = CHART[mode]
    from src.calibration import expected_calibration_error

    p0, p1 = softmax(d["logits"], 1.0), softmax(d["logits"], d["T"])
    with plt.rc_context(_rc(mode)):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.plot([0, 1], [0, 1], ls="--", color=c["muted"], lw=1, label="Perfect calibration")
        for probs, label, colour, marker in ((p0, "Before TS", c["blue"], "o"), (p1, "After TS", c["accent"], "s")):
            bins = reliability_bins(probs, d["y"])
            ece = expected_calibration_error(probs, d["y"])
            ax.plot([b[0] for b in bins], [b[1] for b in bins], marker=marker, color=colour, lw=1.8, ms=5,
                    label=f"{label} (ECE = {ece:.3f})")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Mean predicted probability (confidence)")
        ax.set_ylabel("Fraction correct")
        ax.legend(loc="lower right", fontsize=8)
        return _png(fig)


@st.cache_data(show_spinner=False)
def robustness_png(mode: str, stamp: float) -> bytes:
    """Accuracy as a growing share of each patient's symptoms is removed."""
    df = pd.read_csv(config.REPORTS_DIR / "robustness.csv")
    c = CHART[mode]
    with plt.rc_context(_rc(mode)):
        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        for name in ("logreg", "random_forest", "xgboost", "mlp"):
            sub = df[df["model"] == name].sort_values("fraction_removed")
            if sub.empty:
                continue
            x, m, s = sub["fraction_removed"] * 100, sub["accuracy_mean"], sub["accuracy_std"]
            col = MODEL_COLOURS[mode][name]
            ax.plot(x, m, marker=MODEL_MARKERS[name], color=col, lw=2.2 if name == "mlp" else 1.6, ms=5, label=BASELINE_LABELS[name])
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.13, lw=0)
        ax.set_xlabel("Symptoms removed from each patient (%)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(max(0, float(df["accuracy_mean"].min()) - 0.08), 1.02)
        ax.set_xticks(sorted(df["fraction_removed"].unique() * 100))
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        ax.axhline(1.0, color=c["grid"], lw=0.8)
        return _png(fig)


@st.cache_data(show_spinner=False)
def hard_png(mode: str, stamp: float) -> bytes:
    """Accuracy when 30% of real symptoms are removed and 0..4 wrong symptoms are added."""
    df = pd.read_csv(config.REPORTS_DIR / "hard_eval.csv")
    c = CHART[mode]
    with plt.rc_context(_rc(mode)):
        fig, ax = plt.subplots(figsize=(7.4, 4.0))
        for name in ("logreg", "random_forest", "xgboost", "mlp"):
            sub = df[df["model"] == name].sort_values("wrong_symptoms_added")
            if sub.empty:
                continue
            x, m, s = sub["wrong_symptoms_added"], sub["accuracy_mean"], sub["accuracy_std"]
            col = MODEL_COLOURS[mode][name]
            ax.plot(x, m, marker=MODEL_MARKERS[name], color=col, lw=2.2 if name == "mlp" else 1.6, ms=5, label=BASELINE_LABELS[name])
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.13, lw=0)
        ax.set_xlabel("Wrong symptoms added (after removing 30% of the real ones)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(max(0, float(df["accuracy_mean"].min()) - 0.08), 1.02)
        ax.set_xticks(sorted(df["wrong_symptoms_added"].unique()))
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        ax.axhline(1.0, color=c["grid"], lw=0.8)
        return _png(fig)


@st.cache_data(show_spinner=False)
def comparison_png(mode: str, stamp: float, metrics: dict) -> bytes:
    """Grouped bars: accuracy, macro-F1 and top-3 accuracy per model on the test split."""
    keys = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("top3_accuracy", "Top-3")]
    names = [n for n in ("logreg", "random_forest", "xgboost", "mlp") if n in metrics]
    x = np.arange(len(keys))
    w = 0.8 / max(len(names), 1)
    with plt.rc_context(_rc(mode)):
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for i, name in enumerate(names):
            vals = [metrics[name][k] for k, _ in keys]
            bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, vals, w * 0.9, color=MODEL_COLOURS[mode][name], label=BASELINE_LABELS[name])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=CHART[mode]["fg"])
        ax.set_xticks(x, [lab for _, lab in keys])
        ax.set_ylim(min(0.0, 0.9 * min(metrics[n][k] for n in names for k, _ in keys)), 1.12)
        ax.set_ylabel("Score (test split)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", ncol=len(names), fontsize=8, bbox_to_anchor=(0.5, 1.14))
        return _png(fig)


@st.cache_data(show_spinner=False)
def cv_png(mode: str, stamp: float, cv: dict) -> bytes:
    """Mean accuracy / macro-F1 with +/- one standard deviation from the stratified 5-fold CV."""
    names = [n for n in ("logreg", "random_forest", "xgboost", "mlp") if n in cv]
    x = np.arange(len(names))
    c = CHART[mode]
    with plt.rc_context(_rc(mode)):
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for j, (key, label, hatch) in enumerate((("accuracy", "Accuracy", None), ("macro_f1", "Macro F1", "//"))):
            means = [np.mean(cv[n][key]) * 100 for n in names]
            stds = [np.std(cv[n][key]) * 100 for n in names]
            ax.bar(x + (j - 0.5) * 0.38, means, 0.36, yerr=stds, capsize=3, hatch=hatch, label=label,
                   color=[MODEL_COLOURS[mode][n] for n in names], edgecolor=c["fg"] if hatch else "none", linewidth=0,
                   error_kw={"ecolor": c["fg"], "lw": 1.1})
        ax.set_xticks(x, [BASELINE_LABELS[n] for n in names], fontsize=8)
        ax.set_ylabel("Score over 5 folds (%)")
        lo = min(min(np.mean(cv[n][k]) * 100 - np.std(cv[n][k]) * 100 for n in names) for k in ("accuracy", "macro_f1"))
        ax.set_ylim(max(0, lo - 8), 104)
        ax.grid(axis="x", visible=False)
        from matplotlib.patches import Patch

        ax.legend(handles=[Patch(facecolor=c["muted"], label="Accuracy"), Patch(facecolor=c["muted"], hatch="//", edgecolor=c["fg"], label="Macro F1")],
                  loc="lower right", fontsize=8)
        return _png(fig)


@st.cache_data(show_spinner=False)
def shap_png(mode: str, stamp: float, model: str, top: int = 12) -> bytes | None:
    """Global SHAP importance (mean |value|) of the most important symptoms for one model."""
    path = config.REPORTS_DIR / f"shap_{model}_top_symptoms.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).head(top)[::-1]
    c = CHART[mode]
    with plt.rc_context(_rc(mode)):
        fig, ax = plt.subplots(figsize=(5.2, 0.34 * len(df) + 0.9))
        ax.barh([s.capitalize() for s in df["symptom"]], df["mean_abs_shap"], color=MODEL_COLOURS[mode][model], height=0.66)
        ax.set_xlabel("Mean |SHAP value|")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8, colors=c["fg"])
        return _png(fig)


def shap_stamp() -> float:
    """Timestamp of the newest SHAP CSV (cache key)."""
    files = list(Path(config.REPORTS_DIR).glob("shap_*_top_symptoms.csv"))
    return max((f.stat().st_mtime for f in files), default=0.0)
