"""Probability calibration: temperature scaling, ECE and reliability diagrams.

Why calibrate? A network trained with cross-entropy is often over-confident:
when it says "90%" it may be right only 70% of the time. For a health tool the
shown percentage must mean something, so we learn ONE number T on the
validation set and divide all logits by it before the softmax. T changes the
confidence but never the predicted class.

Run ``python -m src.calibration`` to produce the before/after report for the
saved MLP.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src import config
from src.models import load_mlp, mlp_logits, softmax
from src.preprocess import load_split
from src.utils import ensure_dir, save_json, setup_logging

logger = logging.getLogger(__name__)
BLUE, ORANGE = "#0072B2", "#E69F00"


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T > 0 minimising the negative log-likelihood of ``softmax(logits / T)``.

    We optimise log(T) so T stays positive, using L-BFGS (one parameter, convex
    enough for this to converge reliably). The result is clipped to [0.05, 20].
    """
    z = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: weighted average gap between confidence and accuracy over equal-width bins.

    Uses the top-1 probability as "confidence". 0 is perfect calibration.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean -log p(true class); lower is better and penalises confident mistakes."""
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Return per-bin ``(mean_confidence, accuracy, count)`` lists (empty bins skipped)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out.append((float(conf[mask].mean()), float(correct[mask].mean()), int(mask.sum())))
    return out


def plot_reliability(
    before: np.ndarray, after: np.ndarray, labels: np.ndarray, path: Path, temperature: float
) -> Path:
    """Two-panel reliability diagram (before | after temperature scaling)."""
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
    for ax, probs, title, colour in (
        (axes[0], before, "Before (T = 1)", ORANGE),
        (axes[1], after, f"After (T = {temperature:.2f})", BLUE),
    ):
        bins = reliability_bins(probs, labels)
        ax.plot([0, 1], [0, 1], "--", color="#8A8A8A", label="Perfect calibration")
        if bins:
            conf, acc, cnt = zip(*bins)
            ax.bar(conf, acc, width=0.08, color=colour, edgecolor="white", label="Observed accuracy")
            for c, a, n in zip(conf, acc, cnt):
                ax.text(c, min(a + 0.02, 0.97), str(n), ha="center", fontsize=7, color="#444")
        ax.set_title(f"{title}\nECE = {expected_calibration_error(probs, labels):.3f}")
        ax.set_xlabel("Confidence")
        ax.grid(color="#E5E5E5")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Reliability diagram (numbers above bars = samples in bin)", fontsize=10, y=1.06)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", path)
    return path


def calibration_summary(
    logits: np.ndarray, labels: np.ndarray, temperature: float
) -> dict[str, float]:
    """ECE and NLL before / after scaling; accuracy is unchanged by construction."""
    p0, p1 = softmax(logits, 1.0), softmax(logits, temperature)
    return {
        "temperature": temperature,
        "accuracy": float((p0.argmax(1) == labels).mean()),
        "ece_before": expected_calibration_error(p0, labels),
        "ece_after": expected_calibration_error(p1, labels),
        "nll_before": negative_log_likelihood(p0, labels),
        "nll_after": negative_log_likelihood(p1, labels),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: evaluate calibration of the saved MLP on the test split."""
    parser = argparse.ArgumentParser(description="Calibration report for the saved MLP.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)
    setup_logging()
    model, temperature, _ = load_mlp()
    X, y = load_split(args.split)
    logits = mlp_logits(model, X)
    summary = calibration_summary(logits, y, temperature)
    logger.info("Calibration on %s (n=%d): %s", args.split, len(y), summary)
    save_json(summary, config.REPORTS_DIR / "calibration.json")
    plot_reliability(
        softmax(logits, 1.0), softmax(logits, temperature), y,
        config.FIGURES_DIR / "reliability_diagram.png", temperature,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
