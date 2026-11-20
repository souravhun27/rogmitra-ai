"""Exploratory data analysis: saves figures to ``reports/figures/eda_*.png``.

Run::

    python -m src.eda

The notebook ``notebooks/01_eda.ipynb`` calls the same functions so the figures
you see interactively are exactly the ones saved for the report.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers and in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.preprocess import build_symptom_matrix, deduplicate, load_raw_tables
from src.utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

# Colour-blind-safe pair (Okabe-Ito blue / orange) + neutral grey; one accent per chart.
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#8A8A8A"


def _style() -> None:
    """Quiet, readable defaults: light grid, no top/right spines."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    ensure_dir(out_dir)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    logger.info("saved %s", path)
    return path


def load_for_eda(raw_dir: Path = config.RAW_DIR) -> dict:
    """Return cleaned raw matrix, de-duplicated matrix and vocabulary for plotting."""
    raw = load_raw_tables(raw_dir)
    X_raw, y_raw, symptoms = build_symptom_matrix(raw.dataset)
    X_u, y_u, report = deduplicate(X_raw, y_raw)
    return {
        "X_raw": X_raw, "y_raw": y_raw, "X_u": X_u, "y_u": y_u,
        "symptoms": symptoms, "report": report,
    }


def plot_class_balance(y_raw: np.ndarray, y_u: np.ndarray, out_dir: Path) -> Path:
    """Raw rows vs unique rows per disease: shows how few distinct patterns exist."""
    _style()
    classes = sorted(set(y_raw))
    raw_n = pd.Series(y_raw).value_counts().reindex(classes)
    uni_n = pd.Series(y_u).value_counts().reindex(classes).fillna(0)
    order = uni_n.sort_values().index
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(classes) + 1.5))
    pos = np.arange(len(order))
    ax.barh(pos + 0.2, raw_n[order].values, height=0.4, color=GREY, label="Raw rows")
    ax.barh(pos - 0.2, uni_n[order].values, height=0.4, color=BLUE, label="Unique rows")
    ax.set_yticks(pos, order, fontsize=8)
    ax.set_xlabel("Number of rows")
    ax.set_title("Rows per disease before vs after de-duplication")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, out_dir, "eda_class_balance.png")


def plot_symptoms_per_row(X_u: np.ndarray, out_dir: Path) -> Path:
    """Histogram of how many symptoms a (unique) patient record has."""
    _style()
    counts = X_u.sum(1)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(counts, bins=np.arange(counts.min(), counts.max() + 2) - 0.5, color=BLUE, edgecolor="white")
    ax.set_xlabel("Symptoms per record")
    ax.set_ylabel("Records")
    ax.set_title("How many symptoms does a record list?")
    return _save(fig, out_dir, "eda_symptoms_per_row.png")


def plot_top_symptoms(X_u: np.ndarray, symptoms: list[str], out_dir: Path, top: int = 25) -> Path:
    """Most frequent symptoms across unique records."""
    _style()
    freq = pd.Series(X_u.mean(0), index=symptoms).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.26 * top + 1.2))
    ax.barh(freq.index, freq.values * 100, color=BLUE)
    ax.set_xlabel("% of unique records containing the symptom")
    ax.set_title(f"Top {top} most common symptoms")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, out_dir, "eda_top_symptoms.png")


def plot_disease_symptom_heatmap(
    X_u: np.ndarray, y_u: np.ndarray, symptoms: list[str], out_dir: Path
) -> Path:
    """Disease x symptom frequency heatmap (one hue, light = rare, dark = common)."""
    _style()
    classes = sorted(set(y_u))
    mat = np.vstack([X_u[y_u == c].mean(0) for c in classes])
    keep = mat.sum(0) > 0
    fig, ax = plt.subplots(figsize=(max(12, 0.17 * keep.sum()), 0.3 * len(classes) + 2))
    sns.heatmap(
        mat[:, keep], ax=ax, cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Share of disease's records"},
        xticklabels=np.array(symptoms)[keep], yticklabels=classes, linewidths=0,
    )
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=5, rotation=90)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("Which symptoms belong to which disease?")
    return _save(fig, out_dir, "eda_disease_symptom_heatmap.png")


def disease_jaccard(X_u: np.ndarray, y_u: np.ndarray) -> pd.DataFrame:
    """Jaccard overlap between the symptom SETS of each pair of diseases.

    High overlap predicts which diseases a model will confuse (used in the
    failure-case analysis in Stage 2).
    """
    classes = sorted(set(y_u))
    sets = [set(np.flatnonzero(X_u[y_u == c].sum(0) > 0)) for c in classes]
    mat = np.zeros((len(classes), len(classes)))
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            mat[i, j] = len(a & b) / len(a | b) if a | b else 0.0
    return pd.DataFrame(mat, index=classes, columns=classes)


def plot_disease_overlap(jac: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Heatmap of disease symptom-set overlap + CSV of the most overlapping pairs."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(jac, ax=ax, cmap="Blues", vmin=0, vmax=1, square=True,
                cbar_kws={"label": "Jaccard similarity of symptom sets"})
    ax.grid(False)
    ax.tick_params(labelsize=6)
    ax.set_title("Diseases with similar symptom sets are easy to confuse")
    fig_path = _save(fig, out_dir, "eda_disease_overlap.png")

    pairs = [
        (a, b, jac.loc[a, b])
        for i, a in enumerate(jac.index)
        for j, b in enumerate(jac.columns)
        if i < j
    ]
    top = pd.DataFrame(sorted(pairs, key=lambda t: -t[2])[:15], columns=["disease_a", "disease_b", "jaccard"])
    csv_path = config.REPORTS_DIR / "eda_confusable_pairs.csv"
    ensure_dir(csv_path.parent)
    top.round(3).to_csv(csv_path, index=False)
    logger.info("saved %s", csv_path)
    return fig_path, csv_path


def run_eda(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.FIGURES_DIR) -> list[Path]:
    """Generate every EDA figure and return their paths."""
    d = load_for_eda(raw_dir)
    paths = [
        plot_class_balance(d["y_raw"], d["y_u"], out_dir),
        plot_symptoms_per_row(d["X_u"], out_dir),
        plot_top_symptoms(d["X_u"], d["symptoms"], out_dir),
        plot_disease_symptom_heatmap(d["X_u"], d["y_u"], d["symptoms"], out_dir),
    ]
    fig_path, _ = plot_disease_overlap(disease_jaccard(d["X_u"], d["y_u"]), out_dir)
    paths.append(fig_path)
    return paths


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate EDA figures.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.FIGURES_DIR)
    args = parser.parse_args(argv)
    setup_logging()
    try:
        run_eda(args.raw_dir, args.out_dir)
    except FileNotFoundError as err:
        logger.error("%s", err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
