"""Stricter evaluation: is the near-perfect test accuracy real or just an easy dataset?

Run (after ``train_baselines`` and ``train_mlp``)::

    python -m src.evaluate_hard

Three questions are answered, all on models that are already trained:

1. **Generalisation gap** - accuracy on train vs validation vs test. A large gap
   would indicate overfitting.
2. **Novel-pattern test** - the symptom data contain many rows that are sub- or
   super-sets of another row of the same disease. A test row that is such a
   near-copy of a training row is "easy". We keep only test rows that are NOT
   near-copies of any training row, which is a much stricter test.
3. **Noisy-input test** - 30% of each patient's real symptoms are removed AND
   0..4 unrelated (wrong) symptoms are added, imitating a careless user.

Outputs: reports/hard_eval.json, reports/hard_eval.csv, reports/figures/hard_eval.png
"""
from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.evaluate import COLOURS, load_predictors
from src.models import BASELINE_LABELS
from src.preprocess import load_split, remove_fraction
from src.utils import ensure_dir, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)
NOISE_REMOVE_FRACTION = 0.3
WRONG_COUNTS = (0, 1, 2, 3, 4)


def near_copy_mask(X_test: np.ndarray, y_test: np.ndarray, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """True for test rows that are a sub- or super-set of a same-disease training row.

    Such rows add little information beyond what the model already saw, so
    scoring them flatters the model. Exact duplicates are already gone after
    de-duplication; this catches the remaining near-copies.
    """
    mask = np.zeros(len(X_test), dtype=bool)
    train_bool = X_train.astype(bool)
    for i, (row, label) in enumerate(zip(X_test.astype(bool), y_test)):
        same = train_bool[y_train == label]
        if len(same) == 0:
            continue
        is_subset = np.all(same[:, row], axis=1)          # test row symptoms all in a train row
        is_superset = ~np.any(same & ~row, axis=1)       # train row symptoms all in test row
        mask[i] = bool(np.any(is_subset | is_superset))
    return mask


def add_wrong_symptoms(X: np.ndarray, y_true_rows: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Add ``k`` random symptoms the patient did not have (simulated user mistakes).

    ``y_true_rows`` is unused for the choice (the wrong symptoms are drawn
    uniformly from the absent ones) but kept in the signature to make clear
    that no label information leaks into the noise.
    """
    out = X.copy()
    if k == 0:
        return out
    for i in range(len(out)):
        absent = np.flatnonzero(out[i] == 0)
        out[i, rng.choice(absent, size=min(k, len(absent)), replace=False)] = 1.0
    return out


def accuracy(fn, X: np.ndarray, y: np.ndarray) -> float:
    """Top-1 accuracy of a probability function."""
    return float((fn(X).argmax(1) == y).mean()) if len(y) else float("nan")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Strict evaluation (generalisation gap, novel patterns, noise).")
    parser.add_argument("--reps", type=int, default=30, help="random repetitions for the noisy test")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)
    setup_logging()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    predictors, _, _ = load_predictors()
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    X_te, y_te = load_split("test")

    # 1) generalisation gap -------------------------------------------------------------
    gap = {name: {"train": accuracy(fn, X_tr, y_tr), "val": accuracy(fn, X_va, y_va), "test": accuracy(fn, X_te, y_te)}
           for name, fn in predictors.items()}

    # 2) novel-pattern test -------------------------------------------------------------
    near = near_copy_mask(X_te, y_te, X_tr, y_tr)
    novel = ~near
    n_novel = int(novel.sum())
    novel_acc = {name: accuracy(fn, X_te[novel], y_te[novel]) for name, fn in predictors.items()}
    logger.info("test rows: %d | near-copies of training rows: %d | novel patterns kept: %d",
                len(y_te), int(near.sum()), n_novel)

    # 3) noisy-input test: remove 30% real symptoms, add k wrong ones ------------------
    rows = []
    for k in WRONG_COUNTS:
        accs: dict[str, list[float]] = {name: [] for name in predictors}
        for _ in range(args.reps):
            X_noisy = add_wrong_symptoms(remove_fraction(X_te, NOISE_REMOVE_FRACTION, rng), y_te, k, rng)
            for name, fn in predictors.items():
                accs[name].append(accuracy(fn, X_noisy, y_te))
        for name, vals in accs.items():
            rows.append({"wrong_symptoms_added": k, "model": name,
                         "accuracy_mean": float(np.mean(vals)), "accuracy_std": float(np.std(vals))})
    noise = pd.DataFrame(rows)
    ensure_dir(config.REPORTS_DIR)
    noise.to_csv(config.REPORTS_DIR / "hard_eval.csv", index=False)

    summary = {
        "n_test": int(len(y_te)),
        "n_near_copies": int(near.sum()),
        "n_novel_pattern_test": n_novel,
        "noise_removed_fraction": NOISE_REMOVE_FRACTION,
        "noise_reps": args.reps,
        "generalisation_gap": gap,
        "novel_pattern_accuracy": novel_acc,
        "noisy_accuracy": {name: {int(k): float(v) for k, v in
                                  noise[noise["model"] == name].set_index("wrong_symptoms_added")["accuracy_mean"].items()}
                           for name in predictors},
    }
    save_json(summary, config.REPORTS_DIR / "hard_eval.json")

    # chart: accuracy as more wrong symptoms are added -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name in predictors:
        sub = noise[noise["model"] == name]
        ax.plot(sub["wrong_symptoms_added"], sub["accuracy_mean"] * 100, marker="o", color=COLOURS[name],
                linewidth=2.2 if name == "mlp" else 1.5, label=BASELINE_LABELS[name])
        ax.fill_between(sub["wrong_symptoms_added"], (sub["accuracy_mean"] - sub["accuracy_std"]) * 100,
                        (sub["accuracy_mean"] + sub["accuracy_std"]) * 100, color=COLOURS[name], alpha=0.12)
    ax.set_xlabel(f"Wrong symptoms added (after removing {NOISE_REMOVE_FRACTION:.0%} of the real ones)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Robustness to missing and wrong symptoms")
    ax.set_xticks(list(WRONG_COUNTS))
    ax.grid(color="#E5E5E5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig_path = ensure_dir(config.FIGURES_DIR) / "hard_eval.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    for name in predictors:
        logger.info("%-20s train %.3f | val %.3f | test %.3f | novel-only %.3f | noisy(+2 wrong) %.3f",
                    BASELINE_LABELS[name], gap[name]["train"], gap[name]["val"], gap[name]["test"],
                    novel_acc[name], summary["noisy_accuracy"][name][2])
    logger.info("Saved reports/hard_eval.json, hard_eval.csv and %s", fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
