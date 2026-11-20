"""Train the three baseline models: Logistic Regression, Random Forest, XGBoost.

Run::

    python -m src.train_baselines
    python -m src.train_baselines --aug-copies 0     # no augmentation

Baselines receive the SAME symptom-dropout augmentation as the MLP, but as
static extra copies of the training set (they cannot resample every epoch). This
keeps the baseline-vs-MLP comparison fair.
"""
from __future__ import annotations

import argparse
import logging
import time

import joblib
import numpy as np

from src import config
from src.models import BASELINE_LABELS, BASELINE_NAMES, BaselineClassifier
from src.preprocess import load_split, load_vocab, random_symptom_dropout
from src.utils import ensure_dir, set_seed, setup_logging

logger = logging.getLogger(__name__)
BASELINE_DIR = config.MODELS_DIR / "baselines"


def augmented_copies(X: np.ndarray, y: np.ndarray, copies: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the original rows plus ``copies`` randomly thinned versions of them."""
    if copies <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    parts_X = [X] + [random_symptom_dropout(X, rng) for _ in range(copies)]
    return np.vstack(parts_X), np.concatenate([y] * (copies + 1))


def fit_baseline(
    name: str, X: np.ndarray, y: np.ndarray, n_classes: int, aug_copies: int, seed: int
) -> BaselineClassifier:
    """Fit one baseline on (optionally augmented) data."""
    Xa, ya = augmented_copies(X, y, aug_copies, seed)
    return BaselineClassifier(name, n_classes, seed).fit(Xa, ya)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train baseline classifiers.")
    parser.add_argument("--aug-copies", type=int, default=5, help="augmented copies of the train set")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)
    setup_logging()
    set_seed(args.seed)

    _, classes = load_vocab()
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    ensure_dir(BASELINE_DIR)

    for name in BASELINE_NAMES:
        start = time.time()
        model = fit_baseline(name, X_tr, y_tr, len(classes), args.aug_copies, args.seed)
        val_acc = float((model.predict(X_va) == y_va).mean())
        joblib.dump(model, BASELINE_DIR / f"{name}.joblib")
        logger.info(
            "%-20s val accuracy %.3f | trained in %.1fs", BASELINE_LABELS[name], val_acc, time.time() - start
        )
    logger.info("Saved baselines to %s", BASELINE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
