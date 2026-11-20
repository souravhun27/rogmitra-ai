"""OPTIONAL SYNTHETIC EXPERIMENT: predicting two diseases at once (multi-label).

Run::

    python -m src.train_multilabel

IMPORTANT, say this in your report: the dataset has NO patients with two
diseases. We therefore CREATE co-occurring cases by merging (logical OR) the
symptom vectors of one random row of disease A and one of disease B, with both
diseases as labels. Real co-morbidity looks different (diseases interact, share
symptoms and change each other's presentation), so results show only that the
architecture can handle multiple labels, not that it can diagnose real
multi-disease patients.

Method: same MLP body as the main model, but SIGMOID outputs (one independent
yes/no per disease) trained with binary cross-entropy (BCE), instead of a
softmax that forces probabilities to compete and sum to 1. A decision threshold
is tuned on the validation split.

Leakage control: pairs for train / val / test are built ONLY from rows of that
same split, so no symptom row appears in two splits.
"""
from __future__ import annotations

import argparse
import copy
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, hamming_loss

from src import config
from src.models import build_mlp, load_mlp, mlp_proba, save_mlp
from src.preprocess import load_split, load_vocab, random_symptom_dropout
from src.utils import save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)


def single_cases(X: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Original one-disease rows with one-hot labels (so the model can still say 'just one')."""
    Y = np.zeros((len(y), n_classes), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    return X.astype(np.float32), Y


def pair_cases(
    X: np.ndarray, y: np.ndarray, n_pairs: int, n_classes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Create ``n_pairs`` synthetic two-disease cases from the rows of ONE split.

    A symptom is present if it is present in either parent row (logical OR).
    """
    pools = {c: np.flatnonzero(y == c) for c in range(n_classes)}
    usable = [c for c, idx in pools.items() if len(idx)]
    Xp = np.zeros((n_pairs, X.shape[1]), dtype=np.float32)
    Yp = np.zeros((n_pairs, n_classes), dtype=np.float32)
    for i in range(n_pairs):
        a, b = rng.choice(usable, size=2, replace=False)
        Xp[i] = np.maximum(X[rng.choice(pools[a])], X[rng.choice(pools[b])])
        Yp[i, [a, b]] = 1.0
    return Xp, Yp


def train_bce(
    X_tr: np.ndarray, Y_tr: np.ndarray, X_va: np.ndarray, Y_va: np.ndarray, seed: int,
    max_epochs: int = config.MLP_MAX_EPOCHS, patience: int = config.MLP_EARLY_STOP_PATIENCE,
):
    """Train the sigmoid/BCE MLP with symptom-dropout augmentation and early stopping."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_mlp(X_tr.shape[1], Y_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=config.MLP_LR, weight_decay=config.MLP_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=config.MLP_SCHEDULER_PATIENCE)
    Xv, Yv = torch.as_tensor(X_va), torch.as_tensor(Y_va)
    Yt = torch.as_tensor(Y_tr)
    best, best_state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        Xt = torch.as_tensor(random_symptom_dropout(X_tr, rng))
        order = torch.as_tensor(rng.permutation(len(Xt)))
        for start in range(0, len(order), config.MLP_BATCH_SIZE):
            idx = order[start : start + config.MLP_BATCH_SIZE]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(Xt[idx]), Yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(model(Xv), Yv).item()
        sched.step(val_loss)
        if val_loss < best - 1e-5:
            best, best_state, stale, best_epoch = val_loss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    logger.info("Best epoch %d (val BCE %.4f)", best_epoch, best)
    return model, best_epoch


def predict_scores(model, X: np.ndarray) -> np.ndarray:
    """Independent per-disease probabilities (sigmoid of the logits)."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()


def tune_threshold(scores: np.ndarray, Y: np.ndarray) -> float:
    """Pick the global threshold in 0.05..0.90 that maximises micro-F1 on validation data."""
    grid = np.arange(0.05, 0.91, 0.05)
    f1s = [f1_score(Y, scores >= t, average="micro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def multilabel_metrics(scores: np.ndarray, Y: np.ndarray, threshold: float) -> dict[str, float]:
    """Set-based metrics for multi-label predictions.

    ``top_k_hit`` asks: are ALL true diseases among the k highest scores, where
    k = number of true diseases (the fair 'did it find the right ones' test).
    """
    pred = scores >= threshold
    k = Y.sum(1).astype(int)
    order = np.argsort(-scores, axis=1)
    hit = np.mean([Y[i, order[i, : k[i]]].all() for i in range(len(Y))])
    return {
        "exact_match": float((pred == Y.astype(bool)).all(1).mean()),
        "micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y, pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(Y, pred)),
        "top_k_hit": float(hit),
        "mean_labels_predicted": float(pred.sum(1).mean()),
    }


def softmax_topk_hit(proba: np.ndarray, Y: np.ndarray) -> float:
    """Same 'all true diseases in the top-k' test for the single-label softmax model."""
    k = Y.sum(1).astype(int)
    order = np.argsort(-proba, axis=1)
    return float(np.mean([Y[i, order[i, : k[i]]].all() for i in range(len(Y))]))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Synthetic multi-label (two diseases at once) experiment.")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--pairs-train", type=int, default=config.ML_PAIRS_TRAIN)
    args = parser.parse_args(argv)
    setup_logging()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    _, classes = load_vocab()
    C = len(classes)
    split = {name: load_split(name) for name in ("train", "val", "test")}
    sets = {}
    for name, n_pairs in (("train", args.pairs_train), ("val", config.ML_PAIRS_VAL), ("test", config.ML_PAIRS_TEST)):
        X, y = split[name]
        Xs, Ys = single_cases(X, y, C)
        Xp, Yp = pair_cases(X, y, n_pairs, C, rng)
        sets[name] = {"single": (Xs, Ys), "pair": (Xp, Yp)}
    logger.info("Synthetic pairs: train %d / val %d / test %d (built within each split only)",
                args.pairs_train, config.ML_PAIRS_VAL, config.ML_PAIRS_TEST)

    X_tr = np.vstack([sets["train"]["single"][0], sets["train"]["pair"][0]])
    Y_tr = np.vstack([sets["train"]["single"][1], sets["train"]["pair"][1]])
    X_va = np.vstack([sets["val"]["single"][0], sets["val"]["pair"][0]])
    Y_va = np.vstack([sets["val"]["single"][1], sets["val"]["pair"][1]])

    start = time.time()
    model, best_epoch = train_bce(X_tr, Y_tr, X_va, Y_va, args.seed)
    logger.info("Training took %.1fs", time.time() - start)
    threshold = tune_threshold(predict_scores(model, X_va), Y_va)
    logger.info("Validation-tuned threshold: %.2f", threshold)

    results: dict = {"threshold": threshold, "best_epoch": best_epoch, "n_pairs_train": args.pairs_train}
    softmax_model = None
    try:
        softmax_model, temperature, _ = load_mlp()
    except FileNotFoundError:
        logger.info("Single-label MLP not trained; skipping the softmax comparison.")
    for kind in ("pair", "single"):
        Xk, Yk = sets["test"][kind]
        res = multilabel_metrics(predict_scores(model, Xk), Yk, threshold)
        if softmax_model is not None:
            res["softmax_mlp_top_k_hit"] = softmax_topk_hit(mlp_proba(softmax_model, Xk, temperature), Yk)
        results[f"test_{kind}"] = res
        logger.info("test %-6s %s", kind, {k: round(v, 3) for k, v in res.items()})

    save_mlp(model, config.ML_MODEL_PATH, 1.0, extra={"threshold": threshold, "synthetic_experiment": True})
    save_json(results, config.ML_METRICS_JSON)
    p, s = results["test_pair"], results["test_single"]
    lines = [
        "# Multi-label experiment (SYNTHETIC)", "",
        "Two-disease cases were created by merging the symptom sets of two random single-disease rows. "
        "No real co-morbidity data exists in the dataset, so these results demonstrate the method only.", "",
        "| Test set | Exact match | Micro F1 | Macro F1 | Hamming loss | All true diseases in top-k | Softmax MLP, same test |",
        "|---|---|---|---|---|---|---|",
        f"| Two-disease cases | {p['exact_match']:.3f} | {p['micro_f1']:.3f} | {p['macro_f1']:.3f} | {p['hamming_loss']:.4f} | {p['top_k_hit']:.3f} | {p.get('softmax_mlp_top_k_hit', float('nan')):.3f} |",
        f"| Single-disease cases | {s['exact_match']:.3f} | {s['micro_f1']:.3f} | {s['macro_f1']:.3f} | {s['hamming_loss']:.4f} | {s['top_k_hit']:.3f} | {s.get('softmax_mlp_top_k_hit', float('nan')):.3f} |",
        "", f"Decision threshold (tuned on validation): {threshold:.2f}. 'top-k' uses k = number of true diseases.",
    ]
    config.ML_METRICS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved %s and reports/multilabel_metrics.*", config.ML_MODEL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
