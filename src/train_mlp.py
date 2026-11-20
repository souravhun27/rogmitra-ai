"""Train the main PyTorch MLP, then calibrate it with temperature scaling.

Run::

    python -m src.train_mlp
    python -m src.train_mlp --no-augment --epochs 100

Pipeline: AdamW + label smoothing, ReduceLROnPlateau, early stopping on the
validation NLL, best-checkpoint restore, then temperature scaling fitted on the
validation set. The test split is never touched here (it is used only by
``src.evaluate``) so test numbers stay honest.
"""
from __future__ import annotations

import argparse
import copy
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src import config
from src.calibration import calibration_summary, fit_temperature
from src.models import MLP, build_mlp, mlp_logits, save_mlp
from src.preprocess import load_split, load_vocab, random_symptom_dropout
from src.utils import ensure_dir, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)


def train_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_classes: int,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    seed: int = config.SEED,
    augment: bool = True,
    max_epochs: int = config.MLP_MAX_EPOCHS,
    patience: int = config.MLP_EARLY_STOP_PATIENCE,
    fixed_epochs: int | None = None,
) -> tuple[MLP, pd.DataFrame, int]:
    """Train an MLP and return ``(model, history, best_epoch)``.

    With a validation set: early stopping + best-checkpoint restore.
    With ``fixed_epochs`` (used inside cross-validation, where no separate
    validation set exists): train exactly that many epochs and keep the last
    weights, so no information from the held-out fold leaks into training.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    use_val = X_val is not None and fixed_epochs is None
    model = build_mlp(X_tr.shape[1], n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=config.MLP_LR, weight_decay=config.MLP_WEIGHT_DECAY)
    # Plateau scheduler: halve the LR when validation loss stops improving.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=config.MLP_SCHEDULER_PATIENCE
    )
    y_t = torch.as_tensor(y_tr, dtype=torch.long)
    if use_val:
        Xv = torch.as_tensor(X_val, dtype=torch.float32)
        yv = torch.as_tensor(y_val, dtype=torch.long)

    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    best_nll, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        # New random symptom dropout EVERY epoch: the model never sees the same
        # thinned patient twice, which acts as strong regularisation.
        X_ep = random_symptom_dropout(X_tr, rng) if augment else X_tr
        X_t = torch.as_tensor(X_ep, dtype=torch.float32)
        order = torch.as_tensor(rng.permutation(len(X_t)))
        running, seen = 0.0, 0
        for start in range(0, len(order), config.MLP_BATCH_SIZE):
            idx = order[start : start + config.MLP_BATCH_SIZE]
            if len(idx) < 2:  # BatchNorm cannot normalise a single sample
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(X_t[idx]), y_t[idx], label_smoothing=config.MLP_LABEL_SMOOTHING)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
            seen += len(idx)
        train_loss = running / max(seen, 1)

        row = {"epoch": epoch, "train_loss": train_loss, "lr": opt.param_groups[0]["lr"]}
        if use_val:
            model.eval()
            with torch.no_grad():
                logits = model(Xv)
                # Plain cross-entropy (no smoothing) so the value is a true NLL.
                val_nll = F.cross_entropy(logits, yv).item()
                val_acc = (logits.argmax(1) == yv).float().mean().item()
            row.update(val_nll=val_nll, val_acc=val_acc)
            sched.step(val_nll)
            if val_nll < best_nll - 1e-4:
                best_nll, best_epoch, stale = val_nll, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        else:
            sched.step(train_loss)
            best_epoch = epoch
        history.append(row)
        if use_val and stale >= patience:
            logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history), best_epoch


def plot_history(history: pd.DataFrame, best_epoch: int, path) -> None:
    """Save the training/validation curves."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], color="#0072B2", label="Train loss (smoothed labels, augmented)")
    ax.plot(history["epoch"], history["val_nll"], color="#E69F00", label="Validation NLL")
    ax.axvline(best_epoch, color="#8A8A8A", linestyle="--", label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MLP training curve")
    ax.grid(color="#E5E5E5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train and calibrate the MLP.")
    parser.add_argument("--epochs", type=int, default=config.MLP_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.MLP_EARLY_STOP_PATIENCE)
    parser.add_argument("--no-augment", action="store_true", help="disable symptom-dropout augmentation")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)
    setup_logging()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    symptoms, classes = load_vocab()
    X_tr, y_tr = load_split("train")
    X_va, y_va = load_split("val")
    logger.info("train=%d val=%d | features=%d classes=%d", len(y_tr), len(y_va), len(symptoms), len(classes))

    start = time.time()
    model, history, best_epoch = train_model(
        X_tr, y_tr, len(classes), X_va, y_va, seed=args.seed,
        augment=not args.no_augment, max_epochs=args.epochs, patience=args.patience,
    )
    logger.info("Training finished in %.1fs (best epoch %d)", time.time() - start, best_epoch)

    # Temperature scaling on validation logits (see src/calibration.py).
    val_logits = mlp_logits(model, X_va)
    temperature = fit_temperature(val_logits, y_va)
    summary = calibration_summary(val_logits, y_va, temperature)
    logger.info("Validation accuracy %.3f | T=%.3f | ECE %.3f -> %.3f | NLL %.3f -> %.3f",
                summary["accuracy"], temperature, summary["ece_before"], summary["ece_after"],
                summary["nll_before"], summary["nll_after"])

    save_mlp(model, config.MODELS_DIR / "mlp.pt", temperature,
             extra={"best_epoch": best_epoch, "seed": args.seed, "augment": not args.no_augment})
    history.to_csv(config.REPORTS_DIR / "mlp_history.csv", index=False)
    save_json(summary, config.REPORTS_DIR / "mlp_val_calibration.json")
    plot_history(history, best_epoch, config.FIGURES_DIR / "mlp_training_curve.png")
    logger.info("Saved models/mlp.pt, reports/mlp_history.csv and the training curve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
