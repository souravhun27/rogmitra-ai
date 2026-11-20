"""Full evaluation of baselines and the MLP; everything is saved under ``reports/``.

Run (after ``train_baselines`` and ``train_mlp``)::

    python -m src.evaluate
    python -m src.evaluate --skip-cv          # faster, no 5-fold cross-validation

Outputs
-------
reports/metrics.json, metrics.md          comparison table (test split + CV mean +/- std)
reports/classification_report_mlp.csv     per-class precision / recall / F1 / AUC
reports/robustness.csv, calibration.json
reports/analysis.md                       auto-generated results + failure-case analysis
reports/figures/confusion_matrix_mlp.png, reliability_diagram.png, robustness_curve.png
"""
from __future__ import annotations

import argparse
import logging
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src import config
from src.calibration import (
    calibration_summary, expected_calibration_error, negative_log_likelihood, plot_reliability,
)
from src.eda import disease_jaccard
from src.models import BASELINE_LABELS, BASELINE_NAMES, load_mlp, mlp_logits, softmax
from src.preprocess import load_split, load_vocab, remove_fraction
from src.train_baselines import BASELINE_DIR, fit_baseline
from src.train_mlp import train_model
from src.utils import ensure_dir, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)
Predictor = Callable[[np.ndarray], np.ndarray]  # X -> probabilities (n, n_classes)
COLOURS = {"logreg": "#009E73", "random_forest": "#CC79A7", "xgboost": "#E69F00", "mlp": "#0072B2"}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def per_class_auc(y: np.ndarray, proba: np.ndarray, n_classes: int) -> np.ndarray:
    """One-vs-rest ROC-AUC per class; NaN where the class is absent from ``y``."""
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        pos = y == c
        if pos.any() and (~pos).any():
            out[c] = roc_auc_score(pos.astype(int), proba[:, c])
    return out


def top_k_accuracy(y: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    """Share of samples whose true class is among the k most probable."""
    top = np.argsort(-proba, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def compute_metrics(y: np.ndarray, proba: np.ndarray, n_classes: int) -> dict[str, float]:
    """All headline metrics for one model on one split."""
    pred = proba.argmax(1)
    auc = per_class_auc(y, proba, n_classes)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "top3_accuracy": top_k_accuracy(y, proba, 3),
        "roc_auc_ovr_macro": float(np.nanmean(auc)),
        "ece": expected_calibration_error(proba, y),
        "nll": negative_log_likelihood(proba, y),
    }


# --------------------------------------------------------------------------- #
# Loading predictors
# --------------------------------------------------------------------------- #
def load_predictors() -> tuple[dict[str, Predictor], float, "object"]:
    """Return ``({name: proba_fn}, temperature, mlp_model)`` for every trained model."""
    predictors: dict[str, Predictor] = {}
    for name in BASELINE_NAMES:
        path = BASELINE_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run: python -m src.train_baselines")
        predictors[name] = joblib.load(path).predict_proba
    model, temperature, _ = load_mlp()
    predictors["mlp"] = lambda X: softmax(mlp_logits(model, X), temperature)
    return predictors, temperature, model


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion(y: np.ndarray, pred: np.ndarray, classes: list[str], path: Path) -> Path:
    """Row-normalised confusion matrix (each row sums to 1 = recall per disease)."""
    n = len(classes)
    cm = confusion_matrix(y, pred, labels=range(n)).astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    sns.heatmap(cm, ax=ax, cmap="Blues", vmin=0, vmax=1, square=True, xticklabels=classes,
                yticklabels=classes, cbar_kws={"label": "Share of true class"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("MLP confusion matrix (test split, row-normalised)")
    ax.tick_params(labelsize=6)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", path)
    return path


def robustness_curve(
    predictors: dict[str, Predictor], X: np.ndarray, y: np.ndarray, reps: int, seed: int
) -> pd.DataFrame:
    """Accuracy when a fixed fraction of each patient's symptoms is removed.

    Averaged over ``reps`` random removals (one rep at 0%) because which symptoms
    vanish matters; reports mean and std for every model.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for frac in config.ROBUSTNESS_FRACTIONS:
        n_rep = 1 if frac == 0 else reps
        accs: dict[str, list[float]] = {name: [] for name in predictors}
        for _ in range(n_rep):
            X_f = remove_fraction(X, frac, rng)
            for name, fn in predictors.items():
                accs[name].append(float((fn(X_f).argmax(1) == y).mean()))
        for name, vals in accs.items():
            rows.append({"fraction_removed": frac, "model": name,
                         "accuracy_mean": float(np.mean(vals)), "accuracy_std": float(np.std(vals))})
    return pd.DataFrame(rows)


def plot_robustness(df: pd.DataFrame, path: Path) -> Path:
    """Line chart: accuracy vs fraction of symptoms removed, one line per model."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name in df["model"].unique():
        sub = df[df["model"] == name]
        x = sub["fraction_removed"] * 100
        ax.plot(x, sub["accuracy_mean"] * 100, marker="o", color=COLOURS[name],
                linewidth=2.2 if name == "mlp" else 1.5, label=BASELINE_LABELS[name])
        ax.fill_between(x, (sub["accuracy_mean"] - sub["accuracy_std"]) * 100,
                        (sub["accuracy_mean"] + sub["accuracy_std"]) * 100, color=COLOURS[name], alpha=0.12)
    ax.set_xlabel("Symptoms removed from each patient (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Robustness to missing symptoms")
    ax.grid(color="#E5E5E5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", path)
    return path


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(
    X: np.ndarray, y: np.ndarray, n_classes: int, folds: int, seed: int,
    mlp_epochs: int, aug_copies: int = 5,
) -> tuple[dict[str, dict[str, list[float]]], np.ndarray]:
    """Stratified k-fold CV over ALL de-duplicated rows for every model.

    Returns per-fold metric lists and the MLP's out-of-fold predictions (used for
    the failure analysis because they cover every row, not just the tiny test set).
    The MLP is trained for a fixed number of epochs (the best epoch of the main
    run) because a fold has no spare validation set for early stopping.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    names = list(BASELINE_NAMES) + ["mlp"]
    scores = {n: {"accuracy": [], "macro_f1": [], "top3_accuracy": []} for n in names}
    oof = np.full(len(y), -1, dtype=int)
    with warnings.catch_warnings():
        # Rare diseases have fewer unique rows than folds; sklearn warns, which is expected here.
        warnings.simplefilter("ignore")
        splits = list(skf.split(X, y))
    for k, (tr, te) in enumerate(splits, 1):
        for name in names:
            if name == "mlp":
                model, _, _ = train_model(X[tr], y[tr], n_classes, seed=seed + k, fixed_epochs=mlp_epochs)
                proba = softmax(mlp_logits(model, X[te]))
                oof[te] = proba.argmax(1)
            else:
                proba = fit_baseline(name, X[tr], y[tr], n_classes, aug_copies, seed).predict_proba(X[te])
            pred = proba.argmax(1)
            scores[name]["accuracy"].append(float(accuracy_score(y[te], pred)))
            scores[name]["macro_f1"].append(float(f1_score(y[te], pred, average="macro", zero_division=0)))
            scores[name]["top3_accuracy"].append(top_k_accuracy(y[te], proba, 3))
        logger.info("CV fold %d/%d done", k, folds)
    return scores, oof


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _pm(values: list[float]) -> str:
    return f"{100 * np.mean(values):.1f} ± {100 * np.std(values):.1f}"


def metrics_markdown(test: dict[str, dict], cv: dict | None, n_test: int) -> str:
    """Comparison table (test split, plus CV mean ± std when available), as markdown."""
    head = ["Model", "Accuracy", "Macro F1", "Weighted F1", "Precision", "Recall", "Top-3", "ROC-AUC", "ECE"]
    if cv:
        head += ["CV acc (%)", "CV macro F1 (%)", "CV top-3 (%)"]
    lines = [f"# Model comparison (test split, n = {n_test})", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, m in test.items():
        cells = [BASELINE_LABELS[name]] + [f"{m[k]:.3f}" for k in (
            "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
            "top3_accuracy", "roc_auc_ovr_macro", "ece")]
        if cv:
            cells += [_pm(cv[name]["accuracy"]), _pm(cv[name]["macro_f1"]), _pm(cv[name]["top3_accuracy"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The MLP row uses temperature-scaled probabilities (so its ECE is the calibrated one); "
              "CV rows use the uncalibrated MLP trained for a fixed number of epochs."]
    return "\n".join(lines) + "\n"


def write_analysis(
    path: Path, classes: list[str], symptoms: list[str], test: dict[str, dict], cv: dict | None,
    calib: dict, robust: pd.DataFrame, X_te: np.ndarray, y_te: np.ndarray, p_te: np.ndarray,
    X_full: np.ndarray, y_full: np.ndarray, oof: np.ndarray | None,
) -> None:
    """Generate the written results analysis from the computed numbers.

    Every sentence is derived from data; the script does not invent explanations
    beyond symptom-set overlap. Read it, then edit it into your own words.
    """
    n_te = len(y_te)
    best_base = max(BASELINE_NAMES, key=lambda n: test[n]["accuracy"])
    mlp = test["mlp"]
    r = robust.pivot(index="fraction_removed", columns="model", values="accuracy_mean")
    jac = disease_jaccard(X_full, np.array(classes)[y_full])

    if oof is not None:
        errs = Counter((int(t), int(p)) for t, p in zip(y_full, oof) if t != p)
        source = f"out-of-fold predictions of the MLP over all {len(y_full)} unique rows"
    else:
        pred = p_te.argmax(1)
        errs = Counter((int(t), int(p)) for t, p in zip(y_te, pred) if t != p)
        source = f"the {n_te} test rows"

    L = ["# Results analysis", "",
         "> Auto-generated from `src/evaluate.py`. Numbers are real; review the wording and add your own interpretation.", "",
         "## Headline results", "",
         f"- MLP (calibrated) on the test split (n = {n_te}): accuracy **{mlp['accuracy']:.1%}**, "
         f"macro F1 **{mlp['macro_f1']:.3f}**, top-3 accuracy **{mlp['top3_accuracy']:.1%}**.",
         f"- Best baseline: **{BASELINE_LABELS[best_base]}** with accuracy {test[best_base]['accuracy']:.1%}.",
         f"- With only {n_te} test rows, one mistake moves accuracy by {100 / n_te:.1f} points, so differences "
         "of one or two rows between models are NOT meaningful."]
    if cv:
        L.append("- 5-fold CV accuracy (mean ± std, %): " + ", ".join(
            f"{BASELINE_LABELS[n]} {_pm(cv[n]['accuracy'])}" for n in cv))
    L += ["",
          "## Calibration", "",
          f"Temperature scaling fitted on the validation set gave T = {calib['temperature']:.2f}. "
          f"On the test split ECE changed from {calib['ece_before']:.3f} to {calib['ece_after']:.3f} and NLL from "
          f"{calib['nll_before']:.3f} to {calib['nll_after']:.3f}. Accuracy is unchanged because dividing logits by T "
          "never changes the arg-max. ECE on ~45 samples is noisy; treat it as indicative.", "",
          "## Robustness to missing symptoms", ""]
    for name in r.columns:
        L.append(f"- {BASELINE_LABELS[name]}: {r.loc[0.0, name]:.1%} with all symptoms -> "
                 f"{r.loc[0.5, name]:.1%} when half are removed.")
    L += ["", f"## Failure cases (based on {source})", ""]
    if not errs:
        L.append("No misclassifications were found.")
    else:
        L += ["Most frequent confusions, with the overlap of the two diseases' symptom sets "
              "(Jaccard 0-1; higher = more alike):", "",
              "| True disease | Predicted as | Times | Jaccard | Shared symptoms |", "|---|---|---:|---:|---|"]
        for (t, p), n in errs.most_common(8):
            shared = sorted(set(np.flatnonzero(X_full[y_full == t].sum(0) > 0))
                            & set(np.flatnonzero(X_full[y_full == p].sum(0) > 0)))
            names = ", ".join(symptoms[j] for j in shared[:6]) + (" ..." if len(shared) > 6 else "")
            L.append(f"| {classes[t]} | {classes[p]} | {n} | {jac.loc[classes[t], classes[p]]:.2f} | {names or '-'} |")
        mean_err_j = np.mean([jac.loc[classes[t], classes[p]] for (t, p) in errs])
        off = jac.values[~np.eye(len(classes), dtype=bool)]
        L += ["", f"Average symptom-set Jaccard of confused pairs: **{mean_err_j:.2f}**, versus "
                  f"**{off.mean():.2f}** for a random pair of diseases. If the first number is clearly higher, "
                  "the model confuses diseases mainly because they share symptoms."]
    wrong = np.flatnonzero(p_te.argmax(1) != y_te)
    L += ["", "### Misclassified test rows", ""]
    if len(wrong) == 0:
        L.append("None.")
    for i in wrong:
        present = ", ".join(symptoms[j] for j in np.flatnonzero(X_te[i] > 0))
        L.append(f"- True **{classes[y_te[i]]}**, predicted **{classes[p_te[i].argmax()]}** "
                 f"({p_te[i].max():.0%} confidence). Symptoms: {present}.")
    L += ["", "## Caveats", "",
          "- The dataset is largely synthetic; high scores reflect its simplicity, not clinical reliability.",
          "- After de-duplication each disease has only a handful of distinct rows, and some rows are strict "
          "subsets of others (see `reports/dataset_stats.md`), so test rows can resemble training rows.",
          "- The validation set is used for early stopping AND for temperature scaling; the test split is used only here."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    logger.info("saved %s", path)


def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays for json.dump."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate baselines and MLP.")
    parser.add_argument("--skip-cv", action="store_true", help="skip 5-fold cross-validation")
    parser.add_argument("--folds", type=int, default=config.CV_FOLDS)
    parser.add_argument("--robust-reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)
    setup_logging()
    set_seed(args.seed)

    symptoms, classes = load_vocab()
    n_classes = len(classes)
    X_te, y_te = load_split("test")
    X_full, y_full = load_split("full")
    predictors, temperature, model = load_predictors()
    ensure_dir(config.REPORTS_DIR)

    # 1) headline metrics on the held-out test split
    test = {name: compute_metrics(y_te, fn(X_te), n_classes) for name, fn in predictors.items()}
    for name, m in test.items():
        logger.info("%-20s acc %.3f | macro-F1 %.3f | top-3 %.3f | AUC %.3f | ECE %.3f", BASELINE_LABELS[name],
                    m["accuracy"], m["macro_f1"], m["top3_accuracy"], m["roc_auc_ovr_macro"], m["ece"])
    p_te = predictors["mlp"](X_te)

    # 2) per-class report + confusion matrix (MLP)
    rep = classification_report(y_te, p_te.argmax(1), labels=range(n_classes), target_names=classes,
                                output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    rep_df.loc[classes, "roc_auc_ovr"] = per_class_auc(y_te, p_te, n_classes)
    rep_df.to_csv(config.REPORTS_DIR / "classification_report_mlp.csv")
    plot_confusion(y_te, p_te.argmax(1), classes, config.FIGURES_DIR / "confusion_matrix_mlp.png")

    # 3) calibration before / after temperature scaling
    logits_te = mlp_logits(model, X_te)
    calib = calibration_summary(logits_te, y_te, temperature)
    save_json(calib, config.REPORTS_DIR / "calibration.json")
    plot_reliability(softmax(logits_te), softmax(logits_te, temperature), y_te,
                     config.FIGURES_DIR / "reliability_diagram.png", temperature)

    # 4) robustness curve
    robust = robustness_curve(predictors, X_te, y_te, args.robust_reps, args.seed)
    robust.to_csv(config.REPORTS_DIR / "robustness.csv", index=False)
    plot_robustness(robust, config.FIGURES_DIR / "robustness_curve.png")

    # 5) stratified k-fold cross-validation
    cv, oof = None, None
    if not args.skip_cv:
        best_epoch = int(load_mlp()[2].get("best_epoch", 60))
        cv, oof = cross_validate(X_full, y_full, n_classes, args.folds, args.seed, max(best_epoch, 20))
        for name, s in cv.items():
            logger.info("CV %-20s acc %s | macro-F1 %s", BASELINE_LABELS[name], _pm(s["accuracy"]), _pm(s["macro_f1"]))

    # 6) tables and written analysis
    save_json(to_jsonable({"test": test, "cv": cv, "calibration": calib, "n_test": len(y_te)}),
              config.REPORTS_DIR / "metrics.json")
    (config.REPORTS_DIR / "metrics.md").write_text(metrics_markdown(test, cv, len(y_te)), encoding="utf-8")
    write_analysis(config.REPORTS_DIR / "analysis.md", classes, symptoms, test, cv, calib, robust,
                   X_te, y_te, p_te, X_full, y_full, oof)
    logger.info("Done. See %s", config.REPORTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
