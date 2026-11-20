"""Page 3: Model Performance (loads the files written by ``python -m src.evaluate`` and ``src.explain``)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import charts  # noqa: E402
import theme  # noqa: E402
from common import STRETCH, current_mode, setup_page  # noqa: E402
from src import config  # noqa: E402
from src.models import BASELINE_LABELS  # noqa: E402
from src.utils import load_json  # noqa: E402

setup_page("Model Performance", "performance")
mode = current_mode()
esc = theme.esc

theme.html_block(theme.page_head("Model Performance", "Evaluation results, comparison and analysis."))
theme.html_block(theme.notice(
    "info", "<b>How to read these results.</b> The public dataset is clean and synthetic, so accuracy is close to perfect for every "
            "model. For that reason the evaluation goes further: stratified 5-fold cross-validation, calibration (ECE) and a "
            "missing-symptom robustness test, which show much more about how the models behave than accuracy alone."))

metrics_path = config.REPORTS_DIR / "metrics.json"
if not metrics_path.exists():
    theme.html_block(theme.notice("bad", "No evaluation reports yet. Run <code>python -m src.evaluate</code> first."))
    st.stop()

metrics = load_json(metrics_path)
test, cv, calib = metrics["test"], metrics.get("cv"), metrics.get("calibration")
n_test = metrics.get("n_test", "?")
stamp = charts._stamp()
models = [m for m in ("logreg", "random_forest", "xgboost", "mlp") if m in test]


def image_card(key: str, title: str, icon_name: str, png: bytes | None, caption: str = "") -> None:
    """A card with a heading, a themed chart and an optional caption."""
    with st.container(key=f"card_{key}"):
        theme.html_block(theme.card_title(title, icon_name))
        if png:
            st.image(png, **STRETCH)
        else:
            st.caption("Not available yet: run the corresponding script.")
        if caption:
            theme.html_block(f'<div class="edx-note">{caption}</div>')


def best_rows(col: str) -> set[int]:
    """Row indices of every model that ties for best on a metric (highest; for ECE the lowest), compared at the 3 decimals shown."""
    vals = [round(test[m][col], 3) for m in models]
    target = min(vals) if col == "ece" else max(vals)
    return {i for i, v in enumerate(vals) if v == target}


tabs = st.tabs(["Overview", "Baseline Comparison", "Calibration", "Robustness", "Cross-Validation", "Explainability", "Analysis"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    m = test["mlp"]
    cols = st.columns(4, gap="medium")
    for col, (label, key, ico) in zip(cols, (("Accuracy", "accuracy", "target"), ("Macro F1", "macro_f1", "layers"),
                                            ("Weighted F1", "weighted_f1", "bars"), ("Top-3 Acc.", "top3_accuracy", "star"))):
        col.markdown(theme.metric_card(label, f"{m[key]:.3f}", ico), unsafe_allow_html=True)
    st.caption(f"MLP (temperature-scaled) on the held-out test split (n = {n_test}). Cross-validation over all unique patterns is on the Cross-Validation tab.")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        image_card("conf", "Confusion Matrix (MLP)", "layers", charts.confusion_png(mode, stamp),
                   "Rows = true disease, columns = predicted (numbers are class indices; full names below). A bright diagonal means correct predictions.")
    with c2:
        image_card("rel", "Calibration (Reliability Diagram)", "target", charts.reliability_png(mode, stamp),
                   "Points on the dashed diagonal mean the stated confidence matches how often the model is right.")
    with st.expander("Full-size confusion matrix with disease names"):
        st.image(charts.confusion_png(mode, stamp, True), **STRETCH)

# ---------------------------------------------------------------- Baseline comparison
with tabs[1]:
    head = ["Model", "Accuracy", "Macro F1", "Weighted F1", "Precision", "Recall", "Top-3", "ROC-AUC", "ECE"]
    cols_keys = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "top3_accuracy", "roc_auc_ovr_macro", "ece"]
    rows = [[esc(BASELINE_LABELS[n])] + [f"{test[n][k]:.3f}" for k in cols_keys] for n in models]
    best = {i + 1: best_rows(k) for i, k in enumerate(cols_keys)}
    theme.html_block(theme.card(theme.html_table(head, rows, best_cols=best) +
                                '<div class="edx-note" style="margin-top:.6rem">Orange = best value in the column (lowest for ECE); tied models are all highlighted. '
                                'The MLP row uses temperature-scaled probabilities.</div>',
                                f"Model comparison (test split, n = {n_test})", "layers"))
    image_card("cmp", "Accuracy, Macro F1 and Top-3 by model", "bars", charts.comparison_png(mode, stamp, test))
    text_path = config.REPORTS_DIR / "text_model_metrics.json"
    if text_path.exists():
        tm = load_json(text_path)
        rows_t = [[esc(label)] + [f"{tm['results'][k][c]:.3f}" for c in ("accuracy", "macro_f1", "top3_accuracy", "ece")]
                  for k, label in (("distilbert", "DistilBERT (fine-tuned, calibrated)"), ("tfidf_logreg", "TF-IDF + Logistic Regression"))
                  if k in tm.get("results", {})]
        theme.html_block(theme.card(
            theme.html_table(["Free-text model", "Accuracy", "Macro F1", "Top-3", "ECE"], rows_t) +
            f'<div class="edx-note" style="margin-top:.6rem">Different dataset (Symptom2Disease, {tm.get("n_classes", 24)} diseases, '
            f'{tm.get("n_test", "?")} test texts): not comparable with the table above.</div>',
            "Optional free-text model", "doc"))

# ---------------------------------------------------------------- Calibration
with tabs[2]:
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        image_card("rel2", "Reliability diagram: before vs after temperature scaling", "target", charts.reliability_png(mode, stamp))
    with c2:
        if calib:
            rows = [["Temperature T", f"{calib['temperature']:.3f}", ""],
                    ["ECE (lower is better)", f"{calib['ece_before']:.4f}", f"{calib['ece_after']:.4f}"],
                    ["NLL (lower is better)", f"{calib['nll_before']:.4f}", f"{calib['nll_after']:.4f}"]]
            theme.html_block(theme.card(
                theme.html_table(["Metric", "Before (T = 1)", "After scaling"], rows) +
                '<div class="edx-note" style="margin-top:.6rem">Temperature scaling divides the logits by one number T fitted on the validation set. '
                'It changes confidence, never the predicted disease, so accuracy stays identical.</div>',
                "Calibration summary (test split)", "scale"))
        theme.html_block(theme.card(
            '<ul class="edx-list"><li><b>ECE</b> (Expected Calibration Error) is the average gap between confidence and accuracy over confidence bins.</li>'
            '<li>T below 1 sharpens the probabilities, T above 1 softens them.</li>'
            '<li>The test split is small, so each confidence bin holds few patients; read the curve together with the ECE values.</li></ul>',
            "How to read it", "info"))

# ---------------------------------------------------------------- Robustness
with tabs[3]:
    rob_path = config.REPORTS_DIR / "robustness.csv"
    if rob_path.exists():
        image_card("rob", "Accuracy vs. fraction of symptoms removed", "bars", charts.robustness_png(mode, stamp),
                   "Each test patient loses a random share of symptoms (20 repeats; shaded band = one standard deviation). "
                   "A model that has learned real symptom patterns degrades gracefully; one that memorised full symptom lists does not.")
        rob = pd.read_csv(rob_path)
        piv = rob.pivot(index="model", columns="fraction_removed", values="accuracy_mean")
        rows = [[esc(BASELINE_LABELS[n])] + [f"{piv.loc[n, f]:.1%}" for f in piv.columns] for n in models if n in piv.index]
        theme.html_block(theme.card(theme.html_table(["Model"] + [f"{f:.0%} removed" for f in piv.columns], rows),
                                    "Accuracy at each level of missing symptoms", "layers"))
    else:
        st.caption("robustness.csv not found; run python -m src.evaluate.")

    hard_path = config.REPORTS_DIR / "hard_eval.json"
    if hard_path.exists() and (config.REPORTS_DIR / "hard_eval.csv").exists():
        hard = load_json(hard_path)
        gap, novel, noisy = hard["generalisation_gap"], hard["novel_pattern_accuracy"], hard["noisy_accuracy"]
        n_nov, n_te, k_wrong = hard["n_novel_pattern_test"], hard["n_test"], "2"
        theme.html_block(theme.notice(
            "info", "<b>Is the high accuracy overfitting?</b> Train, validation and test scores are compared below, together with two stricter "
                    f"tests: <b>novel patterns</b> (only the {n_nov} of {n_te} test patients that are not near-copies of a training patient) and "
                    "<b>noisy input</b> (30% of real symptoms removed and wrong symptoms added)."))
        hrows = [[esc(BASELINE_LABELS[n]), f"{gap[n]['train']:.1%}", f"{gap[n]['val']:.1%}", f"{gap[n]['test']:.1%}",
                  f"{novel[n]:.1%}", f"{noisy[n][k_wrong]:.1%}"] for n in models if n in gap]
        theme.html_block(theme.card(theme.html_table(
            ["Model", "Train", "Validation", "Test", f"Novel patterns (n={n_nov})", "Noisy input (+2 wrong)"], hrows),
            "Generalisation gap and stricter tests", "shield"))
        image_card("hard", "Accuracy with missing and wrong symptoms", "bars", charts.hard_png(mode, stamp),
                   "Each test patient loses 30% of the real symptoms and receives 0 to 4 unrelated symptoms (30 repeats; shaded band = one "
                   "standard deviation). Accuracy falling gradually, not suddenly, shows the models rely on the overall symptom pattern. "
                   "The novel-pattern column uses a small number of patients, so read it as an indication, not a precise estimate.")
    else:
        st.caption("Stricter tests not found; run python -m src.evaluate_hard.")

# ---------------------------------------------------------------- Cross-validation
with tabs[4]:
    if cv:
        def pm(n: str, key: str) -> str:
            v = pd.Series(cv[n][key]) * 100
            return f"{v.mean():.1f} &plusmn; {v.std(ddof=0):.1f}"

        rows = [[esc(BASELINE_LABELS[n]), pm(n, "accuracy"), pm(n, "macro_f1")] for n in models if n in cv]
        theme.html_block(theme.card(
            theme.html_table(["Model", "Accuracy % (mean &plusmn; std)", "Macro F1 % (mean &plusmn; std)"], rows) +
            '<div class="edx-note" style="margin-top:.6rem">Stratified 5-fold cross-validation on all de-duplicated patterns. '
            'The standard deviation shows how stable each model is across different splits of the data.</div>',
            f"{config.CV_FOLDS}-fold stratified cross-validation", "layers"))
        image_card("cv", "Cross-validation scores", "bars", charts.cv_png(mode, stamp, cv))
    else:
        st.caption("Cross-validation was skipped (python -m src.evaluate --skip-cv). Re-run without the flag.")

# ---------------------------------------------------------------- Explainability
with tabs[5]:
    theme.html_block(theme.notice("info", "<b>Global explanations.</b> Mean absolute SHAP value per symptom: how strongly each symptom "
                                          "moves the model's output on average. Individual patients are explained on the Results page."))
    sstamp = charts.shap_stamp()
    names = [n for n in ("mlp", "logreg", "random_forest", "xgboost") if (config.REPORTS_DIR / f"shap_{n}_top_symptoms.csv").exists()]
    if names:
        cols = st.columns(2, gap="medium")
        for i, n in enumerate(names):
            with cols[i % 2]:
                image_card(f"shap_{n}", f"Most important symptoms: {BASELINE_LABELS[n]}", "bars", charts.shap_png(mode, sstamp, n))
    else:
        st.caption("No SHAP data yet: run python -m src.explain.")
    sanity = config.REPORTS_DIR / "sanity_check.md"
    if sanity.exists():
        with st.expander("SHAP sanity check against the dataset's disease-symptom map"):
            st.markdown(sanity.read_text(encoding="utf-8"))

# ---------------------------------------------------------------- Analysis
with tabs[6]:
    for title, file in (("Written analysis and failure cases", "analysis.md"), ("Dataset statistics", "dataset_stats.md"),
                        ("Optional: free-text model (DistilBERT)", "text_model.md"),
                        ("Optional: multi-label synthetic experiment", "multilabel_metrics.md")):
        path = config.REPORTS_DIR / file
        if path.exists():
            with st.expander(title, expanded=(file == "analysis.md")):
                st.markdown(path.read_text(encoding="utf-8"))
