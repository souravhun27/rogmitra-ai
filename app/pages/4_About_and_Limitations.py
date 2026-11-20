"""Page 4: About and Limitations."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import theme  # noqa: E402
from common import setup_page  # noqa: E402
from src import config  # noqa: E402
from src.utils import load_json  # noqa: E402

setup_page("About and Limitations", "about")

theme.html_block(theme.page_head(
    "About RogMitra.AI",
    "An explainable AI system that suggests likely diseases from symptoms and shows why, how sure it is, and what to do next. "
    "Every model is trained from scratch on public data: no pretrained medical API and no language model is used for prediction."))

stats_path = config.REPORTS_DIR / "dataset_stats.json"
stats = load_json(stats_path) if stats_path.exists() else None
n_sym = stats["n_symptoms"] if stats else "about 130"

tabs = st.tabs(["Dataset", "How It Works", "Scope & Limitations", "Ethics", "Disclaimer"])

# ---------------------------------------------------------------- Dataset
with tabs[0]:
    left, right = st.columns(2, gap="medium")
    with left:
        if stats:
            d = stats["dedup"]
            first = (f"{stats['n_diseases']} diseases and {stats['n_symptoms']} symptoms. The {d['n_raw']} raw rows reduce to "
                     f"<b>{d['n_unique']} unique symptom patterns</b> once {stats['duplicate_fraction']:.0%} exact duplicates are removed.")
        else:
            first = "41 diseases and about 130 symptoms (run the preprocessing step to see the exact numbers)."
        items = (
            f'<div class="edx-sec"><h4>{theme.icon("db", 17)}Disease Symptom Prediction</h4><p>Kaggle (Pranay Patil). {first}</p></div>'
            f'<div class="edx-sec"><h4>{theme.icon("doc", 17)}Symptom2Disease</h4><p>Kaggle (Niyar R Barman). Free-text symptom '
            'descriptions for 24 diseases, used by the optional free-text (DistilBERT) model.</p></div>'
            f'<div class="edx-sec"><h4>{theme.icon("leaf", 17)}Home-care information</h4><p>No public dataset exists for this, so '
            '<code>data/remedies.json</code> was curated from NHS, MedlinePlus and Mayo Clinic pages. Every entry lists its source link, '
            'contains only mild, low-risk advice and gives no medicine doses.</p></div>')
        theme.html_block(theme.card(items, "Datasets Used", "db"))
    with right:
        steps = ["Symptom names are cleaned: spacing, underscores and spelling variants are normalised.",
                 "Exact duplicate rows are removed <b>before</b> splitting, so identical patients can never appear in both training and test data (no data leakage).",
                 "A stratified 70 / 15 / 15 split with a fixed seed (42) keeps every disease represented and the results reproducible.",
                 "During training, 10 to 30% of each patient's symptoms are dropped at random in every epoch, so the model learns to cope with incomplete input."]
        if stats:
            sp = stats["split_sizes"]
            steps.append(f"Resulting split: {sp['train']} training, {sp['val']} validation and {sp['test']} test patterns.")
        theme.html_block(theme.card(
            '<ul class="edx-list">' + "".join(f"<li>{s}</li>" for s in steps) + "</ul>"
            '<div class="edx-note" style="margin-top:.7rem">The dataset is clean and idealised, so accuracy is naturally high. '
            'That is why the evaluation also reports cross-validation, calibration and robustness to missing symptoms.</div>',
            "Data Preparation", "shield"))

# ---------------------------------------------------------------- How it works
with tabs[1]:
    theme.html_block(theme.card(theme.flow_html([
        ("Symptoms in", "The user selects symptoms or types text; a rule-based matcher finds known symptom names."),
        ("Red-flag rules", "Fixed safety rules (chest pain, breathing difficulty, stroke signs...) raise an emergency banner that overrides the model."),
        ("MLP model", f"A PyTorch network ({n_sym} symptoms -> 256 -> 128 -> diseases) trained from scratch ranks the diseases."),
        ("Calibration", "Temperature scaling makes the stated percentages match how often the model is right."),
        ("Uncertainty", "30 Monte-Carlo-Dropout passes give a confidence range; low confidence triggers a warning."),
        ("Explanation", "Integrated Gradients, counterfactuals and a follow-up question explain and refine the answer."),
    ]), "From symptoms to explanation", "flow"))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        theme.html_block(theme.card(
            '<ul class="edx-list">'
            '<li><b>Integrated Gradients</b> (per patient): how much each symptom moved this prediction.</li>'
            '<li><b>SHAP</b> (global): which symptoms matter most to each model on average.</li>'
            '<li><b>Counterfactuals</b>: remove one symptom, re-run the model and see what changes.</li>'
            '<li><b>Follow-up question</b>: the symptom whose answer would reduce uncertainty the most.</li>'
            '<li><b>Sanity check</b>: the model\'s top SHAP symptoms are compared with the dataset\'s own disease-symptom map.</li></ul>',
            "Explainability", "bulb"))
    with c2:
        theme.html_block(theme.card(
            '<ul class="edx-list">'
            '<li><b>Temperature scaling</b> is fitted on the validation set and reported with ECE and reliability diagrams.</li>'
            '<li><b>Monte-Carlo Dropout</b> keeps dropout active at prediction time; the spread of 30 runs measures uncertainty.</li>'
            '<li>The system answers "unsure" when the top probability is below 50%, the top two are too close, or fewer than 3 symptoms were given.</li></ul>',
            "Uncertainty and calibration", "pulse"))

# ---------------------------------------------------------------- Scope and limitations
with tabs[2]:
    theme.html_block(theme.notice(
        "info", "<b>Scope of the project.</b> RogMitra.AI is a proof of concept for <b>explainable</b> medical-style prediction. "
                "Each known limitation below is paired with what the project already does about it and how it could be extended."))
    rows = [
        ("Public, idealised dataset",
         "Each disease is a small set of textbook symptom lists, so scores are high for every model.",
         "Duplicates removed before splitting, stratified split, random symptom dropout, 5-fold cross-validation, calibration and a missing-symptom robustness test; results are judged on all of these, not on accuracy alone.",
         "Validate on de-identified real clinical records with ethical approval."),
        ("Limited number of distinct cases",
         "A few hundred unique patterns give a small test split, so a single split is noisy.",
         "Stratified 5-fold cross-validation reports mean &plusmn; standard deviation next to the test-split numbers.",
         "Larger dataset and confidence intervals."),
        ("Symptoms only, fixed list of diseases",
         "No age, history or test results; a disease outside the list cannot be predicted.",
         "The system says it is unsure for low confidence or fewer than 3 symptoms, and always advises seeing a doctor. Age and sex are not model inputs, which avoids unsupported demographic assumptions.",
         "More conditions and clinical context features."),
        ("Symptom presence only",
         "The input cannot express \"definitely absent\".",
         "Counterfactuals and follow-up questions explore the effect of adding or removing symptoms.",
         "Three-state input: present / absent / unknown."),
        ("Explanations describe the model",
         "They show what the network relies on, which is not the same as medical causation.",
         "Global SHAP importances are cross-checked against the dataset's own disease-symptom map, and mismatches are reported in a sanity-check report.",
         "Review of the explanations by a clinician."),
        ("Approximate uncertainty",
         "Temperature scaling and MC Dropout reduce, but do not remove, over-confidence.",
         "Reliability diagrams and ECE before and after calibration are reported on the Model Performance page.",
         "Deep ensembles or conformal prediction."),
        ("Rule-based emergency layer",
         "Keyword rules are a simple safety net and cannot cover every situation.",
         "The layer is conservative, overrides the model output, and handles negation (for example \"no chest pain\").",
         "Clinician-reviewed rule set."),
        ("Curated home-care text",
         "It is a summary written for this project, not clinical guidance.",
         "Each entry cites NHS, MedlinePlus or Mayo Clinic, is limited to mild measures without doses and always includes when to see a doctor.",
         "Review by a medical professional."),
    ]
    body = "".join(f"<tr><td><b>{a}</b><br><span style='color:var(--muted);font-size:.8rem'>{b}</span></td><td>{c}</td><td>{d}</td></tr>"
                   for a, b, c, d in rows)
    theme.html_block(theme.card(
        '<table class="edx-table edx-wide"><thead><tr><th style="width:27%">Scope point</th><th style="width:46%">How the project handles it</th>'
        f'<th>Future work</th></tr></thead><tbody>{body}</tbody></table>', "Scope, limitations and future work", "scale"))

# ---------------------------------------------------------------- Ethics
with tabs[3]:
    items = [
        "<b>Privacy by design.</b> No real patient data was used for training. Anything typed into the app stays in the user's session and is not stored.",
        "<b>Fairness.</b> The dataset carries no demographic information, so a fairness audit across groups is not possible. To avoid unsupported assumptions, "
        "age and sex are not model inputs and appear only in the report. Auditing on data with demographics is listed as future work.",
        "<b>Transparency.</b> Every prediction comes with its reasons, a calibrated probability and an uncertainty range, so the user can see how much to trust it.",
        "<b>Human in the loop.</b> The system never presents itself as a diagnosis. Low confidence triggers a warning, emergency rules override the model, and every result points the user to a doctor.",
        "<b>Intended use.</b> Learning and demonstration of explainable AI; not a replacement for professional medical advice.",
    ]
    theme.html_block(theme.card('<ul class="edx-list">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>",
                                "Ethics, Privacy and Fairness", "shield"))

# ---------------------------------------------------------------- Disclaimer
with tabs[4]:
    theme.html_block(theme.notice(
        "warn", "<b>Academic project. Not a medical device and not a diagnosis.</b><br>RogMitra.AI demonstrates explainable AI on public "
                "datasets. It cannot examine a patient or know their medical history, so its output is informational only. For any health "
                "concern please consult a qualified healthcare professional, and in an emergency call your local emergency number."))
