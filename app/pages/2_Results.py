"""Page 2: Results (predictions, uncertainty, explanations, follow-ups, care information)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import theme  # noqa: E402
from common import STRETCH, get_engine, get_patient, get_text_predictor, load_remedies, pretty, setup_page  # noqa: E402
from src.redflags import emergency_banner  # noqa: E402
from src.report_pdf import build_pdf  # noqa: E402
from src.text_model import match_structured, text_model_available  # noqa: E402

setup_page("Results", "results")
engine = get_engine()
patient = get_patient()
esc = theme.esc


def new_check() -> None:
    """Clear the current patient and go back to the form."""
    st.session_state["patient"] = {"symptoms": [], "free_text": "", "age": None, "sex": "Prefer not to say"}
    st.switch_page("Home.py")


def sentences(text: str) -> list[str]:
    """Split a paragraph into sentences (used to show 'see a doctor if' as bullets)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def describe_contribution(symptom: str, disease: str, value: float, biggest: float) -> tuple[str, str]:
    """Short label and sentence for one symptom, from how often the dataset lists it for the diseases."""
    c, j = engine.classes.index(disease), engine.symptoms.index(symptom)
    share = float(engine.freq[c, j])
    if value > 0:
        label = "strong indicator" if value >= 0.5 * biggest else "supports it"
        text = (f"Listed in {share:.0%} of {disease} cases in the data." if share > 0 else
                f"Not typical for {disease} in the data; the model uses it through correlations with other symptoms.")
    else:
        label = "slightly reduces likelihood" if abs(value) < 0.5 * biggest else "reduces likelihood"
        others = [(engine.freq[k, j], engine.classes[k]) for k in range(len(engine.classes)) if k != c]
        q, other = max(others)
        text = (f"More typical of {other} ({q:.0%} of its cases) than of {disease} ({share:.0%})." if q > share else
                f"Common in several conditions, so it does not single out {disease}.")
    return label, text


# --------------------------------------------------------------------------- #
# Run the analysis
# --------------------------------------------------------------------------- #
if not patient["symptoms"] and not patient["free_text"]:
    theme.html_block(theme.page_head("Prediction Results", "No symptoms entered yet."))
    theme.html_block(theme.notice("info", "Go to <b>Check Symptoms</b> and pick the symptoms you have."))
    st.stop()

try:
    result = engine.analyze(patient["symptoms"], patient["free_text"])
except ValueError as err:
    theme.html_block(theme.notice("bad", esc(err)))
    st.stop()

remedy = load_remedies().get(result.top[0].disease) if result.top else None

# --------------------------------------------------------------------------- #
# Header with actions
# --------------------------------------------------------------------------- #
head, b_pdf, b_new = st.columns([5, 1.25, 1.05], vertical_alignment="center")
with head:
    theme.html_block(theme.page_head("Prediction Results",
                                     "Based on your symptoms, here are the most likely conditions with explanations."))
pdf_key = (tuple(result.selected), patient["free_text"], patient["age"], patient["sex"])
if st.session_state.get("_pdf_key") != pdf_key:  # the PDF is rebuilt only when the input changes
    st.session_state["_pdf_key"], st.session_state["_pdf_bytes"] = pdf_key, build_pdf(result, patient, remedy)
b_pdf.download_button("Download PDF", st.session_state["_pdf_bytes"], file_name="rogmitra_report.pdf",
                      mime="application/pdf", icon=":material/download:", **STRETCH)
b_new.button("New Check", icon=":material/refresh:", on_click=new_check, **STRETCH)

# 1) Emergency banner overrides everything else and comes first.
if result.emergency:
    items = "".join(f"<li><b>{esc(f.title)}:</b> {esc(f.message)}</li>" for f in result.red_flags)
    theme.html_block(f'<div class="edx-emergency"><h2>🚨 EMERGENCY WARNING - seek medical help now</h2><ul>{items}</ul>'
                     '<p>This alert comes from fixed safety rules, not from the AI model. This tool is not a medical device; '
                     'when in doubt, call your local emergency number.</p></div>')
if not result.top:
    st.stop()

tags = "".join(f'<span class="edx-tag">{esc(pretty(s))}</span>' for s in result.selected)
theme.html_block(f'<div class="edx-tags"><span class="edx-tagsl">Symptoms used</span>{tags}</div>')
if result.unknown:
    st.caption("Not recognised and ignored: " + ", ".join(result.unknown))

top = result.top[0]

# 2) Uncertainty: when unsure, show the warning instead of a confident answer.
if result.is_uncertain:
    reasons = "".join(f"<li>{esc(r)}</li>" for r in result.uncertainty_reasons)
    theme.html_block(theme.notice("warn", f"<b>{esc(result.uncertainty_message)}</b><ul>{reasons}</ul>", "help"))

width = top.high - top.low
if result.is_uncertain or width >= 0.30:
    level, level_cls = "High", "bad"
elif width >= 0.10:
    level, level_cls = "Moderate", ""
else:
    level, level_cls = "Low", "good"
conf_cls = "warn" if result.is_uncertain or top.prob < 0.7 else "good"

# --------------------------------------------------------------------------- #
# Layout: analysis on the left, diagnosis information on the right
# --------------------------------------------------------------------------- #
main, side = st.columns([2.15, 1], gap="medium")

with main:
    label = "Closest match (low confidence)" if result.is_uncertain else "Top Prediction"
    theme.html_block(theme.card(
        f'<div class="edx-top"><div class="edx-tico">{theme.icon("target", 30)}</div><div>'
        f'<div class="edx-toplabel">{label}:<span class="edx-topname">{esc(top.disease)}</span></div>'
        f'<div class="edx-confrow">Confidence: <span class="edx-badge {conf_cls}">{top.prob:.2f} ({top.prob:.0%})</span>'
        f' &nbsp;range {top.low:.0%} to {top.high:.0%}</div></div>'
        f'<div class="edx-pill {level_cls}" title="Spread of the top probability over 30 Monte-Carlo-Dropout passes">'
        f'{theme.icon("pulse", 16)}Uncertainty: {level}</div></div>'))

    c_top3, c_why = st.columns([1, 1.08], gap="medium")
    with c_top3:
        rows = [(t.disease, t.prob, t.low, t.high) for t in result.top]
        theme.html_block(theme.card(
            theme.top3_html(rows) + '<div class="edx-note">Bar = calibrated probability. Grey whisker = 5th to 95th percentile '
            'over 30 Monte-Carlo-Dropout passes.</div>', "Top-3 Likely Diseases", "layers"))
    with c_why:
        pos, neg = result.supports, result.against
        biggest = max([abs(c.value) for c in pos + neg] or [1.0])
        lis = ""
        for c in pos[:3] + neg[:1]:
            lab, txt = describe_contribution(c.symptom, top.disease, c.value, biggest)
            kind, sym = ("plus", "plus") if c.value > 0 else ("minus", "minus")
            lis += (f'<div class="edx-contrib"><div class="edx-dot {kind}">{theme.icon(sym, 14)}</div><div>'
                    f'<div class="edx-cname">{esc(pretty(c.symptom))} <span>({c.value:+.2f}) - {lab}</span></div>'
                    f'<div class="edx-cdesc">{esc(txt)}</div></div></div>')
        theme.html_block(theme.card(f'<div class="edx-why">{esc(result.explanation_text)}</div>{lis}',
                                    f"Why {top.disease}?", "help"))

    # 4) Contribution chart (Integrated Gradients on the calibrated probability).
    chart_items = [(pretty(c.symptom), c.value) for c in result.supports[:6]] + \
                  [(pretty(c.symptom), c.value) for c in result.against[:3]]
    theme.html_block(theme.card(
        theme.contribution_chart_html(chart_items, f'Contribution to "{top.disease}" prediction') +
        f'<div class="edx-note" style="margin-top:.6rem">{esc(result.explanation_footnote)} Method: Integrated Gradients (Captum) '
        'on the calibrated probability, compared with an empty symptom list.</div>',
        "Symptom Contribution (Integrated Gradients)", "bars",
        sub=f"Green bars support {esc(top.disease)}; red bars argue against it."))

    c_cf, c_next = st.columns([1.35, 1], gap="medium")
    with c_cf:
        if result.counterfactuals:
            body = []
            for cf in result.counterfactuals[:6]:
                before, after = cf["target_before"], cf["target_after"]
                if after is None:
                    change = '<span class="edx-flat">-</span>'
                else:
                    d = (after - before) * 100
                    cls = "edx-neg" if d < -0.5 else ("edx-pos" if d > 0.5 else "edx-flat")
                    change = f'{before:.0%} &rarr; {after:.0%} <span class="{cls}">{(f"{d:+.0f}" if abs(d) >= 0.5 else "0")}</span>'
                new = (f'<b style="color:var(--accent)">{esc(cf["new_top"])}</b>' if cf["changes_prediction"]
                       else f'<span class="edx-flat">unchanged</span>')
                body.append([esc(pretty(cf["symptom"])), new, change])
            theme.html_block(theme.card(
                theme.html_table(["If you did NOT have&hellip;", "New top prediction", f"{esc(top.disease)} (change in pts)"], body,
                                 numeric_from=2), "Counterfactual Analysis", "refresh"))
            with st.expander("Read the counterfactuals as sentences"):
                for cf in result.counterfactuals:
                    st.write("- " + cf["sentence"])
        else:
            theme.html_block(theme.card('<div class="edx-note">No counterfactuals for this input.</div>',
                                        "Counterfactual Analysis", "refresh"))

    # 6) Follow-up questions: clicking adds the symptom and re-predicts.
    useful = [f for f in result.followups if f["expected_entropy_reduction"] > 0.01]
    with c_next:
        with st.container(key="card_next"):
            if useful:
                best = useful[0]
                favour = ", ".join(f"{esc(d)} ({p:.0%})" for d, p in best["favours"]) or "-"
                theme.html_block(theme.card_title("Next Symptom to Ask", "bulb") +
                                 f'<div style="font-size:.95rem;margin-bottom:.3rem">Ask about <b>{esc(pretty(best["symptom"]))}</b>.</div>'
                                 f'<div class="edx-cdesc">It reduces the model\'s uncertainty the most. Typical for: {favour} of cases. '
                                 f'If yes, the model would lean towards <b>{esc(best["top_if_yes"])}</b>.</div>')
                if st.button(f"Add “{pretty(best['symptom'])}” & Re-predict", key=f"fu_{best['symptom']}",
                             icon=":material/add:", **STRETCH):
                    patient["symptoms"] = patient["symptoms"] + [best["symptom"]]
                    st.rerun()
                if len(useful) > 1:
                    st.caption("Other helpful questions")
                    for f in useful[1:3]:
                        if st.button(f"+ {pretty(f['symptom'])}", key=f"fu_{f['symptom']}",
                                     help=f"If yes, the model would lean towards {f['top_if_yes']}.", **STRETCH):
                            patient["symptoms"] = patient["symptoms"] + [f["symptom"]]
                            st.rerun()
            else:
                theme.html_block(theme.card_title("Next Symptom to Ask", "bulb") +
                                 '<div class="edx-cdesc">No further question would change the model\'s answer much.</div>')

    # 5b) Optional second opinion from the free-text DistilBERT model (only if trained and text was typed).
    if patient["free_text"].strip() and text_model_available():
        with st.expander("Second opinion from the free-text model (optional)"):
            try:
                preds = get_text_predictor().predict(patient["free_text"], top_k=3)
                mapped = [match_structured(label_, engine.classes) for label_, _ in preds]
                st.dataframe(pd.DataFrame({"Condition (text model)": [p[0] for p in preds],
                                           "Probability": [f"{p[1]:.0%}" for p in preds]}), hide_index=True, **STRETCH)
                structured_top3 = {t.disease for t in result.top}
                if mapped[0] is None:
                    st.caption("The text model's top answer is not one of the structured model's diseases, so the two cannot be compared.")
                elif mapped[0] in structured_top3:
                    st.success(f"Both models agree: {mapped[0]} is in the structured model's top 3.")
                else:
                    st.warning(f"The models disagree: the text model prefers {mapped[0]}, which is not in the structured model's top 3.")
                st.caption("A separate DistilBERT model fine-tuned on a different dataset (24 diseases). It sees only your typed "
                           "text, not the symptoms you ticked, so disagreement is a reason for caution, not a verdict.")
            except Exception as err:  # noqa: BLE001 - optional feature must never break the page
                st.caption(f"Free-text model unavailable: {err}")

# --------------------------------------------------------------------------- #
# Right column: diagnosis information
# --------------------------------------------------------------------------- #
with side:
    desc = esc(result.description or "No description available.")
    parts = [f'<div class="edx-dxhead"><div class="edx-avatar">{theme.disease_emoji(top.disease)}</div><div>'
             f'<div class="l">{"Closest match" if result.is_uncertain else "Diagnosis"}:</div><div class="n">{esc(top.disease)}</div></div></div>',
             f'<div class="edx-sec"><h4>{theme.icon("doc", 17)}Description</h4><p>{desc}</p></div>']
    if result.precautions:
        parts.append(f'<div class="edx-sec"><h4>{theme.icon("shield", 17)}Precautions</h4>{theme.bullet_list([pretty(p) for p in result.precautions])}</div>')
    if remedy:
        warn = ""
        if not remedy.get("home_care_appropriate", True):
            warn = ('<p style="color:var(--warn);font-size:.82rem"><b>This condition needs professional medical care.</b> '
                    'The tips below only support comfort and are not a treatment.</p>')
        parts.append(f'<div class="edx-sec"><h4>{theme.icon("leaf", 17)}Home Remedies (General Care)</h4>{warn}'
                     f'{theme.bullet_list(remedy["remedies"])}</div>')
        parts.append(f'<div class="edx-doc edx-sec"><h4>{theme.icon("alert", 17)}See a doctor if&hellip;</h4>'
                     f'{theme.bullet_list(sentences(remedy["when_to_see_doctor"]))}</div>')
        src = f'Source: <a href="{esc(remedy["source"])}" target="_blank" rel="noopener">{esc(remedy.get("source_name", "link"))}</a>'
        if not remedy.get("verified"):
            src += ' &middot; summary prepared for this project from the cited page; review pending.'
        parts.append(f'<div class="edx-src">{src}</div>')
    else:
        parts.append('<div class="edx-sec"><h4>' + theme.icon("leaf", 17) + 'Home care</h4>'
                     '<p>No home-care entry for this condition. Please consult a healthcare professional.</p></div>')
    theme.html_block(theme.card("".join(parts)))
