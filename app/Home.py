"""Page 1: Check Symptoms (entry point: ``streamlit run app/Home.py``)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import theme  # noqa: E402
from common import get_engine, get_patient, pretty, setup_page  # noqa: E402
from src.redflags import check_red_flags  # noqa: E402
from src.text_match import extract_symptoms  # noqa: E402

setup_page("Check Symptoms", "check")
engine = get_engine()
patient = get_patient()

theme.html_block(theme.hero_html())

with st.container(key="card_form"):
    theme.html_block('<h2 style="margin:0 0 .2rem 0;font-size:1.9rem">Check Your Symptoms</h2>'
                     '<div class="edx-sub" style="margin:0 0 .6rem 0">Select the symptoms you are experiencing '
                     '(or type them in your own words)</div>')
    left, right = st.columns([2.5, 1], gap="large")
    with left:
        selected = st.multiselect(
            "Symptoms (type to search)",
            options=engine.symptoms,
            default=[s for s in patient["symptoms"] if s in engine.symptoms],
            format_func=pretty,
            placeholder="Search and select symptoms...",
            label_visibility="collapsed",
        )
        free_text = st.text_area(
            "Or describe your symptoms in free text (optional)",
            value=patient["free_text"],
            placeholder="e.g. I have high fever with chills and body ache since 2 days",
            help="We look for known symptom names in your text (simple rule-based matching, no AI chatbot).",
            height=96,
        )
        detected = [s for s in extract_symptoms(free_text, engine.symptoms) if s not in selected]
        if detected:
            theme.html_block(theme.notice("info", "Found in your text and will be added: <b>"
                                          + ", ".join(theme.esc(pretty(s)) for s in detected) + "</b>"))
        c_age, c_sex = st.columns(2)
        age = c_age.number_input("Age (optional)", min_value=0, max_value=120, value=patient["age"], step=1, placeholder="e.g. 22",
                                 help="Shown in your report only. The model does not use it.")
        sexes = ["Prefer not to say", "Female", "Male", "Other"]
        sex = c_sex.selectbox("Sex (optional)", sexes,
                              index=sexes.index(patient["sex"]) if patient["sex"] in sexes else 0,
                              help="Shown in your report only. The model does not use it.")
        if free_text and check_red_flags([], free_text):
            theme.html_block(theme.notice("bad", "Your text mentions something that can be an emergency. Press "
                                          "<b>Predict Disease</b> to see safety advice, or call your local emergency number now."))
        go = st.button("Predict Disease", type="primary", icon=":material/monitor_heart:")
    with right:
        theme.html_block(theme.checklist_html(["Top-3 likely diseases", "Why this prediction?",
                                               "Uncertainty & risk alerts", "Precautions & home care"]))

if go:
    combined = list(dict.fromkeys(selected + detected))
    if not combined and not check_red_flags([], free_text):
        theme.html_block(theme.notice("bad", "Please select at least one symptom (or describe it in the text box)."))
    else:
        st.session_state["patient"] = {"symptoms": combined, "free_text": free_text,
                                       "age": int(age) if age else None, "sex": sex}
        st.switch_page("pages/2_Results.py")

st.caption("Tip: the more symptoms you list, the more reliable the answer. With very few symptoms the model "
           "will tell you it is unsure.")
