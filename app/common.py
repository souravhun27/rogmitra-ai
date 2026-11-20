"""Shared helpers for all Streamlit pages (page setup, cached loading, session state)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # lets `import src...` work however Streamlit was launched
    sys.path.insert(0, str(ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:  # lets `import theme` work for pages and tests alike
    sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402

from src import config  # noqa: E402

import theme  # noqa: E402  (app/theme.py; pages add the app folder to sys.path)

DISCLAIMER = (
    "**Academic project - not a medical device and not a diagnosis.** "
    "RogMitra.AI demonstrates explainable AI on a public dataset. "
    "For any health concern please consult a qualified doctor; in an emergency call your local emergency number."
)


def current_mode() -> str:
    """'dark' or 'light' (kept in session state so it survives page switches)."""
    return st.session_state.get("theme_mode", "dark")


def _set_mode(mode: str) -> None:
    """Button callback of the sidebar Light / Dark switch."""
    st.session_state["theme_mode"] = mode


def _sidebar(active: str) -> None:
    """Brand, navigation, educational note and the Light / Dark switch."""
    with st.sidebar:
        theme.html_block(f'<div class="edx-brand"><div class="edx-logo">{theme.icon("plus", 20)}</div>'
                         '<span class="edx-brandname">RogMitra<b>.AI</b></span></div>')
        for key, page, label, icon_name in theme.NAV:
            with st.container(key=f"nav_{key}"):
                try:
                    st.page_link(page, label=label, icon=icon_name)
                except Exception:  # noqa: BLE001 - a page run on its own (tests) cannot resolve its siblings
                    st.caption(label)
        theme.html_block('<div class="edx-sidenote"><b>Academic project.</b> Not a medical device and not a '
                         'diagnosis. See a doctor for health concerns.</div>')
        mode = current_mode()
        with st.container(key="themebar"):
            light, dark = st.columns(2)
            light.button("☀  Light", key="tm_light", on_click=_set_mode, args=("light",),
                         type="primary" if mode == "light" else "secondary", **STRETCH)
            dark.button("☾  Dark", key="tm_dark", on_click=_set_mode, args=("dark",),
                        type="primary" if mode == "dark" else "secondary", **STRETCH)


def setup_page(title: str, active: str = "check") -> None:
    """Must be the FIRST Streamlit call of every page: page config, theme, sidebar (which carries the short disclaimer note)."""
    st.set_page_config(page_title=f"RogMitra.AI - {title}", page_icon="🩺", layout="wide")
    theme.inject_css(current_mode(), active)
    _sidebar(active)


@st.cache_resource(show_spinner="Loading model...")
def _load_engine():
    """Load the trained model once per server process (training never happens in the app)."""
    from src.engine import DiagnosisEngine

    return DiagnosisEngine()


def get_engine():
    """Return the cached engine, or show setup instructions and stop the page."""
    try:
        return _load_engine()
    except FileNotFoundError as err:
        st.error(
            "The trained model was not found. Train it first (the app never trains):\n\n"
            "```bash\npython -m src.preprocess\npython -m src.train_baselines\npython -m src.train_mlp\n```\n"
            f"Details: {err}"
        )
        st.stop()


@st.cache_resource(show_spinner="Loading text model...")
def get_text_predictor():
    """The optional DistilBERT predictor (cached); callers must check ``text_model_available()`` first."""
    from src.text_model import TextPredictor

    return TextPredictor()


@st.cache_data
def load_remedies() -> dict[str, dict]:
    """Remedy entries keyed by disease name (empty dict if the file is missing)."""
    if not config.REMEDIES_PATH.exists():
        return {}
    from src.check_remedies import load_entries

    return {e["disease"]: e for e in load_entries()}


def get_patient() -> dict:
    """Patient input kept in session state (a plain dict, so it survives page switches)."""
    return st.session_state.setdefault(
        "patient", {"symptoms": [], "free_text": "", "age": None, "sex": "Prefer not to say"}
    )


def pretty(symptom: str) -> str:
    """Display form of a symptom name."""
    return symptom[:1].upper() + symptom[1:]


def _stretch_kwargs() -> dict:
    """Full-width kwarg that works on old and new Streamlit (use_container_width was deprecated in 1.50+)."""
    major, minor = (int(p) for p in st.__version__.split(".")[:2])
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}


STRETCH = _stretch_kwargs()


def home_link(label: str, icon: str = "🩺") -> None:
    """Link back to the Check Symptoms page; degrades to plain text if the page cannot be resolved."""
    try:
        st.page_link("Home.py", label=label, icon=icon)
    except Exception:  # noqa: BLE001 - e.g. page run on its own in tests
        st.caption(f"{icon} {label}: open the Home page from the sidebar.")
