"""Visual theme of the RogMitra.AI web app: dark / light palettes, CSS and small HTML components.

Streamlit has no runtime theme switch, so the look is produced here: ``inject_css(mode)`` writes one
style block (CSS variables for the palette + overrides for Streamlit widgets) and the ``*_html``
helpers build the card-style markup used by every page. All helpers return single-line HTML strings
(no blank lines or indentation) because Markdown would otherwise treat indented lines as code.
"""
from __future__ import annotations

import html
import math
import urllib.parse
from typing import Iterable, Sequence

import streamlit as st

# --------------------------------------------------------------------------- #
# Palettes (every colour used by the app comes from one of these two dicts)
# --------------------------------------------------------------------------- #
PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#100c0c",
        "bg_glow": "radial-gradient(900px 480px at 85% -8%, rgba(236,122,60,.16), transparent 60%), "
                   "radial-gradient(700px 420px at -5% 105%, rgba(180,70,40,.12), transparent 60%)",
        "side": "linear-gradient(180deg,#1b1210 0%,#150d0d 62%,#241210 100%)",
        "card": "#1a1413",
        "card2": "#221a18",
        "input": "#120e0e",
        "border": "rgba(255,236,222,.10)",
        "border2": "rgba(255,236,222,.18)",
        "text": "#f4ece6",
        "muted": "#b9aba2",
        "faint": "#85786f",
        "accent": "#ec7a3c",
        "accent2": "#c9491f",
        "accent_soft": "rgba(236,122,60,.15)",
        "good": "#3ccf76",
        "good_soft": "rgba(60,207,118,.14)",
        "bad": "#f0605d",
        "bad_soft": "rgba(240,96,93,.14)",
        "warn": "#f2b13b",
        "warn_soft": "rgba(242,177,59,.13)",
        "track": "rgba(255,255,255,.09)",
        "shadow": "0 12px 32px rgba(0,0,0,.38)",
        "mount_op": ".9",
    },
    "light": {
        "bg": "#f8f3ee",
        "bg_glow": "radial-gradient(900px 480px at 85% -8%, rgba(236,122,60,.13), transparent 60%)",
        "side": "linear-gradient(180deg,#fffaf5 0%,#f7ece2 70%,#efd9c9 100%)",
        "card": "#ffffff",
        "card2": "#faf4ee",
        "input": "#fffdfb",
        "border": "rgba(90,50,25,.13)",
        "border2": "rgba(90,50,25,.24)",
        "text": "#2b1d17",
        "muted": "#6b5a50",
        "faint": "#9a887c",
        "accent": "#d9602a",
        "accent2": "#b8441b",
        "accent_soft": "rgba(217,96,42,.11)",
        "good": "#1d9a52",
        "good_soft": "rgba(29,154,82,.12)",
        "bad": "#d23b3b",
        "bad_soft": "rgba(210,59,59,.11)",
        "warn": "#b97a0a",
        "warn_soft": "rgba(185,122,10,.12)",
        "track": "rgba(90,50,25,.11)",
        "shadow": "0 8px 24px rgba(120,70,40,.10)",
        "mount_op": ".28",
    },
}

NAV = [
    ("check", "Home.py", "Check Symptoms", ":material/monitor_heart:"),
    ("results", "pages/2_Results.py", "Results", ":material/fact_check:"),
    ("performance", "pages/3_Model_Performance.py", "Model Performance", ":material/bar_chart:"),
    ("about", "pages/4_About_and_Limitations.py", "About & Limitations", ":material/info:"),
]

# --------------------------------------------------------------------------- #
# Inline SVG icons (24x24, stroke = currentColor)
# --------------------------------------------------------------------------- #
_ICON_PATHS = {
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "check-circle": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "minus": '<path d="M5 12h14"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    "alert": '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4M12 17h.01"/>',
    "doc": '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5z"/><path d="M14 2v6h6M8 13h8M8 17h8"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.5 19 2c1 2 2 4.2 2 8 0 5.5-4.8 10-10 10z"/><path d="M2 21c0-3 1.9-5.4 5.1-6.9"/>',
    "bars": '<path d="M3 3v18h18"/><path d="M7 16v-5M12 16V6M17 16v-8"/>',
    "bulb": '<path d="M9 18h6M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.3h6c0-1 .4-1.8 1-2.3A7 7 0 0 0 12 2z"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/>',
    "refresh": '<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    "star": '<path d="m12 2 3.1 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.8 21l1.2-6.8-5-4.9 6.9-1z"/>',
    "db": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5M3 12c0 1.7 4 3 9 3s9-1.3 9-3"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/>',
    "scale": '<path d="M12 3v18M5 21h14M5 7h14"/><path d="m5 7-3 7a3 3 0 0 0 6 0zM19 7l-3 7a3 3 0 0 0 6 0z"/>',
    "lock": '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "flow": '<rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="15" width="6" height="6" rx="1"/><path d="M9 6h4a2 2 0 0 1 2 2v7"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/>',
}


def icon(name: str, size: int = 18, cls: str = "") -> str:
    """Inline stroke icon (inherits the surrounding text colour)."""
    body = _ICON_PATHS.get(name, _ICON_PATHS["info"])
    return (f'<svg class="edx-ico {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


def esc(text: object) -> str:
    """HTML-escape any value."""
    return html.escape(str(text), quote=True)


# --------------------------------------------------------------------------- #
# Artwork: hero banner (dusk mountains, sun, caduceus) and body silhouette
# --------------------------------------------------------------------------- #
_HERO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 320' preserveAspectRatio='xMaxYMid slice'>"
    "<defs><linearGradient id='s' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#1a1316'/>"
    "<stop offset='.5' stop-color='#47261f'/><stop offset='1' stop-color='#c35f2c'/></linearGradient>"
    "<radialGradient id='g' cx='.5' cy='.5' r='.5'><stop offset='0' stop-color='#ffb36b' stop-opacity='.95'/>"
    "<stop offset='1' stop-color='#ff7a2f' stop-opacity='0'/></radialGradient></defs>"
    "<rect width='1200' height='320' fill='url(#s)'/>"
    "<circle cx='1030' cy='235' r='150' fill='url(#g)'/><circle cx='1030' cy='235' r='40' fill='#ffd9a5'/>"
    "<polygon points='420,320 640,200 760,250 880,175 1000,255 1100,185 1200,250 1200,320' fill='#3a1f1d'/>"
    "<polygon points='380,320 560,215 660,262 780,200 900,270 1030,215 1200,290 1200,320' fill='#251516'/>"
    "<polygon points='0,320 0,285 140,250 260,285 420,240 600,300 760,270 900,305 1200,285 1200,320' fill='#170e0f'/>"
    "<g fill='none' stroke='#efe3d6' stroke-linecap='round' stroke-width='5' opacity='.95'>"
    "<path d='M1010 28 V262'/>"
    "<path d='M1010 62 C960 40 928 52 906 40 C920 70 960 76 1010 74'/>"
    "<path d='M1010 62 C1060 40 1092 52 1114 40 C1100 70 1060 76 1010 74'/>"
    "<path d='M1010 96 C960 112 960 140 1010 152 C1060 164 1060 192 1010 206' stroke-width='4'/>"
    "<path d='M1010 96 C1060 112 1060 140 1010 152 C960 164 960 192 1010 206' stroke-width='4'/></g>"
    "<circle cx='1010' cy='24' r='9' fill='#efe3d6'/></svg>"
)
HERO_URI = "url(\"data:image/svg+xml;utf8," + urllib.parse.quote(_HERO_SVG, safe="/:=,;'() ") + "\")"

_MOUNTAINS_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 160' preserveAspectRatio='xMidYMax slice'>"
    "<polygon points='0,160 0,110 60,70 100,100 150,50 210,105 255,75 300,110 300,160' fill='#7a3a24' fill-opacity='.35'/>"
    "<polygon points='0,160 0,130 70,100 130,128 190,92 250,125 300,105 300,160' fill='#0f0909' fill-opacity='.65'/></svg>"
)
MOUNTAINS_URI = "url(\"data:image/svg+xml;utf8," + urllib.parse.quote(_MOUNTAINS_SVG, safe="/:=,;'() ") + "\")"


def body_svg() -> str:
    """Human silhouette with glowing lungs (decorative, shown beside the symptom form)."""
    return (
        '<svg class="edx-body" viewBox="0 0 180 260" xmlns="http://www.w3.org/2000/svg">'
        '<defs><radialGradient id="lg" cx=".5" cy=".4" r=".7"><stop offset="0" stop-color="#ffb06a"/>'
        '<stop offset="1" stop-color="#d4501f"/></radialGradient>'
        '<radialGradient id="gl" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#ec7a3c" stop-opacity=".45"/>'
        '<stop offset="1" stop-color="#ec7a3c" stop-opacity="0"/></radialGradient></defs>'
        '<circle cx="90" cy="118" r="86" fill="url(#gl)"/>'
        '<g fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" opacity=".75">'
        '<circle cx="90" cy="34" r="19"/><path d="M82 52v12M98 52v12"/>'
        '<path d="M82 64C54 68 40 78 38 104L34 170M98 64c28 4 42 14 44 40l4 66"/>'
        '<path d="M60 150c-2 30-4 60-2 96M120 150c2 30 4 60 2 96M60 150h60"/></g>'
        '<path d="M84 84c-6-4-16 2-22 18-6 16-8 34-4 44 3 7 12 6 20 0 7-5 8-14 8-26z" fill="url(#lg)" opacity=".92"/>'
        '<path d="M96 84c6-4 16 2 22 18 6 16 8 34 4 44-3 7-12 6-20 0-7-5-8-14-8-26z" fill="url(#lg)" opacity=".92"/>'
        '<g stroke="#ffe3c4" stroke-width="1.6" fill="none" stroke-linecap="round" opacity=".85">'
        '<path d="M90 62v28M90 90c-6 4-10 8-14 14M90 90c6 4 10 8 14 14M76 104l-6 10M104 104l6 10"/></g></svg>'
    )


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');
:root{__VARS__}
html, body, .stApp, [data-testid="stAppViewContainer"]{font-family:Inter,'Segoe UI',system-ui,sans-serif;color:var(--text);}
.stApp{background:var(--bg_glow), var(--bg) !important;background-attachment:fixed !important;}
[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer{display:none !important;}
.block-container, [data-testid="stMainBlockContainer"]{padding-top:1.4rem !important;padding-bottom:3rem !important;max-width:1480px;}
h1,h2,h3,h4{font-family:Fraunces,Georgia,'Times New Roman',serif !important;color:var(--text) !important;letter-spacing:-.01em;}
.stMarkdown p, .stMarkdown li, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{color:var(--text);}
[data-testid="stCaptionContainer"], .stCaption, small{color:var(--muted) !important;}
a{color:var(--accent) !important;}
hr{border-color:var(--border) !important;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:var(--side) !important;border-right:1px solid var(--border);width:272px !important;min-width:272px !important;max-width:272px !important;}
section[data-testid="stSidebar"]::after{content:"";position:absolute;left:0;right:0;bottom:0;height:210px;background:__MOUNTAINS__ bottom/cover no-repeat;pointer-events:none;opacity:var(--mount_op);}
[data-testid="stSidebarNav"]{display:none !important;}
[data-testid="stSidebarHeader"]{padding-bottom:0 !important;min-height:2.2rem;}
[data-testid="stSidebarUserContent"]{padding-top:.4rem !important;padding-left:1.1rem;padding-right:1.1rem;}
section[data-testid="stSidebar"] *{color:var(--text);}
.edx-brand{display:flex;align-items:center;gap:10px;margin:0 0 1.3rem .15rem;}
.edx-logo{width:34px;height:34px;border:2px solid var(--accent);border-radius:9px;display:grid;place-items:center;color:var(--accent);background:var(--accent_soft);}
.edx-brandname{font-family:Fraunces,Georgia,serif;font-size:1.45rem;font-weight:600;color:var(--text);}
.edx-brandname b{color:var(--accent);font-weight:700;}
[class*="st-key-nav_"] a{display:flex !important;align-items:center;gap:.7rem;padding:.62rem .9rem !important;border-radius:11px !important;color:var(--muted) !important;text-decoration:none !important;font-weight:500;border:1px solid transparent;}
[class*="st-key-nav_"] a:hover{background:var(--accent_soft) !important;color:var(--text) !important;}
[class*="st-key-nav_"] a *{color:inherit !important;}
[class*="st-key-nav_"]{margin-bottom:-.35rem;}
.edx-sidenote{margin-top:1rem;padding:.75rem .85rem;border-radius:12px;background:var(--warn_soft);border:1px solid var(--border);font-size:.78rem;line-height:1.35;color:var(--muted);}
.edx-sidenote b{color:var(--warn);}
.st-key-themebar{position:fixed;bottom:16px;left:18px;width:232px;z-index:5;}
.st-key-themebar [data-testid="stHorizontalBlock"]{gap:.35rem !important;background:var(--card2);border:1px solid var(--border);border-radius:999px;padding:3px;}
.st-key-themebar button{border-radius:999px !important;min-height:34px !important;padding:.1rem .5rem !important;font-size:.82rem !important;border:0 !important;background:transparent !important;box-shadow:none !important;color:var(--muted) !important;}
.st-key-themebar button[data-testid="stBaseButton-primary"]{background:var(--accent_soft) !important;color:var(--accent) !important;border:1px solid var(--accent) !important;font-weight:700;box-shadow:none !important;transform:none !important;filter:none !important;}
.st-key-themebar button[data-testid="stBaseButton-primary"] p{color:var(--accent) !important;}
.st-key-themebar button[data-testid="stBaseButton-secondary"] p{color:var(--muted) !important;font-size:.82rem !important;}

/* ---------- cards (st.container(key="card_...")) and html cards ---------- */
[class*="st-key-card"], .edx-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.05rem 1.25rem;box-shadow:var(--shadow);}
[class*="st-key-card"]{gap:.7rem;box-sizing:border-box !important;width:100% !important;}
[class*="st-key-card"] > div, [class*="st-key-card"] > [data-testid="stElementContainer"], [class*="st-key-card"] [data-testid="stMarkdown"], [class*="st-key-card"] [data-testid="stMarkdownContainer"]{width:100% !important;max-width:100% !important;}
.edx-pagehead h1, .edx-hero h1, .edx-card h3, .edx-sec h4, .edx-emergency h2{padding:0 !important;}
.edx-card{margin-bottom:1rem;}
.edx-card h3, .edx-ctitle{font-family:Inter,sans-serif !important;font-size:1rem !important;font-weight:600 !important;margin:0 0 .65rem 0 !important;display:flex;align-items:center;gap:.5rem;color:var(--text) !important;letter-spacing:0;}
.edx-ctitle .edx-ico, .edx-card h3 .edx-ico{color:var(--accent);}
.edx-sub{color:var(--muted);font-size:.84rem;margin:-.35rem 0 .7rem 0;}
.edx-ico{flex:none;vertical-align:-3px;}

/* ---------- hero ---------- */
.edx-hero{position:relative;overflow:hidden;border-radius:20px;border:1px solid var(--border);padding:2.2rem 2.6rem 1.9rem;min-height:230px;background:linear-gradient(90deg,rgba(14,9,9,.88) 0%,rgba(14,9,9,.55) 48%,rgba(14,9,9,0) 78%),__HERO__ center/cover no-repeat;box-shadow:var(--shadow);margin-bottom:1.2rem;}
.edx-hero h1{font-size:3.4rem !important;line-height:1.05;margin:0 0 .35rem 0 !important;color:#f7efe8 !important;}
.edx-hero h1 b{color:#f08a4b;}
.edx-hero p{font-size:1.2rem;color:#efe3d9 !important;margin:0 0 1.1rem 0;}
.edx-chips{display:flex;flex-wrap:wrap;gap:.6rem;}
.edx-chip{border:1px solid rgba(255,255,255,.3);background:rgba(10,6,6,.5);color:#f4ece6;border-radius:999px;padding:.38rem .95rem;font-size:.83rem;backdrop-filter:blur(2px);}
.edx-pagehead h1{font-size:2.2rem !important;margin:0 !important;line-height:1.1;}
.edx-pagehead p{color:var(--muted);margin:.35rem 0 .2rem 0;font-size:.97rem;}

/* ---------- check-symptoms panel ---------- */
.edx-checklist{list-style:none;margin:.6rem 0 0 0;padding:0;}
.edx-checklist li{display:flex;align-items:center;gap:.6rem;padding:.42rem 0;font-size:.9rem;color:var(--text);}
.edx-checklist li .edx-ico{color:var(--accent);}
.edx-bodywrap{display:flex;justify-content:center;color:var(--muted);padding-top:.2rem;}
.edx-body{width:150px;height:auto;}

/* ---------- widgets ---------- */
[data-testid="stWidgetLabel"] p, label p{color:var(--text) !important;font-size:.9rem !important;font-weight:500;}
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], textarea, input{background:var(--input) !important;color:var(--text) !important;border-color:var(--border2) !important;border-radius:11px !important;}
[data-baseweb="textarea"], [data-baseweb="input"], [data-testid="stNumberInputContainer"], [data-testid="stTextAreaRootElement"], [data-testid="stMultiSelect"] [role="group"], [data-testid="stSelectbox"] [role="group"]{background:var(--input) !important;border:1px solid var(--border2) !important;border-radius:11px !important;}
[data-baseweb="select"] > div{min-height:48px;}
[data-testid="stSpinner"], [data-testid="stSpinner"] > div{background:transparent !important;color:var(--muted) !important;}
[data-baseweb="select"] input, [data-baseweb="select"] div{color:var(--text);}
[data-baseweb="select"] svg{fill:var(--muted) !important;}
[data-baseweb="tag"]{background:var(--card2) !important;color:var(--text) !important;border:1px solid var(--border2) !important;border-radius:9px !important;}
[data-baseweb="tag"] span{color:var(--text) !important;}
[data-baseweb="tag"] svg{fill:var(--muted) !important;}
textarea::placeholder, input::placeholder{color:var(--faint) !important;opacity:1;}
[data-baseweb="popover"], [data-baseweb="popover"] div, [data-baseweb="menu"], ul[role="listbox"]{background:var(--card2) !important;color:var(--text) !important;}
[data-baseweb="popover"] > div{border:1px solid var(--border2);border-radius:12px;box-shadow:var(--shadow);}
[role="option"], [role="option"] *{color:var(--text) !important;background:transparent !important;}
[role="option"]:hover, [role="option"][aria-selected="true"], [role="option"]:hover *{background:var(--accent_soft) !important;}
[data-testid="stNumberInput"] button{background:var(--card2) !important;color:var(--text) !important;border-color:var(--border2) !important;}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div, textarea:focus{border-color:var(--accent) !important;box-shadow:0 0 0 1px var(--accent) !important;}

button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,var(--accent),var(--accent2)) !important;color:#fff !important;border:0 !important;border-radius:12px !important;font-weight:600 !important;min-height:46px;padding:.5rem 1.6rem !important;box-shadow:0 8px 22px rgba(201,73,31,.32);transition:transform .08s ease, box-shadow .15s ease;}
button[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);box-shadow:0 12px 26px rgba(201,73,31,.45);filter:brightness(1.06);}
button[data-testid="stBaseButton-primary"] p, button[data-testid="stBaseButton-primary"] span{color:#fff !important;}
button[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button{background:var(--card2) !important;color:var(--text) !important;border:1px solid var(--border2) !important;border-radius:12px !important;min-height:42px;font-weight:500;}
button[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover{border-color:var(--accent) !important;color:var(--accent) !important;}
button[data-testid="stBaseButton-secondary"] p, [data-testid="stDownloadButton"] button p{color:inherit !important;}
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]{min-height:34px;}
[data-testid="stBaseButton-headerNoPadding"], [data-testid="stBaseButton-header"]{color:var(--muted) !important;}

/* tabs as pills */
[data-baseweb="tab-list"]{gap:.5rem !important;background:transparent !important;border-bottom:0 !important;flex-wrap:wrap;margin-bottom:.4rem;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{display:none !important;}
button[data-baseweb="tab"], button[role="tab"]{background:var(--card) !important;border:1px solid var(--border) !important;border-radius:11px !important;padding:.5rem 1.1rem !important;height:auto !important;color:var(--muted) !important;}
button[data-baseweb="tab"] p, button[role="tab"] p{color:inherit !important;font-size:.88rem;font-weight:500;}
button[data-baseweb="tab"][aria-selected="true"], button[role="tab"][aria-selected="true"]{background:linear-gradient(135deg,var(--accent),var(--accent2)) !important;color:#fff !important;border-color:transparent !important;}
button[data-baseweb="tab"][aria-selected="true"] p, button[role="tab"][aria-selected="true"] p{color:#fff !important;font-weight:600;}

[data-testid="stExpander"]{border:1px solid var(--border) !important;border-radius:14px !important;background:var(--card);}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p{color:var(--text) !important;}
[data-testid="stAlert"]{border-radius:12px;color:var(--text);}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:.8rem 1rem;}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p{color:var(--text) !important;}
pre, code{background:var(--card2) !important;color:var(--text) !important;}
[data-testid="stImage"] img{border-radius:10px;}

/* ---------- html components ---------- */
.edx-top{display:flex;align-items:center;gap:1.1rem;flex-wrap:wrap;}
.edx-top .edx-tico{width:58px;height:58px;border-radius:50%;display:grid;place-items:center;color:var(--accent);background:var(--accent_soft);border:1px solid var(--accent);}
.edx-toplabel{font-size:.95rem;color:var(--muted);}
.edx-topname{font-family:Fraunces,Georgia,serif;font-size:1.65rem;font-weight:600;color:var(--accent);margin-left:.35rem;}
.edx-confrow{margin-top:.25rem;font-size:.92rem;color:var(--muted);}
.edx-badge{display:inline-block;padding:.12rem .6rem;border-radius:8px;font-weight:600;font-size:.88rem;border:1px solid transparent;}
.edx-badge.good{background:var(--good_soft);color:var(--good);border-color:var(--good);}
.edx-badge.warn{background:var(--warn_soft);color:var(--warn);border-color:var(--warn);}
.edx-badge.bad{background:var(--bad_soft);color:var(--bad);border-color:var(--bad);}
.edx-pill{margin-left:auto;display:inline-flex;align-items:center;gap:.4rem;padding:.4rem .9rem;border-radius:999px;font-size:.85rem;font-weight:500;border:1px solid var(--warn);color:var(--warn);background:var(--warn_soft);}
.edx-pill.good{border-color:var(--good);color:var(--good);background:var(--good_soft);}
.edx-pill.bad{border-color:var(--bad);color:var(--bad);background:var(--bad_soft);}
.edx-t3{display:grid;grid-template-columns:28px 1fr 52px;align-items:center;gap:.2rem .7rem;margin:.55rem 0 .75rem 0;}
.edx-rank{width:24px;height:24px;border-radius:50%;display:grid;place-items:center;font-size:.76rem;font-weight:700;color:var(--accent);background:var(--accent_soft);border:1px solid var(--accent);}
.edx-t3name{font-size:.95rem;font-weight:600;color:var(--text);margin-bottom:.3rem;}
.edx-track{height:10px;border-radius:99px;background:var(--track);position:relative;}
.edx-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent2),var(--accent));min-width:3px;}
.edx-whisk{position:relative;height:5px;margin-top:5px;}
.edx-whisk i{position:absolute;top:1px;height:3px;border-radius:2px;background:var(--muted);opacity:.6;}
.edx-whisk i::before, .edx-whisk i::after{content:"";position:absolute;top:-2px;width:1px;height:7px;background:var(--muted);}
.edx-whisk i::before{left:0;} .edx-whisk i::after{right:0;}
.edx-pct{text-align:right;font-weight:600;font-size:.95rem;color:var(--text);}
.edx-note{font-size:.76rem;color:var(--muted);margin-top:.2rem;line-height:1.35;}
.edx-why{font-size:.88rem;color:var(--muted);margin:0 0 .6rem 0;line-height:1.45;}
.edx-contrib{display:flex;gap:.6rem;align-items:flex-start;margin:.5rem 0;}
.edx-dot{flex:none;width:22px;height:22px;border-radius:50%;display:grid;place-items:center;}
.edx-dot.plus{background:var(--good_soft);color:var(--good);border:1px solid var(--good);}
.edx-dot.minus{background:var(--bad_soft);color:var(--bad);border:1px solid var(--bad);}
.edx-cname{font-size:.9rem;font-weight:600;color:var(--text);}
.edx-cname span{font-weight:500;color:var(--muted);}
.edx-cdesc{font-size:.78rem;color:var(--muted);line-height:1.35;}
.edx-div{position:relative;margin:.25rem 0 0 0;}
.edx-drow{display:grid;grid-template-columns:104px 1fr;align-items:center;gap:.6rem;margin:.32rem 0;}
.edx-dlab{text-align:right;font-size:.82rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.edx-dtrack{position:relative;height:22px;}
.edx-dtrack::before{content:"";position:absolute;left:var(--z);top:-5px;bottom:-5px;width:1px;background:var(--border2);}
.edx-dbar{position:absolute;top:2px;height:18px;}
.edx-dbar.pos{background:linear-gradient(90deg,#1f9d55,var(--good));border-radius:0 5px 5px 0;}
.edx-dbar.neg{background:linear-gradient(270deg,#c93a3a,var(--bad));border-radius:5px 0 0 5px;}
.edx-dval{position:absolute;top:3px;font-size:.78rem;font-weight:600;color:var(--text);white-space:nowrap;}
.edx-daxis{position:relative;height:18px;margin-left:calc(104px + .6rem);}
.edx-daxis span{position:absolute;transform:translateX(-50%);font-size:.7rem;color:var(--faint);}
.edx-axname{text-align:center;font-size:.72rem;color:var(--faint);margin-left:calc(104px + .6rem);}
.edx-table{width:100%;border-collapse:collapse;font-size:.85rem;}
.edx-table th{text-align:left;color:var(--muted);font-weight:500;font-size:.76rem;padding:.35rem .5rem;border-bottom:1px solid var(--border2);}
.edx-table td{padding:.5rem .5rem;border-bottom:1px solid var(--border);color:var(--text);}
.edx-table tr:last-child td{border-bottom:0;}
.edx-table td, .edx-table th{vertical-align:top;}
.edx-table td.num, .edx-table th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}
.edx-table td.best{color:var(--accent);font-weight:700;}
.edx-neg{color:var(--bad);font-weight:600;}
.edx-pos{color:var(--good);font-weight:600;}
.edx-flat{color:var(--muted);}
.edx-avatar{width:64px;height:64px;border-radius:50%;display:grid;place-items:center;font-size:2rem;background:var(--accent_soft);border:1px solid var(--accent);}
.edx-dxhead{display:flex;align-items:center;gap:.9rem;margin-bottom:.8rem;}
.edx-dxhead .l{font-size:.85rem;color:var(--muted);}
.edx-dxhead .n{font-family:Fraunces,Georgia,serif;font-size:1.7rem;font-weight:600;color:var(--text);line-height:1.1;}
.edx-sec{margin:.9rem 0 .2rem 0;}
.edx-sec h4{font-family:Inter,sans-serif !important;font-size:.95rem !important;font-weight:600 !important;margin:0 0 .35rem 0 !important;display:flex;align-items:center;gap:.5rem;color:var(--accent) !important;}
.edx-sec p, .edx-sec li{font-size:.86rem;color:var(--text);line-height:1.45;}
.edx-list{margin:.2rem 0 0 0;padding-left:1.1rem;}
.edx-list li{margin:.18rem 0;}
.edx-list li::marker{color:var(--accent);}
.edx-doc{margin-top:1rem;padding:.8rem .95rem;border:1px solid var(--accent);border-radius:12px;background:var(--accent_soft);}
.edx-doc h4{color:var(--warn) !important;}
.edx-src{font-size:.78rem;color:var(--muted);margin-top:.6rem;}
.edx-notice{display:flex;gap:.7rem;align-items:flex-start;padding:.8rem 1rem;border-radius:13px;border:1px solid var(--border2);background:var(--card);margin-bottom:1rem;font-size:.9rem;line-height:1.45;color:var(--text);}
.edx-notice .edx-ico{margin-top:2px;}
.edx-notice.warn{border-color:var(--warn);background:var(--warn_soft);}
.edx-notice.warn .edx-ico{color:var(--warn);}
.edx-notice.info{border-color:var(--accent);background:var(--accent_soft);}
.edx-notice.info .edx-ico{color:var(--accent);}
.edx-notice.good{border-color:var(--good);background:var(--good_soft);}
.edx-notice.good .edx-ico{color:var(--good);}
.edx-notice.bad{border-color:var(--bad);background:var(--bad_soft);}
.edx-notice.bad .edx-ico{color:var(--bad);}
.edx-notice ul{margin:.3rem 0 0 1rem;padding:0;}
.edx-emergency{border-radius:16px;padding:1.1rem 1.3rem;margin-bottom:1rem;background:linear-gradient(135deg,#a11414,#d62a2a);color:#fff;box-shadow:0 10px 30px rgba(214,42,42,.35);}
.edx-emergency h2{font-family:Inter,sans-serif !important;color:#fff !important;font-size:1.25rem !important;margin:0 0 .4rem 0 !important;}
.edx-emergency p, .edx-emergency li{color:#fff !important;font-size:.95rem;margin:.2rem 0;}
.edx-metric{display:flex;align-items:center;gap:.9rem;}
.edx-metric .ico{width:52px;height:52px;border-radius:14px;display:grid;place-items:center;color:var(--accent);background:var(--accent_soft);border:1px solid var(--border2);}
.edx-metric .lbl{font-size:.82rem;color:var(--muted);}
.edx-metric .val{font-family:Fraunces,Georgia,serif;font-size:1.9rem;font-weight:600;color:var(--text);line-height:1.1;}
.edx-disclaimer{display:flex;align-items:center;gap:.6rem;padding:.5rem .9rem;border-radius:12px;background:var(--warn_soft);border:1px solid var(--border);font-size:.8rem;color:var(--muted);margin-bottom:1rem;}
.edx-disclaimer b{color:var(--warn);}
.edx-disclaimer .edx-ico{color:var(--warn);}
.edx-flow{display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;}
.edx-step{flex:1 1 150px;border:1px solid var(--border2);border-radius:13px;padding:.75rem .85rem;background:var(--card2);}
.edx-step b{display:block;font-size:.88rem;margin-bottom:.2rem;color:var(--accent);}
.edx-step span{font-size:.78rem;color:var(--muted);line-height:1.35;display:block;}
.edx-arrow{align-self:center;color:var(--faint);}
.edx-kv{display:grid;grid-template-columns:auto 1fr;gap:.35rem 1rem;font-size:.88rem;}
.edx-kv dt{color:var(--muted);} .edx-kv dd{margin:0;color:var(--text);font-weight:500;}
.edx-tags{display:flex;flex-wrap:wrap;gap:.45rem;align-items:center;margin:.2rem 0 1rem 0;}
.edx-tagsl{font-size:.82rem;color:var(--muted);margin-right:.3rem;}
.edx-tag{padding:.22rem .75rem;border-radius:999px;font-size:.8rem;background:var(--card2);border:1px solid var(--border2);color:var(--text);}
.edx-fig{display:block;width:100%;border-radius:10px;}
@media (max-width:640px){
  [class*="st-key-card"] .stButton, [class*="st-key-card"] .stButton button, [class*="st-key-card"] [data-testid="stButton"]{width:100% !important;max-width:100% !important;}
  [class*="st-key-card"] .stButton button{white-space:normal;height:auto;min-height:2.6rem;}
  [data-testid="column"]:has(.edx-metric), [data-testid="stColumn"]:has(.edx-metric){min-width:calc(50% - .5rem) !important;flex:1 1 calc(50% - .5rem) !important;}
  .edx-metric{gap:.6rem;} .edx-metric .ico{transform:scale(.8);margin:-.3rem;} .edx-metric .val{font-size:1.5rem;}
  .edx-table{display:block;overflow-x:auto;-webkit-overflow-scrolling:touch;}
  .block-container{padding-left:1rem !important;padding-right:1rem !important;}
}
"""


def _css_vars(pal: dict[str, str]) -> str:
    """The ``--name:value`` declarations of one palette."""
    return ";".join(f"--{k}:{v}" for k, v in pal.items()) + ";"


def inject_css(mode: str, active: str | None = None) -> None:
    """Write the style block for ``mode`` ('dark' | 'light'); ``active`` highlights one nav item."""
    css = (_CSS.replace("__VARS__", _css_vars(PALETTES[mode]))
           .replace("__HERO__", HERO_URI).replace("__MOUNTAINS__", MOUNTAINS_URI))
    if active:
        css += (f'.st-key-nav_{active} a, .st-key-nav_{active} a:hover{{background:linear-gradient(135deg,var(--accent),var(--accent2)) !important;'
                f'color:#fff !important;box-shadow:0 8px 20px rgba(201,73,31,.3);}}')
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# HTML components
# --------------------------------------------------------------------------- #
def html_block(markup: str) -> None:
    """Render one HTML string (single line) through Markdown."""
    st.markdown(markup.replace("\n", ""), unsafe_allow_html=True)


def card_title(text: str, icon_name: str) -> str:
    """Card heading with an accent icon."""
    return f'<div class="edx-ctitle">{icon(icon_name, 18)}<span>{esc(text)}</span></div>'


def card(inner: str, title: str | None = None, icon_name: str = "info", sub: str | None = None, extra_cls: str = "") -> str:
    """A rounded card as one HTML string."""
    head = card_title(title, icon_name) if title else ""
    subtitle = f'<div class="edx-sub">{sub}</div>' if sub else ""
    return f'<div class="edx-card {extra_cls}">{head}{subtitle}{inner}</div>'


def notice(kind: str, inner: str, icon_name: str | None = None) -> str:
    """Inline notice box. kind: warn | info | good | bad."""
    ico = icon_name or {"warn": "alert", "info": "info", "good": "check-circle", "bad": "alert"}.get(kind, "info")
    return f'<div class="edx-notice {kind}">{icon(ico, 20)}<div>{inner}</div></div>'


def disclaimer_strip() -> str:
    """The permanent one-line disclaimer shown at the top of every page."""
    return ('<div class="edx-disclaimer">' + icon("alert", 18) +
            '<span><b>Academic project &middot; not a medical device &middot; not a diagnosis.</b> '
            'Please consult a qualified doctor for any health concern; in an emergency call your local emergency number.</span></div>')


def page_head(title: str, subtitle: str) -> str:
    """Page title block (serif heading + muted subtitle)."""
    return f'<div class="edx-pagehead"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>'


def hero_html() -> str:
    """Banner of the Check Symptoms page."""
    chips = "".join(f'<span class="edx-chip">{esc(c)}</span>' for c in
                    ("Predicts likely diseases", "Explains why", "Shows uncertainty", "Provides safe care tips"))
    return ('<div class="edx-hero"><h1>RogMitra<b>.AI</b></h1><p>AI Disease Prediction with Explanations</p>'
            f'<div class="edx-chips">{chips}</div></div>')


def checklist_html(items: Iterable[str]) -> str:
    """Tick list shown beside the body illustration."""
    lis = "".join(f"<li>{icon('check', 18)}<span>{esc(i)}</span></li>" for i in items)
    return f'<div class="edx-bodywrap">{body_svg()}</div><ul class="edx-checklist">{lis}</ul>'


def bullet_list(items: Sequence[str]) -> str:
    """Plain bullet list."""
    return '<ul class="edx-list">' + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


def top3_html(rows: Sequence[tuple[str, float, float, float]]) -> str:
    """Probability bars. ``rows`` = (disease, prob, low, high); the grey whisker is the 5-95 % MC-dropout range."""
    out = []
    for i, (name, p, lo, hi) in enumerate(rows, 1):
        out.append(
            f'<div class="edx-t3"><div class="edx-rank">{i}</div><div><div class="edx-t3name">{esc(name)}</div>'
            f'<div class="edx-track" title="{p:.0%} (range {lo:.0%} to {hi:.0%})"><div class="edx-fill" style="width:{max(p, 0.005) * 100:.1f}%"></div></div>'
            f'<div class="edx-whisk"><i style="left:{lo * 100:.1f}%;width:{max((hi - lo) * 100, 0.8):.1f}%"></i></div></div>'
            f'<div class="edx-pct">{p:.0%}</div></div>')
    return "".join(out)


def contribution_chart_html(items: Sequence[tuple[str, float]], axis_title: str = "Contribution to prediction") -> str:
    """Diverging bar chart (green = supports, red = argues against) drawn with HTML so it follows the theme."""
    if not items:
        return '<div class="edx-note">No symptom had a measurable effect.</div>'
    vals = [v for _, v in items]
    hi = max(max(vals), 0.0)
    lo = min(min(vals), 0.0)
    span = max(hi, -lo, 1e-6)
    hi_d = max(hi * 1.28, span * 0.35)
    lo_d = min(lo * 1.28, -span * 0.12) if lo < 0 else -span * 0.12
    total = hi_d - lo_d
    zero = (-lo_d) / total * 100
    rows = []
    for label, v in items:
        w = abs(v) / total * 100
        if v >= 0:
            bar = f'<div class="edx-dbar pos" style="left:{zero:.2f}%;width:{w:.2f}%"></div><div class="edx-dval" style="left:calc({zero + w:.2f}% + 6px)">{v:+.2f}</div>'
        else:
            bar = f'<div class="edx-dbar neg" style="left:{zero - w:.2f}%;width:{w:.2f}%"></div><div class="edx-dval" style="left:calc({zero:.2f}% + 6px)">{v:+.2f}</div>'
        rows.append(f'<div class="edx-drow"><div class="edx-dlab" title="{esc(label)}">{esc(label)}</div>'
                    f'<div class="edx-dtrack" style="--z:{zero:.2f}%">{bar}</div></div>')
    step = _nice_step(total)
    decimals = len(f"{step:.6f}".rstrip("0").split(".")[1]) if "." in f"{step:.6f}".rstrip("0") else 0
    ticks, k = [], math.ceil(lo_d / step - 1e-9)
    while k * step <= hi_d + 1e-9:
        t = k * step
        ticks.append(f'<span style="left:{(t - lo_d) / total * 100:.2f}%">{t:.{decimals}f}</span>')
        k += 1
    return (f'<div class="edx-div">{"".join(rows)}<div class="edx-daxis">{"".join(ticks)}</div>'
            f'<div class="edx-axname">{esc(axis_title)}</div></div>')


def _nice_step(total: float) -> float:
    """Tick spacing (1, 2, 2.5 or 5 times a power of ten) giving at most about six ticks."""
    mag = 10 ** math.floor(math.log10(total / 5))
    for m in (1, 2, 2.5, 5, 10):
        if total / (m * mag) <= 6:
            return m * mag
    return 10 * mag


def html_table(headers: Sequence[str], rows: Sequence[Sequence[str]], numeric_from: int = 1, best_cols: dict[int, set[int]] | None = None) -> str:
    """Styled table. ``rows`` hold already-escaped HTML strings. ``best_cols`` maps column -> set of row indices to highlight."""
    best_cols = best_cols or {}
    head = "".join(f'<th class="{"num" if i >= numeric_from else ""}">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r, row in enumerate(rows):
        cells = ""
        for i, c in enumerate(row):
            cls = ("num " if i >= numeric_from else "") + ("best" if r in best_cols.get(i, ()) else "")
            cells += f'<td class="{cls.strip()}">{c}</td>'
        body += f"<tr>{cells}</tr>"
    return f'<table class="edx-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def metric_card(label: str, value: str, icon_name: str) -> str:
    """KPI tile with an icon."""
    return card(f'<div class="edx-metric"><div class="ico">{icon(icon_name, 26)}</div>'
                f'<div><div class="lbl">{esc(label)}</div><div class="val">{esc(value)}</div></div></div>')


def flow_html(steps: Sequence[tuple[str, str]]) -> str:
    """Horizontal pipeline of (title, description) steps."""
    parts = []
    for i, (t, d) in enumerate(steps):
        parts.append(f'<div class="edx-step"><b>{i + 1}. {esc(t)}</b><span>{esc(d)}</span></div>')
        if i < len(steps) - 1:
            parts.append('<div class="edx-arrow">&#10140;</div>')
    return f'<div class="edx-flow">{"".join(parts)}</div>'


# Emoji avatar of the diagnosed disease (purely decorative; unknown diseases fall back to a stethoscope).
DISEASE_EMOJI = {
    "Malaria": "🦟", "Dengue": "🦟", "Typhoid": "🧫", "Common Cold": "🤧", "Pneumonia": "🫁", "Tuberculosis": "🫁",
    "Bronchial Asthma": "🫁", "Heart attack": "❤️", "Hypertension": "❤️", "Diabetes": "🩸", "Hypoglycemia": "🩸",
    "Migraine": "🤕", "Fungal infection": "🍄", "Allergy": "🤧", "Gastroenteritis": "🤢", "GERD": "🔥",
    "Urinary tract infection": "💧", "Jaundice": "🟡", "Chickenpox": "🔴", "Paralysis (brain hemorrhage)": "🧠",
    "Hypothyroidism": "🦋", "Hyperthyroidism": "🦋", "Acne": "🧴", "Psoriasis": "🧴", "Impetigo": "🧴",
}


def disease_emoji(name: str) -> str:
    """Decorative emoji for a disease name (case-insensitive)."""
    lowered = {k.lower(): v for k, v in DISEASE_EMOJI.items()}
    return lowered.get(name.strip().lower(), "🩺")
