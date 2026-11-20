"""Rule-based emergency ("red flag") layer.

This layer is deliberately SIMPLE, transparent and over-cautious. It does not
use the ML model at all: if a dangerous symptom is reported, the app must show
an emergency banner that overrides whatever the model predicts. A false alarm
costs a worry; a missed emergency can cost a life.

Triggers come from two places:
1. selected symptoms (matched against the dataset's symptom names), and
2. free text typed by the user (regular expressions, with simple negation
   handling so "no chest pain" does not fire).

This is an educational heuristic, not a clinical triage tool.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.preprocess import clean_symptom


@dataclass(frozen=True)
class RedFlag:
    """One triggered emergency rule."""

    code: str
    title: str
    message: str
    trigger: str  # what the user entered that caused the alert


# Each rule: dataset symptom names, regex patterns for free text, and optional
# symptom combinations (all names in a combo must be present together).
_RULES: list[dict] = [
    {
        "code": "chest_pain",
        "title": "Possible heart emergency",
        "message": "Chest pain can be a sign of a heart attack. Call your local emergency number now.",
        "symptoms": ["chest pain"],
        "patterns": [r"chest (pain|pressure|tightness)", r"pain in (my )?chest", r"heart attack"],
    },
    {
        "code": "breathing",
        "title": "Difficulty breathing",
        "message": "Severe breathing difficulty needs urgent medical care. Call your local emergency number.",
        "symptoms": ["breathlessness"],
        "patterns": [r"(difficulty|trouble|hard) (in )?breathing", r"can'?t breathe", r"cannot breathe",
                     r"short(ness)? of breath", r"gasping"],
    },
    {
        "code": "consciousness",
        "title": "Loss of consciousness or confusion",
        "message": "Fainting, coma or sudden confusion is an emergency. Call your local emergency number.",
        "symptoms": ["coma", "altered sensorium"],
        "patterns": [r"unconscious", r"passed out", r"fainted", r"faint(ing)?", r"lost consciousness",
                     r"loss of consciousness", r"unresponsive", r"seizure", r"convulsion"],
    },
    {
        "code": "stroke",
        "title": "Possible stroke signs",
        "message": "Weakness on one side, slurred speech or a drooping face can mean a stroke. "
                   "Every minute counts: call your local emergency number immediately.",
        "symptoms": ["weakness of one body side", "slurred speech"],
        "patterns": [r"face (is )?droop", r"drooping face", r"slurred speech", r"can'?t speak",
                     r"weakness (on|in) (one|the left|the right)", r"sudden numbness", r"sudden (severe )?headache",
                     r"stroke"],
    },
    {
        "code": "bleeding",
        "title": "Serious bleeding",
        "message": "Heavy bleeding, vomiting blood or coughing up blood needs emergency care.",
        "symptoms": ["stomach bleeding", "bloody stool", "blood in sputum", "acute liver failure"],
        "patterns": [r"severe bleeding", r"heavy bleeding", r"bleeding (a lot|heavily|won'?t stop)",
                     r"won'?t stop bleeding", r"vomit(ing)? blood", r"coughing (up )?blood", r"blood in (my )?(vomit|stool)"],
    },
    {
        "code": "allergic_reaction",
        "title": "Severe allergic reaction",
        "message": "Swelling of the face, lips or throat with trouble breathing can be anaphylaxis. Call emergency services.",
        "symptoms": [],
        "patterns": [r"anaphyla", r"throat (is )?(swelling|closing)", r"swelling (of|in) (my )?(throat|tongue|lips)"],
    },
    {
        "code": "meningitis",
        "title": "Possible meningitis warning",
        "message": "Fever together with a stiff neck can signal a serious infection. Seek urgent medical care.",
        "symptoms": [],
        "combos": [["stiff neck", "high fever"]],
        "patterns": [r"stiff neck.*fever|fever.*stiff neck"],
    },
    {
        "code": "self_harm",
        "title": "You deserve support right now",
        "message": "If you are thinking about harming yourself, please contact your local emergency number or a "
                   "crisis line now, or reach out to someone you trust and stay with them.",
        "symptoms": [],
        "patterns": [r"suicid", r"kill myself", r"end my life", r"want to die", r"hurt myself", r"harm myself"],
    },
]

_NEGATION = re.compile(r"\b(no|not|without|never|denies|deny|nor|don'?t have|do not have|didn'?t have|free of)\b")


def _is_negated(text: str, start: int) -> bool:
    """True if a negation word appears in the ~30 characters before ``start``."""
    window = text[max(0, start - 30): start]
    # Only look at the current clause so "no cough but chest pain" still fires.
    window = re.split(r"[.,;!?]|\bbut\b|\bhowever\b", window)[-1]
    return bool(_NEGATION.search(window))


def check_red_flags(selected_symptoms: Iterable[str], free_text: str = "") -> list[RedFlag]:
    """Return every red flag triggered by the selected symptoms or the free text.

    One flag per rule at most (the first trigger found is reported).
    """
    selected = {clean_symptom(s) for s in selected_symptoms}
    text = free_text.lower()
    flags: list[RedFlag] = []
    for rule in _RULES:
        trigger: str | None = None
        for name in rule["symptoms"]:
            if name in selected:
                trigger = f"symptom: {name}"
                break
        if trigger is None:
            for combo in rule.get("combos", []):
                if all(c in selected for c in combo):
                    trigger = "symptoms: " + " + ".join(combo)
                    break
        if trigger is None and text:
            for pattern in rule["patterns"]:
                for match in re.finditer(pattern, text):
                    if not _is_negated(text, match.start()):
                        trigger = f'text: "{match.group(0)}"'
                        break
                if trigger:
                    break
        if trigger:
            flags.append(RedFlag(rule["code"], rule["title"], rule["message"], trigger))
    return flags


def emergency_banner(flags: list[RedFlag]) -> str:
    """Plain-text banner for the UI/PDF; empty string when there are no flags."""
    if not flags:
        return ""
    lines = ["EMERGENCY WARNING - seek medical help now."]
    lines += [f"- {f.title}: {f.message}" for f in flags]
    lines.append("This tool is not a medical device. When in doubt, call your local emergency number.")
    return "\n".join(lines)
