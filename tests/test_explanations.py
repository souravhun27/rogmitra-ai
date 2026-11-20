"""Tests for the plain-language explanation, counterfactuals and follow-up suggester."""
from __future__ import annotations

import numpy as np

from src.explain import Contribution, counterfactual_table, plain_language_explanation, split_contributions
from src.followup import suggest_followups


def test_split_contributions_sorts_and_ignores_absent_symptoms() -> None:
    attr = np.array([0.30, -0.10, 0.20, 0.50])
    x = np.array([1, 1, 1, 0], dtype=np.float32)  # last symptom absent, so its 0.50 must be ignored
    sup, ag = split_contributions(attr, x, ["a", "b", "c", "d"])
    assert [c.symptom for c in sup] == ["a", "c"]
    assert [c.symptom for c in ag] == ["b"]


def test_plain_language_explanation_names_symptoms_with_signed_values() -> None:
    sup = [Contribution("chills", 0.31), Contribution("high fever", 0.22), Contribution("sweating", 0.18)]
    text, footnote = plain_language_explanation("Malaria", sup, [Contribution("headache", -0.02)])
    assert "Malaria" in text and "chills (+0.31)" in text
    assert "high fever (+0.22) and sweating (+0.18)" in text
    assert "Headache (-0.02) slightly reduced its likelihood." in text
    assert footnote


def test_plain_language_explanation_handles_no_supporting_symptom() -> None:
    text, _ = plain_language_explanation("Flu", [], [])
    assert text.startswith("No single reported symptom")


def test_counterfactuals_report_one_row_per_selected_symptom(toy_world: dict) -> None:
    w = toy_world
    x = np.zeros(len(w["symptoms"]), dtype=np.float32)
    x[[0, 1, 6]] = 1.0  # fever, chills, vomiting -> Malaria pattern
    target = int(np.argmax(w["predict"](x[None])[0]))
    assert w["classes"][target] == "Malaria"
    rows = counterfactual_table(w["predict"], x, w["symptoms"], w["classes"], target)
    assert {r["symptom"] for r in rows} == {"fever", "chills", "vomiting"}
    assert all("target_before" in r for r in rows)


def test_followups_never_suggest_an_already_selected_symptom(toy_world: dict) -> None:
    w = toy_world
    x = np.zeros(len(w["symptoms"]), dtype=np.float32)
    x[0] = 1.0  # fever alone is shared by Flu and Malaria
    out = suggest_followups(w["predict"], x, w["symptoms"], w["classes"], w["freq"], k=3)
    assert 0 < len(out) <= 3
    assert "fever" not in {r["symptom"] for r in out}
    gains = [r["expected_entropy_reduction"] for r in out]
    assert gains == sorted(gains, reverse=True)
    assert all(g >= -1e-9 for g in gains)
    # chills or cough separates Malaria from Flu, so one of them must lead
    assert out[0]["symptom"] in {"chills", "cough", "vomiting", "fatigue"}
