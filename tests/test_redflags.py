"""Tests for the rule-based emergency layer (it must work with or without the model)."""
from __future__ import annotations

import pytest

from src.redflags import check_red_flags, emergency_banner


def codes(flags) -> set[str]:
    return {f.code for f in flags}


def test_selected_chest_pain_triggers_heart_alert() -> None:
    assert "chest_pain" in codes(check_red_flags(["Chest_Pain", "cough"]))


@pytest.mark.parametrize("text,code", [
    ("I have severe chest pain since morning", "chest_pain"),
    ("he can't breathe properly", "breathing"),
    ("she passed out in the kitchen", "consciousness"),
])
def test_free_text_triggers_expected_rule(text: str, code: str) -> None:
    assert code in codes(check_red_flags([], text))


@pytest.mark.parametrize("text", ["no chest pain at all", "I do not have difficulty breathing", "without fainting"])
def test_negated_phrases_do_not_trigger(text: str) -> None:
    assert check_red_flags([], text) == []


def test_negation_only_covers_its_own_clause() -> None:
    # "no cough" must not cancel the chest pain mentioned after "but".
    assert "chest_pain" in codes(check_red_flags([], "no cough but chest pain"))


def test_harmless_symptoms_trigger_nothing_and_banner_is_empty() -> None:
    flags = check_red_flags(["itching", "skin rash"], "mild itchy skin")
    assert flags == []
    assert emergency_banner(flags) == ""


def test_banner_mentions_every_triggered_rule_and_the_disclaimer() -> None:
    flags = check_red_flags(["chest pain", "breathlessness"])
    banner = emergency_banner(flags)
    assert banner.startswith("EMERGENCY WARNING")
    assert all(f.title in banner for f in flags)
    assert "not a medical device" in banner


def test_each_rule_is_reported_at_most_once() -> None:
    flags = check_red_flags(["chest pain"], "chest pain and chest pressure")
    assert len([f for f in flags if f.code == "chest_pain"]) == 1
