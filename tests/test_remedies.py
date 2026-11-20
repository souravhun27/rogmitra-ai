"""Safety checks on the home-care file: complete entries, trusted sources, and no medicine doses."""
from __future__ import annotations

import re

import pytest

from src import config
from src.utils import load_json

PATH = config.DATA_DIR / "remedies.json"
TRUSTED = ("nhs.uk", "medlineplus.gov", "mayoclinic.org")
DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|g|iu)\b|\b(tablet|capsule|dose|dosage)s?\b", re.I)


@pytest.fixture()
def entries() -> list[dict]:
    if not PATH.exists():
        pytest.skip("data/remedies.json not found")
    return load_json(PATH)["entries"]


def test_at_least_25_diseases_are_covered(entries: list[dict]) -> None:
    assert len({e["disease"] for e in entries}) >= 25


def test_every_entry_has_remedies_doctor_note_and_trusted_source(entries: list[dict]) -> None:
    for e in entries:
        assert e["remedies"] and all(isinstance(r, str) and r.strip() for r in e["remedies"]), e["disease"]
        assert e["when_to_see_doctor"].strip(), e["disease"]
        assert e["source"].startswith("https://") and any(d in e["source"] for d in TRUSTED), e["disease"]
        assert isinstance(e.get("verified"), bool), e["disease"]


def test_no_medicine_dosages_appear_in_any_entry(entries: list[dict]) -> None:
    for e in entries:
        for text in [*e["remedies"], e["when_to_see_doctor"]]:
            assert not DOSE.search(text), f"{e['disease']}: {text}"
