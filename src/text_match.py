"""Rule-based mapping of free text to the model's symptom vocabulary.

The structured model only understands its ~130 symptom names, so for the
optional free-text box we look for those names (and a few everyday synonyms)
in what the user typed. No machine learning and no LLM: just transparent
matching that the user can see and correct, because the app shows what was
detected before predicting. (Stage 5 adds an optional DistilBERT text model.)
"""
from __future__ import annotations

import re

from src.redflags import _is_negated

# Everyday wording -> dataset symptom name. Entries whose target is not in the
# loaded vocabulary are ignored, so this stays safe if the dataset changes.
SYNONYMS: dict[str, str] = {
    r"\bfever\b": "high fever",
    r"\btired(ness)?\b|\bexhausted\b|\bno energy\b": "fatigue",
    r"\bthrow(ing)? up\b|\bthrew up\b|\bpuk(e|ing)\b": "vomiting",
    r"\bfeel(ing)? sick\b|\bqueasy\b": "nausea",
    r"\bdiarrh?oea\b|\bloose (stools?|motions?)\b": "diarrhoea",
    r"\bstomach ?ache\b|\btummy ?ache\b|\bstomach pain\b": "abdominal pain",
    r"\bsore throat\b": "throat irritation",
    r"\bstuffy nose\b|\bblocked nose\b": "congestion",
    r"\brash\b": "skin rash",
    r"\bitchy\b|\bitch(ing)?\b": "itching",
    r"\bshort(ness)? of breath\b|\bbreathless\b": "breathlessness",
    r"\bdizzy\b|\bdizziness\b|\blight ?headed\b": "dizziness",
    r"\bsweat(ing|y)?\b": "sweating",
    r"\bpee(ing)? (a lot|often)\b|\bfrequent urination\b": "polyuria",
    r"\bburning (when|while) (peeing|urinating)\b|\bburning urination\b": "burning micturition",
    r"\bweight loss\b|\blosing weight\b": "weight loss",
    r"\bno appetite\b|\bnot hungry\b": "loss of appetite",
    r"\bback ache\b": "back pain",
    r"\bsneezing\b": "continuous sneezing",
    r"\bshiver(ing)?\b": "shivering",
}


def _stem(word: str) -> str:
    """Very light stemming so 'vomiting'/'vomit' and 'pains'/'pain' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def extract_symptoms(text: str, symptoms: list[str]) -> list[str]:
    """Return the vocabulary symptoms mentioned (and not negated) in ``text``.

    Matching rules, in order: exact phrase; synonym regex; for multi-word
    symptoms, every word present in the same sentence. Negated mentions
    ("no cough") are skipped. The result keeps vocabulary order.
    """
    if not text.strip():
        return []
    vocab = set(symptoms)
    low = text.lower()
    found: set[str] = set()

    for symptom in symptoms:
        for match in re.finditer(rf"\b{re.escape(symptom)}", low):
            if not _is_negated(low, match.start()):
                found.add(symptom)
                break

    for pattern, target in SYNONYMS.items():
        if target in vocab:
            for match in re.finditer(pattern, low):
                if not _is_negated(low, match.start()):
                    found.add(target)
                    break

    for sentence in re.split(r"[.;!?\n]", low):
        stems = {_stem(w) for w in re.findall(r"[a-z]+", sentence)}
        if not stems:
            continue
        for symptom in symptoms:
            words = [w for w in re.findall(r"[a-z]+", symptom) if len(w) > 2]
            if len(words) >= 2 and all(_stem(w) in stems for w in words):
                if not _is_negated(sentence, sentence.find(words[0])):
                    found.add(symptom)
    return [s for s in symptoms if s in found]
