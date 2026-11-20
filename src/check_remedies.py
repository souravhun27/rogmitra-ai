"""Validate ``data/remedies.json`` (and optionally check that every source URL opens).

Run::

    python -m src.check_remedies            # offline structure/safety checks
    python -m src.check_remedies --online   # also request every source URL

The offline checks enforce the project rules: required fields, at least 25
diseases, trusted source domains, and NO medicine dosages. The online check
needs internet. It only proves that a page opens; YOU still have to read each
page and compare it with the entry, then set ``"verified": true``.
"""
from __future__ import annotations

import argparse
import logging
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

from src import config
from src.utils import load_json, setup_logging

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("disease", "remedies", "when_to_see_doctor", "source")
TRUSTED_DOMAINS = ("nhs.uk", "medlineplus.gov", "mayoclinic.org")
# A number followed by a dosage-like unit, or the words dose/dosage.
DOSAGE_PATTERN = re.compile(
    r"(\b\d+(\.\d+)?\s*(mg|mcg|µg|g|ml|iu|units?|tablets?|pills?|capsules?|drops?|teaspoons?|tablespoons?|tsp|tbsp)\b)|\bdos(e|es|age)\b",
    re.IGNORECASE,
)


def load_entries() -> list[dict]:
    """Return the list of remedy entries from ``data/remedies.json``."""
    data = load_json(config.REMEDIES_PATH)
    return data["entries"] if isinstance(data, dict) else data


def validate(entries: list[dict], classes: list[str] | None = None, min_entries: int = 25) -> list[str]:
    """Return a list of human-readable problems (empty list = all checks passed)."""
    problems: list[str] = []
    if len(entries) < min_entries:
        problems.append(f"Only {len(entries)} entries; at least {min_entries} are required.")
    seen: set[str] = set()
    for e in entries:
        name = e.get("disease", "<missing name>")
        for field in REQUIRED_FIELDS:
            if field not in e or e[field] in (None, "", []):
                problems.append(f"{name}: missing or empty '{field}'")
        if name in seen:
            problems.append(f"{name}: duplicate entry")
        seen.add(name)
        host = (urlparse(e.get("source", "")).hostname or "").lower()
        if not any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS):
            problems.append(f"{name}: source domain {host!r} is not one of {TRUSTED_DOMAINS}")
        text = " ".join(list(e.get("remedies", [])) + [e.get("when_to_see_doctor", "")])
        for match in DOSAGE_PATTERN.finditer(text):
            problems.append(f"{name}: possible dosage wording {match.group(0)!r}")
    if classes is not None:
        missing = sorted(set(classes) - seen)
        extra = sorted(seen - set(classes))
        if missing:
            problems.append(f"No remedy entry for model classes: {missing}")
        if extra:
            problems.append(f"Entries that match no model class (typo?): {extra}")
    return problems


def check_urls(entries: list[dict], timeout: float = 15.0) -> list[str]:
    """Request each URL; return problems for any that do not answer HTTP 200."""
    problems: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0 (ExplainDx student project link check)"}
    for e in entries:
        url = e["source"]
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
                logger.info("OK   %s  %s", resp.status, url)
        except (urllib.error.URLError, TimeoutError, ValueError) as err:
            problems.append(f"{e['disease']}: {url} -> {err}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 1 when any problem is found."""
    parser = argparse.ArgumentParser(description="Validate data/remedies.json.")
    parser.add_argument("--online", action="store_true", help="also request every source URL")
    args = parser.parse_args(argv)
    setup_logging()
    entries = load_entries()
    classes = load_json(config.CLASSES_JSON) if config.CLASSES_JSON.exists() else None
    problems = validate(entries, classes)
    if args.online:
        problems += check_urls(entries)
    unverified = [e["disease"] for e in entries if not e.get("verified")]
    logger.info("%d entries, %d marked verified, %d still need YOUR verification.",
                len(entries), len(entries) - len(unverified), len(unverified))
    for p in problems:
        logger.error(p)
    if not problems:
        logger.info("All checks passed.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
