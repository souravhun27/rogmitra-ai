"""Fill the README results section from the files in ``reports/`` so it always matches your own run.

Run after ``python -m src.evaluate`` and ``python -m src.evaluate_hard``::

    python -m src.update_readme

Why: typing numbers into a README by hand invites mistakes. This script copies them from the
saved reports and only replaces the text between the METRICS markers.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src import config
from src.models import BASELINE_LABELS
from src.utils import load_json, setup_logging

logger = logging.getLogger(__name__)
START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"
ORDER = ("logreg", "random_forest", "xgboost", "mlp")


def comparison_block() -> str:
    """The comparison table written by ``evaluate.py`` (``reports/metrics.md``)."""
    path = config.REPORTS_DIR / "metrics.md"
    if not path.exists():
        raise FileNotFoundError("reports/metrics.md missing. Run: python -m src.evaluate")
    text = path.read_text(encoding="utf-8").strip()
    return "### " + text[2:] if text.startswith("# ") else text  # keep README heading levels consistent


def hard_block() -> str:
    """Markdown for the generalisation-gap and stricter-test results, or an empty string if not run."""
    path = config.REPORTS_DIR / "hard_eval.json"
    if not path.exists():
        return ""
    h = load_json(path)
    gap, novel, noisy = h["generalisation_gap"], h["novel_pattern_accuracy"], h["noisy_accuracy"]
    lines = [
        f"### Stricter tests (test n = {h['n_test']}; novel-pattern subset n = {h['n_novel_pattern_test']})",
        "",
        "| Model | Train | Validation | Test | Novel patterns only | Noisy input (+2 wrong symptoms) |",
        "|---|---|---|---|---|---|",
    ]
    for name in ORDER:
        if name in gap:
            g = gap[name]
            lines.append(f"| {BASELINE_LABELS[name]} | {g['train']:.1%} | {g['val']:.1%} | {g['test']:.1%} | "
                         f"{novel[name]:.1%} | {noisy[name]['2']:.1%} |")
    lines += ["", f"*Noisy input: {h['noise_removed_fraction']:.0%} of each patient's real symptoms removed and two unrelated "
                  "symptoms added, averaged over repeated random draws. The novel-pattern subset is small, so treat it as an indication.*"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Insert report tables into README.md.")
    parser.add_argument("--readme", type=Path, default=config.PROJECT_ROOT / "README.md")
    args = parser.parse_args(argv)
    setup_logging()
    text = args.readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"Markers {START} / {END} not found in {args.readme}")
    body = comparison_block()
    hard = hard_block()
    if hard:
        body += "\n\n" + hard
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    args.readme.write_text(f"{head}{START}\n{body}\n{END}{tail}", encoding="utf-8")
    logger.info("Updated %s", args.readme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
