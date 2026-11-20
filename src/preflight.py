"""Pre-deployment check: are all files the deployed app loads actually present?

Run before pushing to GitHub / Hugging Face::

    python -m src.preflight

The deployed app never trains, so a missing model or report file shows up as an error page
only after deployment. This script finds the problem on your own machine first.
"""
from __future__ import annotations

import argparse
import logging
import subprocess

from src import config
from src.utils import load_json, setup_logging

logger = logging.getLogger(__name__)

REQUIRED = {
    "trained model": config.MODELS_DIR / "mlp.pt",
    "symptom list": config.SYMPTOMS_JSON,
    "class list": config.CLASSES_JSON,
    "disease info": config.DISEASE_INFO_JSON,
    "disease-symptom map": config.DISEASE_SYMPTOM_MAP_JSON,
    "train split": config.TRAIN_CSV,
    "validation split": config.VAL_CSV,
    "test split": config.TEST_CSV,
    "full dataset": config.FULL_DEDUP_CSV,
    "home-care notes": config.DATA_DIR / "remedies.json",
    "metrics": config.REPORTS_DIR / "metrics.json",
    "robustness results": config.REPORTS_DIR / "robustness.csv",
}
OPTIONAL = {
    "stricter tests": config.REPORTS_DIR / "hard_eval.json",
    "SHAP importance (MLP)": config.REPORTS_DIR / "shap_mlp_top_symptoms.csv",
}


def git_ignored(path) -> bool:
    """True when git would ignore ``path`` (so it would be missing from the deployed repo)."""
    try:
        out = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=config.PROJECT_ROOT, capture_output=True)
        return out.returncode == 0
    except (OSError, FileNotFoundError):
        return False  # git not installed or not a repository: nothing to report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 1 if a required file is missing or git-ignored."""
    argparse.ArgumentParser(description="Check deployment files.").parse_args(argv)
    setup_logging()
    problems = 0
    for label, path in REQUIRED.items():
        if not path.exists():
            logger.error("MISSING  %-22s %s", label, path.relative_to(config.PROJECT_ROOT))
            problems += 1
        elif git_ignored(path):
            logger.error("IGNORED  %-22s %s is in .gitignore, so it would not be deployed", label,
                         path.relative_to(config.PROJECT_ROOT))
            problems += 1
        else:
            logger.info("ok       %-22s %s (%.1f KB)", label, path.relative_to(config.PROJECT_ROOT), path.stat().st_size / 1024)
    for label, path in OPTIONAL.items():
        if not path.exists():
            logger.warning("optional %-22s not found (the page section will show a hint instead)", label)
    remedies = config.DATA_DIR / "remedies.json"
    if remedies.exists():
        data = load_json(remedies)
        items = data if isinstance(data, list) else data.get("entries", [])
        flags = [bool(e.get("verified")) for e in items if isinstance(e, dict)]
        logger.info("home-care notes: %d of %d entries marked verified", sum(flags), len(flags))
        if flags and not all(flags):
            logger.warning("Unverified entries are shown as 'review pending'. Open each source link, then set verified to true.")
    if problems:
        logger.error("%d problem(s). Fix them before deploying.", problems)
        return 1
    logger.info("All required files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
