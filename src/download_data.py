"""Download (or verify) the Kaggle datasets used by ExplainDx.

Usage::

    python -m src.download_data            # structured dataset only
    python -m src.download_data --text     # also Symptom2Disease (Stage 5)
    python -m src.download_data --check    # only report which files are present

If the Kaggle CLI is not installed/configured this script prints the manual
steps instead of failing silently. The preprocessing script works with files
placed by hand in ``data/raw/``.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from src import config
from src.utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

MANUAL_INSTRUCTIONS = """
MANUAL DOWNLOAD (works without the Kaggle CLI)
  1. Log in at https://www.kaggle.com and open:
       https://www.kaggle.com/datasets/{structured}
       https://www.kaggle.com/datasets/{text}      (only needed for Stage 5)
  2. Click "Download", unzip, and copy these files into: {raw_dir}
       dataset.csv, symptom_Description.csv, symptom_precaution.csv,
       Symptom-severity.csv                (and Symptom2Disease.csv for Stage 5)
  3. Run:  python -m src.download_data --check

KAGGLE CLI SETUP
  Kaggle -> Settings -> API -> "Create New Token" downloads kaggle.json.
  Move it to ~/.kaggle/kaggle.json (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json),
  then rerun:  python -m src.download_data
""".strip()


def find_file(directory: Path, name: str) -> Path | None:
    """Return the file in ``directory`` whose name equals ``name`` ignoring case."""
    if not directory.is_dir():
        return None
    for candidate in directory.iterdir():
        if candidate.name.lower() == name.lower():
            return candidate
    return None


def check_files(raw_dir: Path, names: tuple[str, ...]) -> list[str]:
    """Log presence of each expected file and return the list of missing names."""
    missing: list[str] = []
    for name in names:
        found = find_file(raw_dir, name)
        if found is None:
            missing.append(name)
            logger.warning("MISSING  %s", name)
        else:
            logger.info("found    %s (%d KB)", found.name, found.stat().st_size // 1024)
    return missing


def kaggle_download(slug: str, raw_dir: Path) -> bool:
    """Download and unzip one Kaggle dataset; return True on success."""
    if shutil.which("kaggle") is None:
        logger.error("Kaggle CLI not found (pip install kaggle).")
        return False
    ensure_dir(raw_dir)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir), "--unzip"]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("Kaggle download failed: %s", (result.stderr or result.stdout).strip())
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = all files present)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--text", action="store_true", help="also fetch Symptom2Disease")
    parser.add_argument("--check", action="store_true", help="only verify files, no download")
    args = parser.parse_args(argv)
    setup_logging()

    needed = config.STRUCTURED_RAW_FILES + ((config.RAW_TEXT_FILE,) if args.text else ())

    if not args.check:
        # Skip the network call when everything is already there.
        if check_files(args.raw_dir, config.STRUCTURED_RAW_FILES):
            kaggle_download(config.KAGGLE_SLUGS["structured"], args.raw_dir)
        if args.text and find_file(args.raw_dir, config.RAW_TEXT_FILE) is None:
            kaggle_download(config.KAGGLE_SLUGS["text"], args.raw_dir)

    logger.info("Final check of %s", args.raw_dir)
    missing = check_files(args.raw_dir, needed)
    if missing:
        logger.error("Still missing: %s", ", ".join(missing))
        print(
            MANUAL_INSTRUCTIONS.format(
                structured=config.KAGGLE_SLUGS["structured"],
                text=config.KAGGLE_SLUGS["text"],
                raw_dir=args.raw_dir,
            ),
            file=sys.stderr,
        )
        return 1
    logger.info("All required files present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
