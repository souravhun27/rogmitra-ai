"""Small helpers shared by every ExplainDx script (logging, seeding, JSON I/O)."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from src import config

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once; calling it again is harmless.

    We use logging (not print) so output can be silenced in tests and captured
    in files when a long training run is started from a terminal.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S")


def set_seed(seed: int = config.SEED) -> None:
    """Seed python, numpy and (if installed) torch so runs are reproducible.

    torch is imported lazily because Stage-1 scripts must work without it.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic kernels cost a little speed but make results repeatable.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as pretty UTF-8 JSON, creating parent folders."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Read a JSON file written by :func:`save_json`."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
