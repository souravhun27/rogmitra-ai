"""Clean, de-duplicate and split the Kaggle disease-symptom data.

Run from the project root::

    python -m src.preprocess
    python -m src.preprocess --drop-conflicts      # also drop ambiguous vectors

Why this script matters
-----------------------
The raw ``dataset.csv`` has ~4,900 rows but only a few hundred DISTINCT symptom
patterns: it is synthetic, and each pattern is repeated many times. If you split
before removing duplicates, identical rows land in both train and test and every
model "scores" ~100% (data leakage). We therefore:

1. clean names, 2. build a 0/1 symptom matrix, 3. DROP DUPLICATES,
4. only then split (stratified 70/15/15, fixed seed).
"""
from __future__ import annotations

import argparse
import difflib
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import config
from src.download_data import MANUAL_INSTRUCTIONS, find_file
from src.utils import ensure_dir, load_json, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Name cleaning
# --------------------------------------------------------------------------- #
def clean_symptom(raw: object) -> str:
    """Return the canonical form of a symptom name.

    Steps: underscores -> spaces, collapse repeated whitespace, strip, lower-case,
    then apply the typo dictionary in ``config.SYMPTOM_FIXES``. Raw data contains
    things like ``" dischromic _patches"`` and ``"spotting_ urination"``, which would
    otherwise become separate, duplicated features.
    """
    text = str(raw).replace("_", " ")
    text = _WHITESPACE.sub(" ", text).strip().lower()
    text = text.replace("( ", "(").replace(" )", ")")
    return config.SYMPTOM_FIXES.get(text, text)


def clean_disease(raw: object) -> str:
    """Return the canonical disease name (whitespace-collapsed, typos fixed).

    All four CSVs are passed through this so that e.g. ``"Peptic ulcer diseae"``
    matches across files when we join descriptions and precautions.
    """
    text = _WHITESPACE.sub(" ", str(raw)).strip()
    return config.DISEASE_FIXES.get(text.lower(), text)


def disease_key(raw: object) -> str:
    """Case-insensitive join key for a disease name."""
    return clean_disease(raw).lower()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class RawTables:
    """The four raw CSVs as DataFrames."""

    dataset: pd.DataFrame
    description: pd.DataFrame
    precaution: pd.DataFrame
    severity: pd.DataFrame


def load_raw_tables(raw_dir: Path = config.RAW_DIR) -> RawTables:
    """Load the four Kaggle CSVs, with a helpful error if any is missing."""
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for name in config.STRUCTURED_RAW_FILES:
        found = find_file(raw_dir, name)
        if found is None:
            missing.append(name)
        else:
            paths[name] = found
    if missing:
        raise FileNotFoundError(
            f"Missing in {raw_dir}: {', '.join(missing)}\n\n"
            + MANUAL_INSTRUCTIONS.format(
                structured=config.KAGGLE_SLUGS["structured"],
                text=config.KAGGLE_SLUGS["text"],
                raw_dir=raw_dir,
            )
        )
    # utf-8-sig silently strips a BOM if Excel added one.
    read = lambda n: pd.read_csv(paths[n], encoding="utf-8-sig")  # noqa: E731
    return RawTables(
        dataset=read(config.RAW_DATASET_FILE),
        description=read(config.RAW_DESCRIPTION_FILE),
        precaution=read(config.RAW_PRECAUTION_FILE),
        severity=read(config.RAW_SEVERITY_FILE),
    )


def _disease_column(df: pd.DataFrame) -> str:
    """Find the disease column (case-insensitive 'disease'; else the first column)."""
    for col in df.columns:
        if col.strip().lower() == "disease":
            return col
    return df.columns[0]


# --------------------------------------------------------------------------- #
# Matrix building, de-duplication, splitting
# --------------------------------------------------------------------------- #
def build_symptom_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert the wide ``Symptom_1..Symptom_17`` table to a binary matrix.

    Returns ``(X, y, symptoms)`` where ``X`` is uint8 ``(n_rows, n_symptoms)``,
    ``y`` an array of cleaned disease names and ``symptoms`` the alphabetically
    sorted vocabulary. Sorting makes the feature order deterministic, which is
    essential because the saved model depends on column order.
    """
    disease_col = _disease_column(df)
    symptom_cols = [c for c in df.columns if c.strip().lower().startswith("symptom")]
    if not symptom_cols:
        raise ValueError("No 'Symptom_*' columns found in dataset.csv")

    row_sets: list[set[str]] = []
    for row in df[symptom_cols].itertuples(index=False, name=None):
        # A set also collapses a symptom accidentally listed twice in one row.
        row_sets.append({clean_symptom(v) for v in row if pd.notna(v) and str(v).strip()})

    vocab = sorted(set().union(*row_sets))
    index = {s: i for i, s in enumerate(vocab)}
    X = np.zeros((len(row_sets), len(vocab)), dtype=np.uint8)
    for i, symptoms in enumerate(row_sets):
        X[i, [index[s] for s in symptoms]] = 1
    y = df[disease_col].map(clean_disease).to_numpy()
    return X, y, vocab


@dataclass
class DedupReport:
    """Numbers explaining what de-duplication removed (goes into the README)."""

    n_raw: int
    n_exact_duplicates: int
    n_unique: int
    n_conflicting_vectors: int  # same symptoms, different diseases
    n_conflict_rows_dropped: int
    n_final: int


def deduplicate(
    X: np.ndarray, y: np.ndarray, drop_conflicts: bool = False
) -> tuple[np.ndarray, np.ndarray, DedupReport]:
    """Remove exact (symptom-vector, disease) duplicates.

    Rows whose symptom vector appears under SEVERAL diseases are genuinely
    ambiguous. By default we keep them (they are honest hard cases and no model
    can get all of them right); ``drop_conflicts=True`` removes them instead.
    """
    n_raw = len(y)
    keys = [x.tobytes() + b"|" + str(label).encode() for x, label in zip(X, y)]
    seen: set[bytes] = set()
    keep = np.zeros(n_raw, dtype=bool)
    for i, key in enumerate(keys):  # keep first occurrence, preserve original order
        if key not in seen:
            seen.add(key)
            keep[i] = True
    Xu, yu = X[keep], y[keep]

    vec_counts = Counter(x.tobytes() for x in Xu)
    conflicting = {v for v, c in vec_counts.items() if c > 1}
    n_dropped = 0
    if drop_conflicts and conflicting:
        mask = np.array([x.tobytes() not in conflicting for x in Xu])
        n_dropped = int((~mask).sum())
        Xu, yu = Xu[mask], yu[mask]

    report = DedupReport(
        n_raw=n_raw,
        n_exact_duplicates=int(n_raw - keep.sum()),
        n_unique=int(keep.sum()),
        n_conflicting_vectors=len(conflicting),
        n_conflict_rows_dropped=n_dropped,
        n_final=len(yu),
    )
    return Xu, yu, report


def count_subset_rows(X: np.ndarray, y: np.ndarray) -> int:
    """Count rows whose symptom set is a strict subset of another row of the SAME disease.

    These "near duplicates" survive exact de-duplication and are one reason test
    accuracy stays optimistic; we report them so the README can be honest.
    """
    total = 0
    for label in np.unique(y):
        M = X[y == label].astype(bool)
        for i in range(len(M)):
            others = np.delete(M, i, axis=0)
            if len(others) and np.any(np.all(others[:, M[i]], axis=1) & (others.sum(1) > M[i].sum())):
                total += 1
    return total


def stratified_split(
    labels: np.ndarray,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    test_frac: float = config.TEST_FRAC,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class shuffled split returning ``(train_idx, val_idx, test_idx)``.

    We do not use sklearn's ``train_test_split(stratify=...)`` because after
    de-duplication some diseases have only a handful of rows and sklearn raises an
    error when a class cannot appear in every split. Here every class with >= 3
    rows is guaranteed at least one row in each split; smaller classes go entirely
    to train (and a warning is logged).
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            logger.warning("Class %r has only %d unique rows; all placed in train.", label, n)
            train.extend(idx.tolist())
            continue
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n - n_test - n_val < 1:  # only possible for tiny classes
            n_test = n_val = 1
        test.extend(idx[:n_test].tolist())
        val.extend(idx[n_test : n_test + n_val].tolist())
        train.extend(idx[n_test + n_val :].tolist())
    out = []
    for part in (train, val, test):
        arr = np.array(part, dtype=int)
        rng.shuffle(arr)  # avoid class-ordered batches
        out.append(arr)
    return out[0], out[1], out[2]


# --------------------------------------------------------------------------- #
# Side tables for the app (descriptions, precautions, severities, symptom map)
# --------------------------------------------------------------------------- #
def _match_disease(key: str, candidates: list[str]) -> str | None:
    """Exact key match, else a close fuzzy match (typos differ between files)."""
    if key in candidates:
        return key
    close = difflib.get_close_matches(key, candidates, n=1, cutoff=0.8)
    return close[0] if close else None


def build_disease_info(raw: RawTables, classes: list[str]) -> dict[str, dict]:
    """Join description + up to four precautions onto each class name."""
    desc_df = raw.description.copy()
    desc_df["_key"] = desc_df.iloc[:, 0].map(disease_key)
    desc_by_key = dict(zip(desc_df["_key"], desc_df.iloc[:, 1].astype(str)))

    prec_df = raw.precaution.copy()
    prec_df["_key"] = prec_df.iloc[:, 0].map(disease_key)
    prec_by_key = {
        row["_key"]: [
            _WHITESPACE.sub(" ", str(v)).strip()
            for k, v in row.items()
            if k.lower().startswith("precaution") and pd.notna(v) and str(v).strip()
        ]
        for _, row in prec_df.iterrows()
    }

    info: dict[str, dict] = {}
    for disease in classes:
        key = disease.lower()
        d_key = _match_disease(key, list(desc_by_key))
        p_key = _match_disease(key, list(prec_by_key))
        if d_key is None:
            logger.warning("No description found for %r", disease)
        if p_key is None:
            logger.warning("No precautions found for %r", disease)
        info[disease] = {
            "description": _WHITESPACE.sub(" ", desc_by_key[d_key]).strip() if d_key else "",
            "precautions": prec_by_key[p_key] if p_key else [],
        }
    return info


def build_severity(raw: RawTables, symptoms: list[str]) -> dict[str, int]:
    """Map each model symptom to its Kaggle severity weight (1-7)."""
    sev = raw.severity
    names = sev.iloc[:, 0].map(clean_symptom)
    weights = pd.to_numeric(sev.iloc[:, 1], errors="coerce")
    table = {n: int(w) for n, w in zip(names, weights) if pd.notna(w)}
    result: dict[str, int] = {}
    unmatched: list[str] = []
    for s in symptoms:
        if s in table:
            result[s] = table[s]
            continue
        close = difflib.get_close_matches(s, list(table), n=1, cutoff=0.85)
        if close:
            result[s] = table[close[0]]
        else:
            unmatched.append(s)
    if unmatched:
        logger.warning("No severity weight for %d symptoms: %s", len(unmatched), unmatched)
    return result


def build_disease_symptom_map(
    X: np.ndarray, y: np.ndarray, symptoms: list[str]
) -> dict[str, dict[str, float]]:
    """For each disease: fraction of its unique rows that contain each symptom.

    This is the dataset's "known mapping" used later by the SHAP sanity check.
    """
    mapping: dict[str, dict[str, float]] = {}
    for label in sorted(set(y)):
        sub = X[y == label]
        freq = sub.mean(axis=0)
        mapping[label] = {
            symptoms[j]: round(float(freq[j]), 4) for j in np.flatnonzero(freq > 0)
        }
    return mapping


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def compute_stats(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    X_dedup: np.ndarray,
    y_dedup: np.ndarray,
    symptoms: list[str],
    report: DedupReport,
    n_subset_rows: int,
    split_sizes: dict[str, int],
) -> dict:
    """Collect dataset statistics for the logs, reports/ and the README."""
    classes = sorted(set(y_raw))
    per_disease = []
    for c in classes:
        raw_rows = X_raw[y_raw == c]
        uniq_rows = X_dedup[y_dedup == c]
        per_disease.append(
            {
                "disease": c,
                "raw_rows": int(len(raw_rows)),
                "unique_rows": int(len(uniq_rows)),
                "distinct_symptoms": int((uniq_rows.sum(0) > 0).sum()) if len(uniq_rows) else 0,
                "mean_symptoms_per_row": round(float(uniq_rows.sum(1).mean()), 2)
                if len(uniq_rows)
                else 0.0,
            }
        )
    counts = np.array([d["unique_rows"] for d in per_disease])
    sym_per_row = X_dedup.sum(1)
    return {
        "n_diseases": len(classes),
        "n_symptoms": len(symptoms),
        "dedup": asdict(report),
        "duplicate_fraction": round(report.n_exact_duplicates / report.n_raw, 4),
        "subset_rows_same_disease": n_subset_rows,
        "unique_rows_per_disease": {
            "min": int(counts.min()),
            "median": float(np.median(counts)),
            "max": int(counts.max()),
        },
        "symptoms_per_unique_row": {
            "min": int(sym_per_row.min()),
            "mean": round(float(sym_per_row.mean()), 2),
            "max": int(sym_per_row.max()),
        },
        "split_sizes": split_sizes,
        "per_disease": per_disease,
    }


def stats_to_markdown(stats: dict) -> str:
    """Render the stats dict as a markdown report."""
    d = stats["dedup"]
    lines = [
        "# Dataset statistics",
        "",
        f"- Diseases: **{stats['n_diseases']}**, symptoms (features): **{stats['n_symptoms']}**",
        f"- Raw rows: **{d['n_raw']}**; exact duplicates removed: **{d['n_exact_duplicates']}** "
        f"({stats['duplicate_fraction']:.1%}); unique rows: **{d['n_unique']}**",
        f"- Symptom vectors shared by several diseases (ambiguous): **{d['n_conflicting_vectors']}**; "
        f"rows dropped for this reason: **{d['n_conflict_rows_dropped']}**",
        f"- Rows that are a strict subset of another row of the same disease: "
        f"**{stats['subset_rows_same_disease']}**",
        f"- Final rows: **{d['n_final']}** -> split sizes {stats['split_sizes']}",
        f"- Unique rows per disease (min / median / max): "
        f"{stats['unique_rows_per_disease']['min']} / {stats['unique_rows_per_disease']['median']} / "
        f"{stats['unique_rows_per_disease']['max']}",
        f"- Symptoms per unique row (min / mean / max): "
        f"{stats['symptoms_per_unique_row']['min']} / {stats['symptoms_per_unique_row']['mean']} / "
        f"{stats['symptoms_per_unique_row']['max']}",
        "",
        "> High model accuracy on this dataset reflects its simplicity and synthetic origin, "
        "NOT clinical reliability.",
        "",
        "| Disease | Raw rows | Unique rows | Distinct symptoms | Mean symptoms/row |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in stats["per_disease"]:
        lines.append(
            f"| {row['disease']} | {row['raw_rows']} | {row['unique_rows']} | "
            f"{row['distinct_symptoms']} | {row['mean_symptoms_per_row']} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _write_split(path: Path, X: np.ndarray, y: np.ndarray, symptoms: list[str]) -> None:
    """Save a split as a readable CSV: one 0/1 column per symptom + the label."""
    frame = pd.DataFrame(X, columns=symptoms)
    frame[config.LABEL_COLUMN] = y
    frame.to_csv(path, index=False)


def run_preprocessing(
    raw_dir: Path = config.RAW_DIR, drop_conflicts: bool = False, seed: int = config.SEED
) -> dict:
    """Execute the whole pipeline and write every artifact; returns the stats dict."""
    set_seed(seed)
    raw = load_raw_tables(raw_dir)

    X_raw, y_raw, symptoms = build_symptom_matrix(raw.dataset)
    logger.info("Raw rows: %d | diseases: %d | symptoms: %d", len(y_raw), len(set(y_raw)), len(symptoms))

    X, y, report = deduplicate(X_raw, y_raw, drop_conflicts=drop_conflicts)
    logger.info(
        "De-duplication: %d raw -> %d unique (%d exact duplicates removed, %.1f%%)",
        report.n_raw, report.n_final, report.n_exact_duplicates,
        100 * report.n_exact_duplicates / report.n_raw,
    )
    if report.n_conflicting_vectors:
        logger.info(
            "%d symptom vectors occur under more than one disease (%s).",
            report.n_conflicting_vectors,
            "dropped" if drop_conflicts else "kept as honest hard cases",
        )
    n_subset = count_subset_rows(X, y)

    encoder = LabelEncoder().fit(y)
    classes = [str(c) for c in encoder.classes_]
    codes = encoder.transform(y)

    tr, va, te = stratified_split(codes, seed=seed)
    sizes = {"train": len(tr), "val": len(va), "test": len(te)}
    logger.info("Split sizes: %s", sizes)
    # Leakage guard: no row index may appear in two splits.
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te)), "split overlap!"

    ensure_dir(config.PROCESSED_DIR)
    _write_split(config.TRAIN_CSV, X[tr], y[tr], symptoms)
    _write_split(config.VAL_CSV, X[va], y[va], symptoms)
    _write_split(config.TEST_CSV, X[te], y[te], symptoms)
    _write_split(config.FULL_DEDUP_CSV, X, y, symptoms)

    ensure_dir(config.MODELS_DIR)
    save_json(symptoms, config.SYMPTOMS_JSON)
    save_json(classes, config.CLASSES_JSON)
    joblib.dump(encoder, config.LABEL_ENCODER_PKL)

    save_json(build_disease_info(raw, classes), config.DISEASE_INFO_JSON)
    save_json(build_severity(raw, symptoms), config.SEVERITY_JSON)
    save_json(build_disease_symptom_map(X, y, symptoms), config.DISEASE_SYMPTOM_MAP_JSON)

    stats = compute_stats(X_raw, y_raw, X, y, symptoms, report, n_subset, sizes)
    ensure_dir(config.REPORTS_DIR)
    save_json(stats, config.DATASET_STATS_JSON)
    config.DATASET_STATS_MD.write_text(stats_to_markdown(stats), encoding="utf-8")

    table = pd.DataFrame(stats["per_disease"]).set_index("disease")
    logger.info("Per-disease statistics:\n%s", table.to_string())
    logger.info(
        "Unique rows per disease min/median/max: %s | rows that are a subset of a same-disease row: %d",
        stats["unique_rows_per_disease"], n_subset,
    )
    logger.info(
        "NOTE: very high accuracy on this data reflects its synthetic simplicity, not clinical reliability."
    )
    return stats


# --------------------------------------------------------------------------- #
# Helpers used by later stages (training, evaluation, app)
# --------------------------------------------------------------------------- #
def load_vocab() -> tuple[list[str], list[str]]:
    """Return ``(symptoms, classes)`` in the exact order the models use."""
    return load_json(config.SYMPTOMS_JSON), load_json(config.CLASSES_JSON)


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ``'train'``, ``'val'``, ``'test'`` or ``'full'`` as ``(X float32, y int)``."""
    paths = {
        "train": config.TRAIN_CSV,
        "val": config.VAL_CSV,
        "test": config.TEST_CSV,
        "full": config.FULL_DEDUP_CSV,
    }
    if name not in paths:
        raise ValueError(f"split must be one of {sorted(paths)}")
    symptoms, classes = load_vocab()
    frame = pd.read_csv(paths[name])
    class_index = {c: i for i, c in enumerate(classes)}
    X = np.array(frame[symptoms].to_numpy(dtype=np.float32))  # writable copy
    y = np.array(frame[config.LABEL_COLUMN].map(class_index).to_numpy(dtype=np.int64))
    return X, y


def encode_symptoms(selected: Iterable[str], symptoms: list[str]) -> tuple[np.ndarray, list[str]]:
    """Turn user-selected symptom names into a model input vector.

    Returns ``(vector, unknown)`` where ``unknown`` lists names not in the
    vocabulary, so the UI can warn instead of silently ignoring them.
    """
    index = {s: i for i, s in enumerate(symptoms)}
    vec = np.zeros(len(symptoms), dtype=np.float32)
    unknown: list[str] = []
    for name in selected:
        cleaned = clean_symptom(name)
        if cleaned in index:
            vec[index[cleaned]] = 1.0
        else:
            unknown.append(str(name))
    return vec, unknown


def random_symptom_dropout(
    X: np.ndarray,
    rng: np.random.Generator,
    drop_min: float = config.AUG_DROP_MIN,
    drop_max: float = config.AUG_DROP_MAX,
) -> np.ndarray:
    """Training augmentation: hide a random 10-30% of each patient's symptoms.

    Real users forget or do not notice symptoms, but the dataset lists every
    symptom of a pattern. Training on thinned copies teaches the model to cope
    with incomplete input. Each row gets its own drop rate, and at least one
    symptom is always kept so no row becomes an all-zero vector.
    """
    X = np.asarray(X, dtype=np.float32)
    rate = rng.uniform(drop_min, drop_max, size=(len(X), 1))
    keep = (rng.random(X.shape) >= rate).astype(np.float32)
    out = X * keep
    empty = (out.sum(1) == 0) & (X.sum(1) > 0)
    out[empty] = X[empty]  # restore rows that lost everything
    return out


def remove_fraction(X: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Remove exactly ``round(fraction * n_present)`` symptoms from every row.

    Used by the robustness curve (unlike the random-rate augmentation above, the
    removed share is fixed so the x-axis of the plot is exact). One symptom is
    always kept.
    """
    out = np.asarray(X, dtype=np.float32).copy()
    for i in range(len(out)):
        present = np.flatnonzero(out[i] > 0)
        k = min(int(round(fraction * len(present))), max(len(present) - 1, 0))
        if k > 0:
            out[i, rng.choice(present, size=k, replace=False)] = 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Clean, de-duplicate and split the dataset.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--drop-conflicts",
        action="store_true",
        help="remove symptom vectors that appear under more than one disease",
    )
    args = parser.parse_args(argv)
    setup_logging()
    try:
        run_preprocessing(args.raw_dir, args.drop_conflicts, args.seed)
    except FileNotFoundError as err:
        logger.error("%s", err)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
