"""Central configuration for ExplainDx.

Every path, seed, threshold and hyper-parameter lives here so that a student can
change one file instead of hunting through scripts. Other modules import values
from this file and never hard-code them.

Set the environment variable ``EXPLAINDX_ROOT`` to redirect ALL inputs/outputs
(data, models, reports) to another folder. We use this for tests and for dry
runs so the real project folder is never touched.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(
    os.environ.get("EXPLAINDX_ROOT", Path(__file__).resolve().parent.parent)
).resolve()

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
REMEDIES_PATH: Path = DATA_DIR / "remedies.json"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# --------------------------------------------------------------------------- #
# Raw dataset files (names exactly as shipped on Kaggle; lookup is case-insensitive)
# --------------------------------------------------------------------------- #
RAW_DATASET_FILE = "dataset.csv"
RAW_DESCRIPTION_FILE = "symptom_Description.csv"
RAW_PRECAUTION_FILE = "symptom_precaution.csv"
RAW_SEVERITY_FILE = "Symptom-severity.csv"
RAW_TEXT_FILE = "Symptom2Disease.csv"  # only needed for the optional text model

STRUCTURED_RAW_FILES: tuple[str, ...] = (
    RAW_DATASET_FILE,
    RAW_DESCRIPTION_FILE,
    RAW_PRECAUTION_FILE,
    RAW_SEVERITY_FILE,
)

KAGGLE_SLUGS: dict[str, str] = {
    "structured": "itachi9604/disease-symptom-description-dataset",
    "text": "niyarrbarman/symptom2disease",
}

# --------------------------------------------------------------------------- #
# Processed artifacts (written by src.preprocess, read by everything else)
# --------------------------------------------------------------------------- #
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
FULL_DEDUP_CSV = PROCESSED_DIR / "full_dedup.csv"
DISEASE_INFO_JSON = PROCESSED_DIR / "disease_info.json"
SEVERITY_JSON = PROCESSED_DIR / "symptom_severity.json"
DISEASE_SYMPTOM_MAP_JSON = PROCESSED_DIR / "disease_symptom_map.json"

SYMPTOMS_JSON = MODELS_DIR / "symptoms.json"  # ordered feature list = model input order
CLASSES_JSON = MODELS_DIR / "classes.json"  # ordered class list = model output order
LABEL_ENCODER_PKL = MODELS_DIR / "label_encoder.joblib"

DATASET_STATS_JSON = REPORTS_DIR / "dataset_stats.json"
DATASET_STATS_MD = REPORTS_DIR / "dataset_stats.md"

LABEL_COLUMN = "disease"  # name of the target column in processed CSVs

# --------------------------------------------------------------------------- #
# Reproducibility and splitting
# --------------------------------------------------------------------------- #
SEED: int = 42
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15

# --------------------------------------------------------------------------- #
# Training (used from Stage 2 onward; kept here so there is one source of truth)
# --------------------------------------------------------------------------- #
AUG_DROP_MIN: float = 0.10  # randomly hide 10-30% of a patient's symptoms ...
AUG_DROP_MAX: float = 0.30  # ... every epoch, to mimic incomplete real-world input

MLP_HIDDEN_DIMS: tuple[int, ...] = (256, 128)
MLP_DROPOUT: float = 0.3
MLP_LABEL_SMOOTHING: float = 0.1
MLP_LR: float = 1e-3
MLP_WEIGHT_DECAY: float = 1e-2
MLP_BATCH_SIZE: int = 32
MLP_MAX_EPOCHS: int = 300
MLP_EARLY_STOP_PATIENCE: int = 30
MLP_SCHEDULER_PATIENCE: int = 10

CV_FOLDS: int = 5
ROBUSTNESS_FRACTIONS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# --------------------------------------------------------------------------- #
# Uncertainty / safety (Stage 3)
# --------------------------------------------------------------------------- #
MC_DROPOUT_PASSES: int = 30
CONFIDENCE_THRESHOLD: float = 0.50  # top-1 probability below this => "model is unsure"
TOP2_GAP_THRESHOLD: float = 0.10  # top-1 minus top-2 below this => "model is unsure"
MIN_SYMPTOMS_FOR_CONFIDENCE: int = 3  # fewer reported symptoms => always flagged as unsure
MC_RANGE_PERCENTILES: tuple[float, float] = (5.0, 95.0)  # confidence range shown to the user
UNSURE_MESSAGE: str = "The model is unsure, please consult a doctor."

# --------------------------------------------------------------------------- #
# Explainability (Stage 3)
# --------------------------------------------------------------------------- #
TOP_K_DISPLAY: int = 3  # diseases shown to the user
IG_STEPS: int = 64  # Integrated Gradients path steps (more = closer to exact)
SHAP_BACKGROUND_SIZE: int = 50  # background rows for SHAP
SHAP_NSAMPLES: int = 50  # GradientExplainer samples per row (default 200 is ~4x slower)
SHAP_TREE_MAX_ROWS: int = 100  # Random Forest TreeExplainer is slow; explain a random subset
FOLLOWUP_CANDIDATES: int = 5  # top diseases considered when choosing a follow-up question
FOLLOWUP_TOP_N: int = 5  # number of follow-up questions suggested
SANITY_TOP_K: int = 5  # symptoms compared per disease in the SHAP sanity check

# --------------------------------------------------------------------------- #
# Name-cleaning dictionaries
# --------------------------------------------------------------------------- #
# Keys are the symptom after lower-casing, underscore->space and whitespace
# collapsing (see preprocess.clean_symptom). Values are the canonical spelling.
# The raw Kaggle files contain these genuine typos / odd spacings.
SYMPTOM_FIXES: dict[str, str] = {
    "dischromic patches": "discolored patches",
    "scurring": "scurfing",
    "swollen extremeties": "swollen extremities",
    "cold hands and feets": "cold hands and feet",
    "extra marital contacts": "extra-marital contacts",
}

# Keys: lower-cased, whitespace-collapsed disease name as found in the raw files.
# The three CSVs spell some diseases differently, so every file is passed through
# the same function and joined on the cleaned name.
DISEASE_FIXES: dict[str, str] = {
    "peptic ulcer diseae": "Peptic ulcer disease",
    "dimorphic hemmorhoids(piles)": "Hemorrhoids (piles)",
    "dimorphic hemorrhoids(piles)": "Hemorrhoids (piles)",
    "osteoarthristis": "Osteoarthritis",
    "(vertigo) paroymsal positional vertigo": "Paroxysmal positional vertigo",
    "(vertigo) paroxysmal positional vertigo": "Paroxysmal positional vertigo",
    "hepatitis a": "Hepatitis A",
    "hepatitis b": "Hepatitis B",
    "hepatitis c": "Hepatitis C",
    "hepatitis d": "Hepatitis D",
    "hepatitis e": "Hepatitis E",
    "gerd": "GERD",
    "aids": "AIDS",
    "common cold": "Common cold",
    "drug reaction": "Drug reaction",
    "bronchial asthma": "Bronchial asthma",
    "paralysis (brain hemorrhage)": "Paralysis (brain hemorrhage)",
    "chicken pox": "Chickenpox",
    "gastroesophageal reflux disease": "GERD",
}


# --------------------------------------------------------------------------- #
# Optional text model (Stage 5): DistilBERT fine-tuned on Symptom2Disease
# --------------------------------------------------------------------------- #
TEXT_MODEL_NAME: str = "distilbert-base-uncased"
TEXT_MODEL_DIR: Path = MODELS_DIR / "text_model"
TEXT_METRICS_JSON: Path = REPORTS_DIR / "text_model_metrics.json"
TEXT_METRICS_MD: Path = REPORTS_DIR / "text_model.md"
TEXT_MAX_LEN: int = 64  # short on purpose: symptom descriptions are 1-3 sentences and CPU time grows with length
TEXT_EPOCHS: int = 4
TEXT_BATCH_SIZE: int = 16
TEXT_LR: float = 5e-5

# --------------------------------------------------------------------------- #
# Optional multi-label synthetic experiment (Stage 5)
# --------------------------------------------------------------------------- #
ML_MODEL_PATH: Path = MODELS_DIR / "multilabel_mlp.pt"
ML_METRICS_JSON: Path = REPORTS_DIR / "multilabel_metrics.json"
ML_METRICS_MD: Path = REPORTS_DIR / "multilabel_metrics.md"
ML_PAIRS_TRAIN: int = 2000  # synthetic two-disease training cases
ML_PAIRS_VAL: int = 400
ML_PAIRS_TEST: int = 400
