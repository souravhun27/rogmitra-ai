"""SHAP sanity check: does the model's idea of each disease match the dataset?

For every disease we ask: "which symptoms does the MLP treat as the strongest
evidence FOR this disease?" (mean SHAP toward the disease when the symptom is
present, across all rows). We then compare the top symptoms with the dataset's
known disease-symptom mapping (``data/processed/disease_symptom_map.json``).

* mismatch  = a top SHAP symptom that never appears for that disease in the data
  (the model learned an association the dataset does not contain: investigate).
* coverage  = share of the top-k symptoms that ARE known symptoms of the disease.
* recall    = how many of the disease's 3 most frequent symptoms the model puts
  in its top-10.

Run: ``python -m src.sanity_check``. Output: ``reports/sanity_check.md`` / ``.csv``.
Use the flagged rows as discussion material in your report (e.g. correlated
symptoms, shortcut learning, tiny dataset).
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src import config
from src.explain import class_symptom_effect, shap_mlp
from src.models import load_mlp
from src.preprocess import load_split, load_vocab
from src.utils import load_json, set_seed, setup_logging

logger = logging.getLogger(__name__)


def run_sanity_check(top_k: int = config.SANITY_TOP_K, seed: int = config.SEED) -> pd.DataFrame:
    """Compute the per-disease comparison table and write the report files."""
    set_seed(seed)
    symptoms, classes = load_vocab()
    known_map: dict[str, dict[str, float]] = load_json(config.DISEASE_SYMPTOM_MAP_JSON)
    X_tr, _ = load_split("train")
    X, _ = load_split("full")
    rng = np.random.default_rng(seed)
    bg = X_tr[rng.choice(len(X_tr), size=min(config.SHAP_BACKGROUND_SIZE, len(X_tr)), replace=False)]
    model, temperature, _ = load_mlp()
    effect = class_symptom_effect(shap_mlp(model, temperature, bg, X), X)

    rows = []
    for c, disease in enumerate(classes):
        known = known_map.get(disease, {})
        scores = np.nan_to_num(effect[c], nan=-np.inf)
        ranked = [j for j in np.argsort(-scores) if np.isfinite(scores[j]) and scores[j] > 0]
        top = [symptoms[j] for j in ranked[:top_k]]
        mismatches = [s for s in top if s not in known]
        frequent = sorted(known, key=lambda s: -known[s])[:3]
        top10 = {symptoms[j] for j in ranked[:10]}
        recall = sum(s in top10 for s in frequent) / max(len(frequent), 1)
        rows.append({
            "disease": disease,
            "model_top_symptoms": "; ".join(top),
            "mismatches": "; ".join(mismatches),
            "coverage": (len(top) - len(mismatches)) / len(top) if top else float("nan"),
            "frequent_symptom_recall": recall,
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.REPORTS_DIR / "sanity_check.csv", index=False)

    flagged = df[df["mismatches"] != ""]
    lines = [
        "# SHAP sanity check", "",
        f"- Diseases checked: {len(df)}; top-k symptoms compared per disease: {top_k}",
        f"- Mean coverage of known symptoms: **{df['coverage'].mean():.1%}**",
        f"- Mean recall of the 3 most frequent known symptoms (in model top-10): **{df['frequent_symptom_recall'].mean():.1%}**",
        f"- Diseases with at least one mismatch: **{len(flagged)}**", "",
        "A mismatch means the model treats a symptom as evidence for a disease although that symptom never "
        "appears with the disease in the dataset. Possible reasons: symptoms shared by similar diseases, very "
        "few unique rows per disease, or the network learning shortcuts. Discuss these in your report.", "",
        "| Disease | Model's top symptoms | Mismatches | Coverage | Frequent-symptom recall |",
        "|---|---|---|---:|---:|",
    ]
    for r in df.itertuples():
        lines.append(f"| {r.disease} | {r.model_top_symptoms} | {r.mismatches or '-'} | {r.coverage:.0%} | {r.frequent_symptom_recall:.0%} |")
    (config.REPORTS_DIR / "sanity_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Coverage %.1f%% | flagged diseases: %d/%d | report: reports/sanity_check.md",
                100 * df["coverage"].mean(), len(flagged), len(df))
    return df


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare SHAP-top symptoms with the dataset mapping.")
    parser.add_argument("--top-k", type=int, default=config.SANITY_TOP_K)
    args = parser.parse_args(argv)
    setup_logging()
    run_sanity_check(args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
