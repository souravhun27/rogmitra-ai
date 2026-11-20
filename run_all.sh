#!/usr/bin/env bash
# Run the whole RogMitra.AI pipeline end to end (CPU is enough; a few minutes on a laptop).
#
#   bash run_all.sh              # everything except the optional text/multi-label models
#   bash run_all.sh --with-text  # also train the optional DistilBERT free-text model
#
# Prerequisite: the Kaggle CSV files are in data/raw/ (see README, section "Get the data"),
# and the virtual environment is active.
set -euo pipefail

PY="${PY:-python}"
WITH_TEXT=0
for arg in "$@"; do
  [[ "$arg" == "--with-text" ]] && WITH_TEXT=1
done

step() { printf '\n=== %s ===\n' "$1"; }

step "1/8 Preprocess (clean, de-duplicate, split)"; $PY -m src.preprocess
step "2/8 Exploratory data analysis";             $PY -m src.eda
step "3/8 Train baselines";                        $PY -m src.train_baselines
step "4/8 Train MLP + temperature scaling";        $PY -m src.train_mlp
step "5/8 Evaluate (metrics, CV, calibration, robustness)"; $PY -m src.evaluate
step "6/8 Stricter tests (novel patterns, noisy input)";    $PY -m src.evaluate_hard
step "7/8 Explainability (SHAP) + sanity check";   $PY -m src.explain; $PY -m src.sanity_check; $PY -m src.update_readme

if [[ "$WITH_TEXT" == "1" ]]; then
  step "Optional: DistilBERT text model";          $PY -m src.train_text_model
  step "Optional: multi-label experiment";         $PY -m src.train_multilabel
fi

step "8/8 Unit tests";                             $PY -m pytest -q

printf '\nDone. Start the app with:  streamlit run app/Home.py\n'
