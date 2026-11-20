# Data folder

| Path | What | In git? |
|---|---|---|
| `raw/` | Original Kaggle CSVs (you download them) | no |
| `processed/` | Cleaned, de-duplicated train/val/test CSVs and JSON side tables | no (re-created by `python -m src.preprocess`) |
| `remedies.json` | Hand-curated home-remedy entries with source URLs (added in Stage 4) | yes |

## Sources

1. **Disease Symptom Prediction** (Pranay Patil) - https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
   Files: `dataset.csv`, `symptom_Description.csv`, `symptom_precaution.csv`, `Symptom-severity.csv`
2. **Symptom2Disease** (Niyar R Barman) - https://www.kaggle.com/datasets/niyarrbarman/symptom2disease
   File: `Symptom2Disease.csv` (optional, Stage 5)

Check the license shown on each Kaggle page and cite it in your report.

## Download

```bash
# Option A: Kaggle CLI (needs ~/.kaggle/kaggle.json)
python -m src.download_data            # add --text for Symptom2Disease

# Option B: manual - download the zips from the links above and copy the CSVs into data/raw/
python -m src.download_data --check    # confirms the files are in place
```
