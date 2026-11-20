# Model comparison (test split, n = 45)

| Model | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Top-3 | ROC-AUC | ECE | CV acc (%) | CV macro F1 (%) | CV top-3 (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.051 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.044 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.026 | 99.7 ± 0.7 | 99.4 ± 1.2 | 100.0 ± 0.0 |
| MLP (calibrated) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |

The MLP row uses temperature-scaled probabilities (so its ECE is the calibrated one); CV rows use the uncalibrated MLP trained for a fixed number of epochs.
