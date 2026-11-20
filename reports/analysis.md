# Results analysis

> Auto-generated from `src/evaluate.py`. Numbers are real; review the wording and add your own interpretation.

## Headline results

- MLP (calibrated) on the test split (n = 45): accuracy **100.0%**, macro F1 **1.000**, top-3 accuracy **100.0%**.
- Best baseline: **Logistic Regression** with accuracy 100.0%.
- With only 45 test rows, one mistake moves accuracy by 2.2 points, so differences of one or two rows between models are NOT meaningful.
- 5-fold CV accuracy (mean ± std, %): Logistic Regression 100.0 ± 0.0, Random Forest 100.0 ± 0.0, XGBoost 99.7 ± 0.7, MLP (calibrated) 100.0 ± 0.0

## Calibration

Temperature scaling fitted on the validation set gave T = 0.22. On the test split ECE changed from 0.041 to 0.000 and NLL from 0.043 to 0.000. Accuracy is unchanged because dividing logits by T never changes the arg-max. ECE on ~45 samples is noisy; treat it as indicative.

## Robustness to missing symptoms

- Logistic Regression: 100.0% with all symptoms -> 91.0% when half are removed.
- MLP (calibrated): 100.0% with all symptoms -> 92.0% when half are removed.
- Random Forest: 100.0% with all symptoms -> 92.1% when half are removed.
- XGBoost: 100.0% with all symptoms -> 78.9% when half are removed.

## Failure cases (based on out-of-fold predictions of the MLP over all 304 unique rows)

No misclassifications were found.

### Misclassified test rows

None.

## Caveats

- The dataset is largely synthetic; high scores reflect its simplicity, not clinical reliability.
- After de-duplication each disease has only a handful of distinct rows, and some rows are strict subsets of others (see `reports/dataset_stats.md`), so test rows can resemble training rows.
- The validation set is used for early stopping AND for temperature scaling; the test split is used only here.
