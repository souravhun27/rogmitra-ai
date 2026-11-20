# RogMitra.AI: Viva and Presentation Cheat Sheet

Every technique in the project, explained in simple words, followed by the hard questions a teacher is likely to ask.
Numbers such as accuracy come from **your own run**: open `reports/metrics.md` and `reports/hard_eval.json` before the viva.

---

## 1. The 30-second pitch

"RogMitra.AI takes a patient's symptoms and suggests the three most likely diseases. Unlike a black box, it also explains
*why* (which symptoms pushed the answer), says *how sure* it is, shows what would change if a symptom were missing, asks the
most useful next question, and raises an emergency alert for dangerous symptoms. I trained every model myself on public
data. It is an academic demonstration of explainable AI, not a medical device."

## 2. The pipeline in one picture

Raw CSV → clean names → remove duplicates → split train / validation / test → train models → calibrate → evaluate → explain → web app.

---

## 3. Data

| Topic | Simple explanation |
|---|---|
| Input format | Each patient is a row of 0/1 values, one per symptom (about 130 symptoms). 1 means the symptom is present. |
| Cleaning | Names are tidied (spaces, underscores, typos) so "skin_rash" and " skin rash" are one feature, not two. |
| Duplicates | The dataset repeats the same patient many times (thousands of rows shrink to a few hundred unique ones). If copies sit in both train and test, the model is tested on what it memorised. So I **remove duplicates first, then split**. |
| Stratified split | Train 70%, validation 15%, test 15%, with every disease present in each part. A fixed seed (42) makes the split repeatable. |
| Subset rows | Some rows are shorter versions of another row of the same disease. They survive duplicate removal and make the test a bit easier. I counted them and report this honestly. |
| Augmentation | While training, 10 to 30% of each patient's symptoms are hidden at random each epoch. Real users forget symptoms, so the model learns to cope. |

## 4. Models

| Model | One-line explanation |
|---|---|
| Logistic Regression | Adds up weighted symptoms and turns the total into probabilities. Simple, fast, easy to interpret. |
| Random Forest | Many decision trees vote. |
| XGBoost | Trees built one after another, each fixing the previous mistakes. |
| **MLP (main model)** | A small neural network: 131 inputs → 256 → 128 → 41 diseases. |

MLP ingredients, in plain words:

* **ReLU:** keeps positive signals, zeroes negative ones; lets the network learn non-linear patterns.
* **Batch Normalisation:** keeps each layer's numbers in a stable range so training is smoother.
* **Dropout (0.3):** randomly switches off 30% of neurons during training so the network cannot depend on any one of them (reduces overfitting).
* **Label smoothing (0.1):** instead of "100% this disease", the training target is "about 90%", which stops the network becoming over-confident.
* **AdamW:** the optimiser that updates the weights; "W" means weight decay is applied properly, which also fights overfitting.
* **ReduceLROnPlateau:** lowers the learning rate when the validation score stops improving.
* **Early stopping:** stops training when validation stops improving and keeps the best checkpoint.

## 5. Calibration (are the percentages honest?)

* A model saying "80%" should be right about 80% of the time. Neural networks tend to be over-confident.
* **Temperature scaling:** divide the network's raw scores by one number T, fitted on the validation set. T changes confidence, not the ranking of diseases, so accuracy stays the same.
* **ECE (Expected Calibration Error):** the average gap between stated confidence and real accuracy. Lower is better.
* **Reliability diagram:** a chart of confidence against actual accuracy; a perfect model follows the diagonal.

## 6. Evaluation metrics

| Metric | Meaning |
|---|---|
| Accuracy | Share of patients whose top answer is right. |
| Macro F1 | Average F1 over diseases, each disease counted equally (F1 balances precision and recall). |
| Weighted F1 | Same, but bigger diseases count more. |
| Precision / recall | Of the patients I labelled X, how many really had X / of the patients who had X, how many I found. |
| Top-3 accuracy | The right disease is among my three suggestions. |
| ROC-AUC (one-vs-rest) | How well the model ranks the true disease above the others, for each disease. |
| Confusion matrix | Table of true vs predicted disease; shows which diseases get mixed up. |
| 5-fold cross-validation | Split the data five ways, train five times, test on each part in turn; report mean ± standard deviation. More reliable than one small test split. |
| Robustness curve | Accuracy as 0 to 50% of the symptoms are removed. |
| Stricter tests | (a) only test patients that are not near-copies of a training patient; (b) 30% of symptoms removed plus wrong symptoms added. |

## 7. Explainability

* **Integrated Gradients (per patient):** starts from "no symptoms" and slowly adds the patient's symptoms, adding up how much each one raised the chosen disease's probability. The contributions sum to the total change (a property called completeness). Green bars support the disease, red bars oppose it.
* **SHAP (global):** based on Shapley values from game theory: each symptom's fair share of the credit, averaged over many symptom combinations. I use it to rank the most important symptoms per model.
* **Plain-language explanation:** a fixed template that fills in the top symptoms and values. No language model is involved.
* **Counterfactual:** remove each selected symptom one at a time, run the model again, and report what the top disease and confidence become.
* **Follow-up suggester:** for each symptom not yet selected, estimate how much the model's uncertainty (entropy) would fall if the answer were "yes", weighted by how likely "yes" is. The best one is suggested.
* **Sanity check:** compares the symptoms the model finds important with the symptoms the dataset says belong to each disease; mismatches are listed in the report.

## 8. Uncertainty and safety

* **Monte Carlo Dropout:** keep dropout switched **on** at prediction time and predict 30 times. If the answers vary a lot, the model is unsure; the 5th to 95th percentile gives the confidence range shown to the user.
* **"The model is unsure" rule:** shown when top-1 probability is below 50%, the gap between first and second is below 10 points, or fewer than 3 symptoms were given. All thresholds are in `config.py`.
* **Red-flag layer:** plain rules, independent of the model: chest pain, breathing difficulty, loss of consciousness, stroke signs and similar. It checks selected symptoms and typed text, ignores negations like "no chest pain", and shows an emergency banner that overrides the prediction. Rules are used because a safety net must be predictable, not learned.
* **Disclaimer:** shown permanently; the app is educational, not a diagnosis.

## 9. App and engineering

* **Streamlit:** turns Python scripts into a web app; four pages (Check Symptoms, Results, Model Performance, About & Limitations).
* **`st.cache_resource`:** loads the model once and shares it, so each click is fast.
* **Training is separate from the app:** the app only loads saved files, so it starts quickly and cannot change the model.
* **PDF report:** built with ReportLab from the same result object the page shows.
* **Tests:** pytest checks cleaning, duplicate removal, split, prediction shapes, red flags, explanation text, follow-ups and the home-care file (no medicine doses).
* **Reproducibility:** fixed seeds, pinned package versions, one `config.py`, one command `run_all.sh`.
* **Optional extras:** DistilBERT (a compact language model fine-tuned on Symptom2Disease) for free-text input; a multi-label experiment (sigmoid outputs with binary cross-entropy) on **synthetic** two-disease patients, documented as an experiment only.

---

## 10. Hard questions and honest answers

**Q1. Your accuracy is almost 100%. Isn't that overfitting?**
Overfitting means doing well on training data but badly on unseen data. My test split is held out, with duplicates removed before splitting. The near-perfect score comes from the dataset: each disease is a small set of textbook symptom lists, so even logistic regression scores about the same. I checked this directly: train, validation and test scores are compared, and under stricter tests (novel patterns, wrong or missing symptoms) accuracy drops gradually, as it should. I do not claim clinical accuracy.

**Q2. Then why is the dataset so easy?**
It is largely synthetic. It does not contain the overlap and noise of real patients. That is the first limitation I list, and validation on de-identified real clinical data is my first future-work item.

**Q3. What did you do about data leakage?**
Removed exact duplicates before splitting, split with stratification and a fixed seed, and measured near-duplicate (subset) rows. The novel-pattern test keeps only test patients that are not near-copies of a training patient.

**Q4. Why a neural network if logistic regression is as accurate?**
Accuracy is not the only goal. The MLP lets me demonstrate calibration, Monte Carlo Dropout uncertainty and gradient-based explanations, and I compared it fairly with three baselines. On this dataset the gap in accuracy is small; I report that openly.

**Q5. Why should anyone trust the percentages?**
They are temperature-scaled and I report ECE before and after scaling, plus a reliability diagram. They are still estimates from a small validation set, which I state as a limitation.

**Q6. Do your explanations prove why a disease occurs?**
No. They show what the *model* relies on, not medical causation. That is why I compare them with the dataset's own disease-symptom map in the sanity check.

**Q7. What happens if the user is in real danger?**
The red-flag rules run first and independently of the model, and show an emergency banner that overrides the prediction. The rule set is conservative and simple, and would need clinician review before real use.

**Q8. What if the model is not confident?**
It says "The model is unsure, please consult a doctor" instead of a confident answer, and shows a confidence range from Monte Carlo Dropout.

**Q9. Where do the home-care tips come from?**
No public dataset exists, so I summarised pages from NHS, MedlinePlus and Mayo Clinic. Each entry keeps its source link, has only mild measures with no medicine doses, and always says when to see a doctor. Entries show "reviewed against the source" only after I checked the link and marked them verified.

**Q10. Why don't age and sex change the prediction?**
The dataset has no age or sex information, so the model cannot learn from them. They appear only in the report. Using them without data would be an unsupported assumption.

**Q11. Is it biased or private?**
No real patient data is used and typed input is not stored. Fairness across groups cannot be tested because the dataset has no demographic columns; that is listed as future work.

**Q12. Could this be used in a hospital?**
No. It is an academic proof of concept. Real use would need clinical data, validation, regulatory approval and clinician oversight.

**Q13. What would you improve next?**
Real clinical data; three-state symptom input (present / absent / unknown); deep ensembles or conformal prediction for uncertainty; clinician review of rules and explanations; more diseases; a fairness audit.

**Q14. How can I verify your numbers?**
Run `python -m src.evaluate` and `python -m src.evaluate_hard`; every table in the README and the app is produced from files in `reports/`, and `run_all.sh` reproduces everything with fixed seeds.

---

## 11. Demo script (about 3 minutes)

1. Open **Check Symptoms**; pick 4 to 5 symptoms (for example chills, high fever, sweating, headache); press **Predict Disease**.
2. On **Results**, show the top-3 bars and the confidence range; read the plain-language explanation; point to green and red bars.
3. Show the **counterfactual** table, then click a **follow-up** symptom and watch the prediction update.
4. Type "severe chest pain" to show the **emergency banner**.
5. Download the **PDF report**.
6. Open **Model Performance** (comparison, calibration, robustness, stricter tests) and **About & Limitations**; finish by stating one limitation and how you handled it.
