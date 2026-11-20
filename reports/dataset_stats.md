# Dataset statistics

- Diseases: **41**, symptoms (features): **131**
- Raw rows: **4920**; exact duplicates removed: **4616** (93.8%); unique rows: **304**
- Symptom vectors shared by several diseases (ambiguous): **0**; rows dropped for this reason: **0**
- Rows that are a strict subset of another row of the same disease: **263**
- Final rows: **304** -> split sizes {'train': 214, 'val': 45, 'test': 45}
- Unique rows per disease (min / median / max): 5 / 7.0 / 10
- Symptoms per unique row (min / mean / max): 3 / 7.65 / 17

> High model accuracy on this dataset reflects its simplicity and synthetic origin, NOT clinical reliability.

| Disease | Raw rows | Unique rows | Distinct symptoms | Mean symptoms/row |
|---|---:|---:|---:|---:|
| AIDS | 120 | 5 | 4 | 3.2 |
| Acne | 120 | 5 | 4 | 3.2 |
| Alcoholic hepatitis | 120 | 8 | 7 | 6.12 |
| Allergy | 120 | 5 | 4 | 3.2 |
| Arthritis | 120 | 6 | 5 | 4.17 |
| Bronchial asthma | 120 | 7 | 6 | 5.14 |
| Cervical spondylosis | 120 | 6 | 5 | 4.17 |
| Chickenpox | 120 | 10 | 11 | 10.1 |
| Chronic cholestasis | 120 | 8 | 7 | 6.12 |
| Common cold | 120 | 9 | 17 | 16.11 |
| Dengue | 120 | 10 | 14 | 13.1 |
| Diabetes | 120 | 9 | 10 | 9.11 |
| Drug reaction | 120 | 6 | 5 | 4.17 |
| Fungal infection | 120 | 5 | 4 | 3.2 |
| GERD | 120 | 7 | 6 | 5.14 |
| Gastroenteritis | 120 | 5 | 4 | 3.2 |
| Heart attack | 120 | 5 | 4 | 3.2 |
| Hemorrhoids (piles) | 120 | 6 | 5 | 4.17 |
| Hepatitis A | 120 | 9 | 11 | 10.11 |
| Hepatitis B | 120 | 9 | 12 | 11.11 |
| Hepatitis C | 120 | 7 | 6 | 5.14 |
| Hepatitis D | 120 | 10 | 9 | 8.1 |
| Hepatitis E | 120 | 9 | 13 | 12.11 |
| Hypertension | 120 | 6 | 5 | 4.17 |
| Hyperthyroidism | 120 | 9 | 11 | 10.11 |
| Hypoglycemia | 120 | 9 | 12 | 11.11 |
| Hypothyroidism | 120 | 8 | 13 | 12.12 |
| Impetigo | 120 | 6 | 5 | 4.17 |
| Jaundice | 120 | 9 | 8 | 7.11 |
| Malaria | 120 | 8 | 8 | 7.12 |
| Migraine | 120 | 10 | 9 | 8.1 |
| Osteoarthritis | 120 | 7 | 6 | 5.14 |
| Paralysis (brain hemorrhage) | 120 | 5 | 4 | 3.2 |
| Paroxysmal positional vertigo | 120 | 7 | 6 | 5.14 |
| Peptic ulcer disease | 120 | 7 | 6 | 5.14 |
| Pneumonia | 120 | 9 | 11 | 10.11 |
| Psoriasis | 120 | 7 | 6 | 5.14 |
| Tuberculosis | 120 | 9 | 16 | 15.11 |
| Typhoid | 120 | 9 | 11 | 10.11 |
| Urinary tract infection | 120 | 5 | 4 | 3.2 |
| Varicose veins | 120 | 8 | 7 | 6.12 |
