# Convenience targets. On Windows use Git Bash / WSL, or run the python commands directly.
PY ?= python

.PHONY: all install download preprocess eda baselines mlp evaluate hard explain sanity remedies text multilabel app test clean

install:
	$(PY) -m pip install -r requirements.txt

# Needs ~/.kaggle/kaggle.json, or place the CSVs in data/raw/ manually (see README).
download:
	$(PY) -m src.download_data

preprocess:
	$(PY) -m src.preprocess

eda:
	$(PY) -m src.eda

baselines:
	$(PY) -m src.train_baselines

mlp:
	$(PY) -m src.train_mlp

evaluate:
	$(PY) -m src.evaluate

hard:
	$(PY) -m src.evaluate_hard

all:
	bash run_all.sh

explain:
	$(PY) -m src.explain

sanity:
	$(PY) -m src.sanity_check

remedies:
	$(PY) -m src.check_remedies --online

app:
	streamlit run app/Home.py

text:
	$(PY) -m src.train_text_model

multilabel:
	$(PY) -m src.train_multilabel

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/processed/*.csv data/processed/*.json reports/figures/*.png
