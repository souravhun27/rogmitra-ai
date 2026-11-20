"""OPTIONAL: fine-tune DistilBERT on Symptom2Disease (free-text symptom descriptions).

Run::

    python -m src.train_text_model                 # ~5-15 min on a laptop CPU
    python -m src.train_text_model --skip          # do nothing (handy in scripts)
    python -m src.train_text_model --epochs 2 --max-train-samples 300   # quick trial

Needs ``data/raw/Symptom2Disease.csv`` (``python -m src.download_data --text``)
and internet access the first time (to download the pretrained DistilBERT
weights from Hugging Face). Note: DistilBERT is a general language model that
we FINE-TUNE here on symptom text; it is not a medical model, and it is only
used for the optional "second opinion" box, never for the main predictions.

CPU-friendly choices: max length 64 tokens, 4 epochs, batch 16, a plain PyTorch
loop (no Trainer/accelerate). A TF-IDF + Logistic Regression baseline is trained
on the same split so you can see whether the transformer is actually worth it.
"""
from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src import config
from src.calibration import fit_temperature
from src.download_data import find_file
from src.evaluate import compute_metrics, plot_confusion
from src.models import softmax
from src.preprocess import clean_disease, stratified_split
from src.utils import ensure_dir, load_json, save_json, set_seed, setup_logging

logger = logging.getLogger(__name__)


def load_text_dataset(raw_dir: Path = config.RAW_DIR) -> tuple[pd.DataFrame, int]:
    """Load, clean and de-duplicate Symptom2Disease; returns ``(frame[text,label], n_duplicates)``.

    Labels are mapped to the structured dataset's spelling when possible (e.g.
    'Common Cold' -> 'Common cold') so both models speak about the same diseases.
    """
    path = find_file(raw_dir, config.RAW_TEXT_FILE)
    if path is None:
        raise FileNotFoundError(
            f"{config.RAW_TEXT_FILE} not found in {raw_dir}. Run: python -m src.download_data --text "
            "(or copy the file from https://www.kaggle.com/datasets/niyarrbarman/symptom2disease)"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = {c.lower().strip(): c for c in df.columns}
    if "text" not in cols or "label" not in cols:
        raise ValueError(f"Expected columns 'label' and 'text', found {list(df.columns)}")
    df = df[[cols["label"], cols["text"]]].rename(columns={cols["label"]: "label", cols["text"]: "text"}).dropna()

    canon: dict[str, str] = {}
    if config.CLASSES_JSON.exists():
        canon = {c.lower(): c for c in load_json(config.CLASSES_JSON)}

    def canonical(label: str) -> str:
        cleaned = clean_disease(label)
        return canon.get(cleaned.lower(), cleaned[:1].upper() + cleaned[1:])

    df["label"] = df["label"].map(canonical)
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    before = len(df)
    # De-duplicate BEFORE splitting for the same leakage reason as the structured data.
    df = df.assign(_key=df["text"].str.lower()).drop_duplicates("_key").drop(columns="_key").reset_index(drop=True)
    return df, before - len(df)


def encode_batch(tokenizer, texts: list[str], max_len: int):
    """Tokenise a list of texts into padded PyTorch tensors."""
    return tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def predict_logits(model, tokenizer, texts: list[str], max_len: int, batch_size: int = 64) -> np.ndarray:
    """Logits for many texts (eval mode, no gradients)."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            out.append(model(**encode_batch(tokenizer, texts[i : i + batch_size], max_len)).logits.numpy())
    return np.vstack(out)


def fine_tune(
    model_name: str, texts_tr: list[str], y_tr: np.ndarray, texts_va: list[str], y_va: np.ndarray,
    labels: list[str], epochs: int, batch_size: int, max_len: int, lr: float, seed: int,
):
    """Fine-tune and return ``(model, tokenizer, history)`` with the best-validation weights restored."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), ignore_mismatched_sizes=True,  # new head is randomly initialised anyway
        id2label={i: l for i, l in enumerate(labels)}, label2id={l: i for i, l in enumerate(labels)},
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(texts_tr) / batch_size))
    # Linear warm-up then decay: the standard, stable recipe for fine-tuning transformers.
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = np.random.default_rng(seed)
    best_acc, best_loss, best_state, history = -1.0, float("inf"), None, []

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(texts_tr))
        running, start_t = 0.0, time.time()
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            batch = encode_batch(tokenizer, [texts_tr[j] for j in idx], max_len)
            loss = model(**batch, labels=torch.as_tensor(y_tr[idx])).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guards against rare exploding gradients
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item() * len(idx)
        logits = predict_logits(model, tokenizer, texts_va, max_len)
        val_acc = float((logits.argmax(1) == y_va).mean())
        z = logits - logits.max(1, keepdims=True)
        val_loss = float(-(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(len(y_va)), y_va].mean())
        history.append({"epoch": epoch, "train_loss": running / len(order), "val_acc": val_acc, "val_loss": val_loss})
        logger.info("epoch %d/%d | train loss %.3f | val acc %.3f | val loss %.3f | %.0fs",
                    epoch, epochs, running / len(order), val_acc, val_loss, time.time() - start_t)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, tokenizer, history


def tfidf_baseline(texts_tr, y_tr, texts_te, n_classes: int, seed: int) -> np.ndarray:
    """TF-IDF (1-2 grams) + Logistic Regression probabilities: the classical text baseline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    clf = LogisticRegression(max_iter=3000, C=10.0, random_state=seed).fit(vec.fit_transform(texts_tr), y_tr)
    part = clf.predict_proba(vec.transform(texts_te))
    out = np.zeros((len(texts_te), n_classes))
    out[:, clf.classes_] = part
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Symptom2Disease (optional).")
    parser.add_argument("--skip", action="store_true", help="skip text-model training entirely")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--model-name", default=config.TEXT_MODEL_NAME, help="HF model id or local folder")
    parser.add_argument("--epochs", type=int, default=config.TEXT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TEXT_BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=config.TEXT_MAX_LEN)
    parser.add_argument("--lr", type=float, default=config.TEXT_LR)
    parser.add_argument("--max-train-samples", type=int, default=0, help="use only N training rows (quick trial)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)
    setup_logging()
    if args.skip:
        logger.info("Text model skipped (--skip).")
        return 0
    try:
        import transformers  # noqa: F401
    except ImportError:
        logger.error("transformers is not installed. Run: pip install transformers (or rerun with --skip)")
        return 1

    try:
        df, n_dup = load_text_dataset(args.raw_dir)
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return 1
    labels = sorted(df["label"].unique())
    y = df["label"].map({l: i for i, l in enumerate(labels)}).to_numpy()
    tr, va, te = stratified_split(y, seed=args.seed)
    if args.max_train_samples:
        tr = tr[: args.max_train_samples]
    texts = df["text"].tolist()
    pick = lambda idx: [texts[i] for i in idx]  # noqa: E731
    logger.info("Rows: %d (removed %d duplicate texts) | classes: %d | train/val/test = %d/%d/%d",
                len(df), n_dup, len(labels), len(tr), len(va), len(te))

    start = time.time()
    model, tok, history = fine_tune(args.model_name, pick(tr), y[tr], pick(va), y[va], labels,
                                    args.epochs, args.batch_size, args.max_len, args.lr, args.seed)
    logger.info("Fine-tuning finished in %.0fs", time.time() - start)

    # Temperature scaling on validation logits, exactly as for the structured MLP.
    temperature = fit_temperature(predict_logits(model, tok, pick(va), args.max_len), y[va])
    test_logits = predict_logits(model, tok, pick(te), args.max_len)
    p_bert = softmax(test_logits, temperature)
    p_tfidf = tfidf_baseline(pick(tr), y[tr], pick(te), len(labels), args.seed)
    results = {"distilbert": compute_metrics(y[te], p_bert, len(labels)),
               "tfidf_logreg": compute_metrics(y[te], p_tfidf, len(labels))}
    for name, m in results.items():
        logger.info("%-13s acc %.3f | macro-F1 %.3f | top-3 %.3f | ECE %.3f", name, m["accuracy"], m["macro_f1"],
                    m["top3_accuracy"], m["ece"])

    out_dir = ensure_dir(config.TEXT_MODEL_DIR)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    save_json({"labels": labels, "temperature": temperature, "max_len": args.max_len,
               "base_model": args.model_name, "seed": args.seed}, out_dir / "explaindx_text_meta.json")

    save_json({"n_rows": len(df), "n_duplicates_removed": n_dup, "n_classes": len(labels),
               "n_test": len(te), "temperature": temperature, "history": history, "results": results},
              config.TEXT_METRICS_JSON)
    rep = classification_report(y[te], p_bert.argmax(1), labels=range(len(labels)), target_names=labels,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(config.REPORTS_DIR / "classification_report_text.csv")
    plot_confusion(y[te], p_bert.argmax(1), labels, config.FIGURES_DIR / "confusion_matrix_text.png")
    lines = ["# Free-text model (Symptom2Disease)", "",
             f"{len(labels)} diseases, {len(te)} test texts. This is a DIFFERENT dataset and label set from the structured "
             "model, so these numbers are not directly comparable with `metrics.md`.", "",
             "| Model | Accuracy | Macro F1 | Top-3 | ECE |", "|---|---|---|---|---|"]
    for name, label in (("distilbert", "DistilBERT (fine-tuned, calibrated)"), ("tfidf_logreg", "TF-IDF + Logistic Regression")):
        m = results[name]
        lines.append(f"| {label} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['top3_accuracy']:.3f} | {m['ece']:.3f} |")
    config.TEXT_METRICS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved model to %s and reports to reports/text_model*.", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
