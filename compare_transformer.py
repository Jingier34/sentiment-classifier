import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import pipeline

from preprocess import clean_text

DATA_DIR = Path("data")
MODEL_PATH = Path("models/best.joblib")
TRANSFORMER_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SAMPLE_SIZE = 2000
MAX_TOKENS = 512


def main():
    test = pd.read_csv(DATA_DIR / "test.csv")
    # Balanced random sample, so both models are compared on exactly the same reviews
    sample = test.groupby("label", group_keys=False).sample(n=SAMPLE_SIZE // 2, random_state=42)
    texts = sample["text"].map(clean_text).tolist()
    labels = sample["label"].tolist()

    tfidf_model = joblib.load(MODEL_PATH)
    start = time.time()
    tfidf_preds = list(tfidf_model.predict(texts))
    tfidf_time = time.time() - start

    # Use the Apple GPU if available, otherwise fall back to the CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running the transformer on: {device}")
    classifier = pipeline("sentiment-analysis", model=TRANSFORMER_ID, device=device)
    start = time.time()
    outputs = classifier(texts, batch_size=16, truncation=True, max_length=MAX_TOKENS)
    transformer_time = time.time() - start
    transformer_preds = [1 if o["label"] == "POSITIVE" else 0 for o in outputs]

    print(f"\nResults on {SAMPLE_SIZE} test reviews:")
    for name, preds, elapsed in [
        ("TF-IDF + calibrated SVM", tfidf_preds, tfidf_time),
        ("DistilBERT (SST-2)", transformer_preds, transformer_time),
    ]:
        accuracy = accuracy_score(labels, preds)
        per_review_ms = elapsed / SAMPLE_SIZE * 1000
        print(f"  {name:<26} accuracy: {accuracy:.4f}   time: {elapsed:.1f}s ({per_review_ms:.2f} ms/review)")

    # Where do the two models disagree, and which one is usually right?
    disagree = [i for i in range(SAMPLE_SIZE) if tfidf_preds[i] != transformer_preds[i]]
    tfidf_right = sum(tfidf_preds[i] == labels[i] for i in disagree)
    print(f"\nThe models disagree on {len(disagree)} reviews ({len(disagree) / SAMPLE_SIZE:.1%})")
    print(f"  TF-IDF model right: {tfidf_right}")
    print(f"  DistilBERT right:   {len(disagree) - tfidf_right}")

    # Reviews longer than the token limit are cut off, so the transformer only sees the beginning
    token_counts = [len(classifier.tokenizer(t)["input_ids"]) for t in texts]
    truncated = [count > MAX_TOKENS for count in token_counts]
    print(f"\nReviews truncated to {MAX_TOKENS} tokens: {sum(truncated)} ({sum(truncated) / SAMPLE_SIZE:.1%})")
    for name, flag in [("truncated", True), ("not truncated", False)]:
        idx = [i for i in range(SAMPLE_SIZE) if truncated[i] == flag]
        if idx:
            acc = accuracy_score([labels[i] for i in idx], [transformer_preds[i] for i in idx])
            print(f"  DistilBERT accuracy on {name} reviews: {acc:.4f}")


if __name__ == "__main__":
    main()