import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

DATA_DIR = Path("data")
MODEL_DIR = Path("models")


def build_candidates():
    return {
        "LogReg, unigrams": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "LogReg, uni+bigrams": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "LinearSVC, uni+bigrams": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LinearSVC()),
        ]),
        "Naive Bayes, uni+bigrams": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", MultinomialNB()),
        ]),
    }


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X = train["text"].map(clean_text)
    y = train["label"]

    # Hold out 20% of the training data for model selection; the test set stays untouched
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    for name, pipeline in build_candidates().items():
        start = time.time()
        pipeline.fit(X_train, y_train)
        accuracy = accuracy_score(y_val, pipeline.predict(X_val))
        elapsed = time.time() - start
        results.append((name, accuracy, elapsed))
        print(f"{name:<28} val accuracy: {accuracy:.4f}  ({elapsed:.1f}s)")

    best_name = max(results, key=lambda r: r[1])[0]
    print(f"\nBest on validation: {best_name}")

    # Retrain the chosen model on all training data, then evaluate once on the test set
    best = build_candidates()[best_name]
    best.fit(X, y)
    test_accuracy = accuracy_score(test["label"], best.predict(test["text"].map(clean_text)))
    print(f"Test accuracy of {best_name}: {test_accuracy:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_DIR / "best.joblib")
    print(f"Model saved to {MODEL_DIR / 'best.joblib'}")


if __name__ == "__main__":
    main()