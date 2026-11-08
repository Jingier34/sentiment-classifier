import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from preprocess import clean_text

DATA_DIR = Path("data")
MODEL_DIR = Path("models")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train["text"].map(clean_text)
    X_test = test["text"].map(clean_text)
    y_train = train["label"]
    y_test = test["label"]

    # Baseline: single-word TF-IDF features + logistic regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"Training time: {time.time() - start:.1f}s")

    predictions = pipeline.predict(X_test)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["negative", "positive"]))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))

    # Inspect which words push predictions most strongly in each direction
    words = pipeline.named_steps["tfidf"].get_feature_names_out()
    weights = pipeline.named_steps["clf"].coef_[0]
    order = weights.argsort()
    print("\nMost negative words:", ", ".join(words[order[:15]]))
    print("Most positive words:", ", ".join(words[order[-15:][::-1]]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "baseline.joblib")
    print(f"\nModel saved to {MODEL_DIR / 'baseline.joblib'}")


if __name__ == "__main__":
    main()