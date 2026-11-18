import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# The previous default (50,000 features, C=1.0) is included, so tuning can only match or beat it
PARAM_GRID = {
    "tfidf__max_features": [50000, 100000, 200000],
    "clf__C": [0.1, 0.3, 1.0, 3.0],
}


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X = train["text"].map(clean_text)
    y = train["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LinearSVC()),
    ])

    # 3-fold cross-validation on the training set only
    search = GridSearchCV(pipeline, PARAM_GRID, cv=3, scoring="accuracy", n_jobs=-1)
    start = time.time()
    search.fit(X, y)
    print(f"Grid search finished in {time.time() - start:.0f}s\n")

    results = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
    print("Cross-validation accuracy by setting:")
    for _, row in results.iterrows():
        params = row["params"]
        print(
            f"  max_features={params['tfidf__max_features']:<7} C={params['clf__C']:<4}"
            f" accuracy={row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})"
        )

    best_features = search.best_params_["tfidf__max_features"]
    best_c = search.best_params_["clf__C"]
    print(f"\nBest setting: max_features={best_features}, C={best_c}")

    # Rebuild the best setting with calibration so the API can report confidence
    final_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=best_features)),
        ("clf", CalibratedClassifierCV(LinearSVC(C=best_c), cv=3)),
    ])
    final_model.fit(X, y)
    test_accuracy = accuracy_score(test["label"], final_model.predict(test["text"].map(clean_text)))
    print(f"Test accuracy of tuned, calibrated model: {test_accuracy:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "best.joblib")
    print(f"Model saved to {MODEL_DIR / 'best.joblib'}")


if __name__ == "__main__":
    main()