from pathlib import Path

import joblib
import pandas as pd

from preprocess import clean_text

DATA_DIR = Path("data")
MODEL_PATH = Path("models/best.joblib")
NEGATION_PATTERN = r"\bnot (?:bad|good|great|terrible|funny|worth)\b"


def main():
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(DATA_DIR / "test.csv")
    cleaned = test["text"].map(clean_text)

    test["predicted"] = model.predict(cleaned)
    # Signed score: positive means the model leans positive; larger magnitude means more confident
    test["score"] = model.decision_function(cleaned)
    test["length"] = test["text"].str.split().str.len()

    correct = test[test["predicted"] == test["label"]]
    errors = test[test["predicted"] != test["label"]]
    print(f"Misclassified: {len(errors)} of {len(test)} ({len(errors) / len(test):.1%})")
    print(f"  Negative reviews predicted positive: {(errors['label'] == 0).sum()}")
    print(f"  Positive reviews predicted negative: {(errors['label'] == 1).sum()}")

    print(f"\nMedian length, correct predictions: {correct['length'].median():.0f} words")
    print(f"Median length, wrong predictions:   {errors['length'].median():.0f} words")

    has_negation = cleaned.str.contains(NEGATION_PATTERN)
    negation_accuracy = (test.loc[has_negation, "predicted"] == test.loc[has_negation, "label"]).mean()
    other_accuracy = (test.loc[~has_negation, "predicted"] == test.loc[~has_negation, "label"]).mean()
    print(f"\nReviews with phrases like 'not bad': {has_negation.sum()}")
    print(f"  Accuracy on those:     {negation_accuracy:.4f}")
    print(f"  Accuracy on the rest:  {other_accuracy:.4f}")

    print("\nMost confident mistakes:")
    worst = errors.assign(confidence=errors["score"].abs()).sort_values("confidence", ascending=False)
    for _, row in worst.head(5).iterrows():
        actual = "positive" if row["label"] == 1 else "negative"
        print(f"\n[actual: {actual}, score: {row['score']:.2f}] {row['text'][:300]}...")

    errors.to_csv(DATA_DIR / "errors.csv", index=False)
    print(f"\nAll misclassified reviews saved to {DATA_DIR / 'errors.csv'}")


if __name__ == "__main__":
    main()