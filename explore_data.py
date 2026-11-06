from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path("data")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # IMDB movie reviews: 25,000 for training and 25,000 for testing
    dataset = load_dataset("stanfordnlp/imdb")
    train = dataset["train"].to_pandas()
    test = dataset["test"].to_pandas()

    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")

    print("\nLabel distribution (0 = negative, 1 = positive):")
    print(train["label"].value_counts().sort_index())

    lengths = train["text"].str.split().str.len()
    print("\nReview length in words:")
    print(lengths.describe().round(1))

    print("\nSample reviews:")
    for label in (0, 1):
        sample = train[train["label"] == label].iloc[0]["text"]
        print(f"\n[label={label}] {sample[:300]}...")

    html_share = train["text"].str.contains("<br").mean()
    print(f"\nShare of reviews containing <br /> HTML tags: {html_share:.1%}")


if __name__ == "__main__":
    main()