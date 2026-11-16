# Sentiment Classifier

![Tests](https://github.com/Jingier34/sentiment-classifier/actions/workflows/tests.yml/badge.svg)

A sentiment classification service for movie reviews. Trains and compares classical machine learning models on the IMDB dataset, then serves the best model through a FastAPI endpoint that returns the predicted sentiment and a confidence score.

## Results

Models were compared on a validation split held out from the training data (20%, stratified). The test set was used only once, for the final evaluation of the selected model.

| Model | Features | Validation accuracy |
|---|---|---|
| Logistic regression | Unigrams | 88.92% |
| Logistic regression | Unigrams + bigrams | 89.52% |
| Linear SVM | Unigrams + bigrams | 90.68% |
| **Calibrated linear SVM** | Unigrams + bigrams | **90.64%** |
| Naive Bayes | Unigrams + bigrams | 87.72% |

The calibrated linear SVM was selected because it matches the plain SVM's accuracy while also producing probabilities, which the API uses as confidence scores. It reaches **90.04% accuracy on the 25,000-review test set**, compared with 88.37% for the unigram logistic regression baseline.

## Error Analysis

- **Bigrams help with negation.** Reviews containing phrases like "not bad" or "not good" are classified slightly more accurately (91.0%) than the rest (90.0%), since bigram features capture these phrases as units.
- **Errors are not related to review length.** Correct and incorrect predictions have nearly the same median length (171 vs. 177 words).
- **Some confident mistakes reflect label noise.** Several reviews the model misclassifies with near-total confidence have text that contradicts their label, for example a review labeled positive that calls the film "such a disappointment". IMDB labels are derived from star ratings, which do not always match the written review.
- **Reviews dominated by plot summary are harder.** When most of a review describes the plot or the cast's history, the opinion is diluted by neutral text.

## Design Notes

- **Stop words are kept.** Common stop word lists include negations such as "not" and "no", which carry strong sentiment signal.
- **HTML is stripped during preprocessing.** 58.7% of reviews contain HTML line-break tags.
- **The model is loaded once** on the first request and reused, rather than reloaded per request.
- **API tests use a small in-memory model**, so they run quickly and do not depend on the trained model file, which is not committed to the repository.

## Setup

```bash
git clone https://github.com/Jingier34/sentiment-classifier.git
cd sentiment-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python explore_data.py      # download the dataset and print summary statistics
python experiment.py        # compare models and save the best one to models/best.joblib
uvicorn app:app --reload    # start the API
```

Then open http://127.0.0.1:8000/docs for the interactive API docs.

## API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Body: `{"text": "..."}`. Returns `label` (`positive` or `negative`) and `confidence` (0.5 to 1.0) |
| GET | `/health` | Health check |

Input is rejected with 400 if empty or longer than 10,000 characters, and the API returns 503 if no trained model is available.

## Project Structure

| File | Purpose |
|---|---|
| `explore_data.py` | Downloads the IMDB dataset and prints summary statistics |
| `preprocess.py` | Text cleaning |
| `train.py` | Trains the baseline model and prints its most influential words |
| `experiment.py` | Compares candidate models and saves the best one |
| `analyze_errors.py` | Breaks down the best model's mistakes on the test set |
| `app.py` | FastAPI prediction service |
| `test_preprocess.py`, `test_app.py` | Unit and API tests |

## Running Tests

```bash
pytest -v
```

Tests run automatically on every push via GitHub Actions.