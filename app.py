from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from preprocess import clean_text

app = FastAPI()

MODEL_PATH = Path("models/best.joblib")
MAX_TEXT_LENGTH = 10000
TOP_N_WORDS = 5
_model = None


def get_model():
    # Load the model once on first use and reuse it for later requests
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not available. Run tune.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_feature_weights(model):
    """Return one weight per vocabulary term, or None if the classifier is not linear."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return clf.coef_[0]
    # A calibrated classifier holds several fitted SVMs; average their weights
    if hasattr(clf, "calibrated_classifiers_"):
        return np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    return None


def explain(model, cleaned):
    """List the words in the text that pushed the prediction most strongly each way."""
    weights = get_feature_weights(model)
    if weights is None:
        return None
    tfidf = model.named_steps["tfidf"]
    vector = tfidf.transform([cleaned])
    terms = tfidf.get_feature_names_out()[vector.indices]
    # Each term's contribution is its TF-IDF value times the model's weight for it
    contributions = vector.data * weights[vector.indices]
    ranked = sorted(zip(terms, contributions), key=lambda pair: pair[1])

    def to_list(pairs):
        return [{"term": term, "weight": round(float(weight), 4)} for term, weight in pairs]

    negative = [pair for pair in ranked[:TOP_N_WORDS] if pair[1] < 0]
    positive = [pair for pair in reversed(ranked[-TOP_N_WORDS:]) if pair[1] > 0]
    return {"positive": to_list(positive), "negative": to_list(negative)}


class PredictRequest(BaseModel):
    text: str


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text must be at most {MAX_TEXT_LENGTH} characters")

    model = get_model()
    cleaned = clean_text(text)
    label = int(model.predict([cleaned])[0])
    result = {"label": "positive" if label == 1 else "negative"}

    # Linear SVM has no probability output, so confidence is only returned when available
    if hasattr(model, "predict_proba"):
        result["confidence"] = round(float(max(model.predict_proba([cleaned])[0])), 4)

    explanation = explain(model, cleaned)
    if explanation is not None:
        result["explanation"] = explanation
    return result