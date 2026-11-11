from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from preprocess import clean_text

app = FastAPI()

MODEL_PATH = Path("models/best.joblib")
MAX_TEXT_LENGTH = 10000
_model = None


def get_model():
    # Load the model once on first use and reuse it for later requests
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not available. Run experiment.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    text: str


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
    return result