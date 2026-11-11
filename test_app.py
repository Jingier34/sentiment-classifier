import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import app as app_module


@pytest.fixture
def client():
    # A tiny model trained in place, so tests don't depend on the real model file
    texts = [
        "great movie loved it", "wonderful and amazing", "excellent film",
        "terrible movie hated it", "awful and boring", "worst film ever",
    ]
    labels = [1, 1, 1, 0, 0, 0]
    model = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())])
    model.fit(texts, labels)
    app_module._model = model
    yield TestClient(app_module.app)
    app_module._model = None


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_positive(client):
    response = client.post("/predict", json={"text": "A great and wonderful movie"})
    assert response.status_code == 200
    assert response.json()["label"] == "positive"


def test_predict_negative(client):
    response = client.post("/predict", json={"text": "Terrible and boring"})
    assert response.status_code == 200
    assert response.json()["label"] == "negative"


def test_predict_returns_confidence(client):
    confidence = client.post("/predict", json={"text": "great movie"}).json()["confidence"]
    assert 0.5 <= confidence <= 1.0


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 400


def test_predict_rejects_too_long_text(client):
    response = client.post("/predict", json={"text": "a" * (app_module.MAX_TEXT_LENGTH + 1)})
    assert response.status_code == 400


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(app_module, "_model", None)
    monkeypatch.setattr(app_module, "MODEL_PATH", app_module.Path("does/not/exist.joblib"))
    client = TestClient(app_module.app)
    response = client.post("/predict", json={"text": "great movie"})
    assert response.status_code == 503