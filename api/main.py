"""
FastAPI inference API for Twitter Customer Support Classifier.
Endpoints:
  POST /predict        — classify a single tweet
  POST /predict/batch  — classify a list of tweets
  GET  /health         — liveness check
  GET  /model/info     — model metadata
  GET  /logs           — tail recent log lines
"""

import re
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logger import get_logger, LOGS_DIR  # noqa: E402

log = get_logger("api")

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"

# ── NLTK setup ────────────────────────────────────────────────────────────────
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def _clean(text: str) -> str:
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)


# ── App lifespan — load artifacts once at startup ────────────────────────────
model = None
vectorizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    log.info("Loading artifacts from %s", ARTIFACTS_DIR)
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise RuntimeError(f"Artifacts not found in {ARTIFACTS_DIR}. Run train_pipeline.py first.")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    log.info("Startup complete | model=%s features=%d", type(model).__name__, vectorizer.max_features)
    yield
    log.info("Shutting down API.")


app = FastAPI(
    title="Twitter Customer Support Classifier",
    description="Classifies customer tweets to the most likely support team.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, example="My order hasn't arrived and tracking shows no update")


class PredictResponse(BaseModel):
    text: str
    predicted_company: str
    confidence: float
    all_scores: dict[str, float]


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100)


class BatchResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    classes: list[str]
    num_features: int
    artifact_dir: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    log.info("Health check | model_loaded=%s", model is not None)
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "classes": list(model.classes_),
        "num_features": vectorizer.max_features,
        "artifact_dir": str(ARTIFACTS_DIR),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    clean = _clean(req.text)
    if not clean.strip():
        raise HTTPException(status_code=422, detail="Text is empty after cleaning.")

    X = vectorizer.transform([clean])
    predicted = model.predict(X)[0]
    probas = model.predict_proba(X)[0]
    scores = {cls: round(float(p), 4) for cls, p in zip(model.classes_, probas)}
    confidence = round(float(max(probas)), 4)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    log.info(
        "predict | company=%s confidence=%.4f elapsed_ms=%.1f text=%.60r",
        predicted, confidence, elapsed_ms, req.text,
    )

    return {
        "text": req.text,
        "predicted_company": predicted,
        "confidence": confidence,
        "all_scores": dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
    }


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(req: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.time()
    results = []
    for text in req.texts:
        clean = _clean(text)
        if not clean.strip():
            results.append({
                "text": text,
                "predicted_company": "unknown",
                "confidence": 0.0,
                "all_scores": {},
            })
            continue
        X = vectorizer.transform([clean])
        predicted = model.predict(X)[0]
        probas = model.predict_proba(X)[0]
        scores = {cls: round(float(p), 4) for cls, p in zip(model.classes_, probas)}
        results.append({
            "text": text,
            "predicted_company": predicted,
            "confidence": round(float(max(probas)), 4),
            "all_scores": dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
        })

    elapsed_ms = round((time.time() - t0) * 1000, 1)
    log.info("predict_batch | count=%d elapsed_ms=%.1f", len(req.texts), elapsed_ms)

    return {"results": results}


@app.get("/logs", tags=["System"])
def get_logs(source: str = Query("api", enum=["api", "train"]), lines: int = Query(50, ge=1, le=500)):
    """Return the last N lines from a log file as parsed JSON entries."""
    log_file = LOGS_DIR / f"{source}.log"
    if not log_file.exists():
        return {"source": source, "entries": []}

    raw = log_file.read_text().splitlines()
    tail = raw[-lines:]

    entries = []
    for line in tail:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"msg": line})

    return {"source": source, "entries": entries}
