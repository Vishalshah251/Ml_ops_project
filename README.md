# Customer Support Ticket Classifier — MLOps Project

[![CI/CD](https://github.com/Vishalshah251/Ml_ops_project/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/Vishalshah251/Ml_ops_project/actions/workflows/ci_cd.yml)

End-to-end MLOps pipeline that classifies Twitter customer support tweets to the most likely support team. Covers data versioning, experiment tracking, automated training, REST API inference, a Streamlit UI, structured logging, and CI/CD deployment to Azure.

**Live demo:** [http://20.41.123.141](http://20.41.123.141) · **API docs:** [http://20.41.123.141/api/docs](http://20.41.123.141/api/docs)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Modules](#modules)
4. [Setup](#setup)
5. [Running Locally](#running-locally)
6. [API Reference](#api-reference)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Data Versioning with DVC](#data-versioning-with-dvc)
9. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
10. [Logging](#logging)
11. [Configuration](#configuration)
12. [Tests](#tests)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Data Flow                                   │
│                                                                     │
│  twcs.csv ──DVC──► data/loader.py ──► preprocessing/cleaner.py     │
│                                              │                      │
│                                    features/vectorizer.py           │
│                                              │                      │
│                                       model/trainer.py              │
│                                              │                      │
│                                      model/evaluator.py             │
│                                         │        │                  │
│                              artifacts/       tracking/             │
│                           (model.joblib)    (mlflow_logger.py)      │
│                                 │                                   │
│                           api/main.py  ◄──── app.py (Streamlit)    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline                               │
│                                                                     │
│  git push ──► Lint ──► Unit Tests ──► Train & Evaluate ──► Deploy  │
│                                              │                      │
│                              accuracy ≥ 0.70 gate                  │
│                                              │                      │
│                                   SSH ──► Azure VM                 │
│                               FastAPI (8000) + Streamlit (8502)    │
│                                  Nginx reverse proxy (:80)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
mlops_project/
│
├── config.py                   # Single source of truth for all hyperparameters & paths
├── train_pipeline.py           # Orchestrates the full training pipeline
├── logger.py                   # Shared structured JSON logging utility
├── app.py                      # Streamlit web UI
├── requirements.txt
│
├── data/
│   └── loader.py               # Dataset loading (raw twcs + pre-labeled format support)
│
├── preprocessing/
│   └── cleaner.py              # Text cleaning: mentions, URLs, lemmatization, stopwords
│
├── features/
│   └── vectorizer.py           # TF-IDF vectorization with train/test split
│
├── model/
│   ├── trainer.py              # LinearSVC / LogisticRegression training & artifact saving
│   └── evaluator.py            # Accuracy, F1, precision, recall + metrics.json
│
├── tracking/
│   └── mlflow_logger.py        # MLflow run management & Model Registry
│
├── api/
│   └── main.py                 # FastAPI inference server
│
├── tests/
│   └── test_pipeline.py        # Unit tests (pytest)
│
├── artifacts/                  # Generated at runtime
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metrics.json
│
├── logs/                       # Generated at runtime
│   ├── api.log                 # JSON-lines inference logs
│   └── train.log               # JSON-lines training logs
│
└── .github/workflows/
    ├── ci_cd.yml               # Main CI/CD: lint → test → train → deploy
    └── retrain.yml             # Auto-retrain on new data or config change
```

---

## Modules

### `config.py` — Central Configuration

All paths, hyperparameters, and constants live here. Change a value once and it propagates everywhere.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOP_N_COMPANIES` | `5` | Number of top companies to include |
| `MAX_FEATURES` | `15000` | TF-IDF vocabulary size |
| `MODEL_TYPE` | `linearsvc` | Classifier: `linearsvc` or `logreg` |
| `C` | `0.5` | Regularization strength |
| `MAX_ITER` | `2000` | Maximum training iterations |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

### `data/loader.py` — Data Loading

Loads the [Twitter Customer Support dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

Supports two formats automatically:
- **Raw twcs format** (7 columns): joins inbound tweets with the responding company via `response_tweet_id`
- **Pre-labeled format** (`text` + `company` columns): loads directly

Filters to the top `TOP_N_COMPANIES` by tweet volume.

---

### `preprocessing/cleaner.py` — Text Cleaning

```
raw tweet → remove @mentions & URLs → lowercase → remove non-alpha
         → tokenize → remove stopwords → lemmatize → clean_text
```

Key design choice: **@mentions and URLs are removed** to prevent company identity from leaking into training features (e.g. `@AmazonHelp` would trivially predict Amazon).

Exports two functions:
- `preprocess(df)` — processes a full DataFrame (used in training)
- `clean_text(text)` — cleans a single string (used by the API at inference time)

---

### `features/vectorizer.py` — Feature Engineering

TF-IDF vectorizer with:
- Unigrams + bigrams (`ngram_range=(1, 2)`)
- Sublinear TF scaling
- Min document frequency of 3 (filters very rare terms)
- Stratified 80/20 train/test split

---

### `model/trainer.py` — Model Training

**LinearSVC** (default) wrapped in `CalibratedClassifierCV` to produce probability estimates. Balanced class weights handle the uneven company distribution.

To switch to Logistic Regression, set `MODEL_TYPE = "logreg"` in `config.py`.

| Model | Accuracy (5 companies) | Notes |
|-------|----------------------|-------|
| LinearSVC | ~85% | Default, faster |
| LogisticRegression | ~73% | Alternative |

---

### `tracking/mlflow_logger.py` — Experiment Tracking

- Logs all hyperparameters, metrics, and the trained model to MLflow
- Registers the model under `twitter-support-classifier` in the Model Registry
- After every run, promotes the best-accuracy version to the **`champion`** alias

View the MLflow UI locally:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

### `api/main.py` — FastAPI Inference Server

Loads `model.joblib` and `vectorizer.joblib` once at startup. Applies the same text cleaning pipeline as training (via `preprocessing.cleaner.clean_text`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/model/info` | GET | Model type, classes, feature count |
| `/predict` | POST | Classify a single tweet |
| `/predict/batch` | POST | Classify up to 100 tweets |
| `/logs` | GET | Tail recent log entries |

---

### `app.py` — Streamlit UI

Four tabs:

| Tab | Description |
|-----|-------------|
| **Single Predict** | Type or select an example tweet, see company prediction + confidence bar chart |
| **Batch Predict** | Paste multiple tweets, get a table + pie chart + CSV download |
| **MLflow Runs** | View experiment history, compare accuracy across runs |
| **Logs** | Live view of `api.log` or `train.log` with color-coded log levels |

---

### `logger.py` — Structured Logging

Every module imports `get_logger(name)` which writes:
- **`logs/{name}.log`** — JSON-lines format (machine-readable, queryable)
- **stdout** — human-readable format

Example log entry:
```json
{"ts": "2026-04-21T10:23:01Z", "level": "INFO", "logger": "api", "msg": "predict | company=AppleSupport confidence=0.8200 elapsed_ms=12.3 text='my iphone crashed'"}
```

---

## Setup

### Prerequisites

- Python 3.11+
- Git
- (Optional) Azure VM + GitHub Actions secrets for deployment

### Install

```bash
git clone https://github.com/Vishalshah251/Ml_ops_project.git
cd Ml_ops_project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Dataset

Download `twcs.csv` from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it in the project root.

---

## Running Locally

### 1. Train the model

```bash
python train_pipeline.py twcs.csv
```

Outputs:
- `artifacts/model.joblib`
- `artifacts/vectorizer.joblib`
- `artifacts/metrics.json`
- `logs/train.log`

### 2. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Start the UI

```bash
streamlit run app.py
```

UI available at [http://localhost:8501](http://localhost:8501)

### 4. View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My iPhone keeps restarting after the iOS update"}'
```

Response:
```json
{
  "text": "My iPhone keeps restarting after the iOS update",
  "predicted_company": "AppleSupport",
  "confidence": 0.8731,
  "all_scores": {
    "AppleSupport": 0.8731,
    "AmazonHelp": 0.0512,
    "SpotifyCares": 0.0312,
    "Uber_Support": 0.0241,
    "comcastcares": 0.0204
  }
}
```

### `POST /predict/batch`

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["My order is late", "App crashes on startup"]}'
```

### `GET /logs?source=api&lines=50`

Returns the last 50 lines from `logs/api.log` as parsed JSON.

---

## CI/CD Pipeline

### Main Pipeline (`ci_cd.yml`)

Triggered on every push to `main` or PR to `main`.

```
push to main
     │
     ▼
  Lint (flake8)
     │
     ▼
  Unit Tests (pytest)
     │
     ▼
  Train & Evaluate
     │
  accuracy ≥ 0.70?
    YES │        NO → abort, do not deploy
     ▼
  Deploy to Azure VM
  (rsync + SSH restart)
     │
     ▼
  Smoke Tests
  (Streamlit + FastAPI health)
```

### Auto-Retrain (`retrain.yml`)

Triggered automatically when `twcs.csv.dvc` or `config.py` changes on `main`. Pulls new data, retrains, validates accuracy, and hot-swaps model artifacts on the VM without redeploying the full app.

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `VM_SSH_KEY` | Private SSH key for Azure VM |
| `DVC_REMOTE_URL` | URL of the DVC remote storage |

---

## Data Versioning with DVC

```bash
# Add dataset to DVC tracking
dvc add twcs.csv

# Push data to remote
dvc push

# Pull data from remote
dvc pull twcs.csv.dvc
```

### Rolling back to a previous data version

```bash
git checkout v1.0           # switch to old git tag
dvc pull twcs.csv.dvc       # pull matching data version
python train_pipeline.py twcs.csv
```

---

## Experiment Tracking with MLflow

Every training run logs:
- **Parameters:** model type, C, max_features, max_iter, top_n_companies
- **Metrics:** accuracy, F1, precision, recall
- **Artifacts:** model.joblib, vectorizer.joblib, metrics.json
- **Model signature:** input/output schema for deployment validation

The best model (by accuracy) is automatically promoted to the `champion` alias in the Model Registry.

```python
import mlflow
model = mlflow.sklearn.load_model("models:/twitter-support-classifier@champion")
```

---

## Logging

Logs are written to `logs/` as JSON lines and are visible in the **Logs tab** of the Streamlit UI.

```bash
# Tail training logs
tail -f logs/train.log

# Tail API logs  
tail -f logs/api.log

# Filter only errors
grep '"level": "ERROR"' logs/api.log
```

---

## Configuration

All configuration is in [`config.py`](config.py). Key settings:

```python
TOP_N_COMPANIES = 5       # increase to train on more companies
MAX_FEATURES    = 15000   # increase for richer vocabulary (slower training)
MODEL_TYPE      = "linearsvc"  # or "logreg"
C               = 0.5     # lower = stronger regularization
ACCURACY_THRESHOLD = 0.70 # minimum accuracy to allow deployment (set in ci_cd.yml)
```

---

## Tests

```bash
pytest tests/ -v --tb=short
```

| Test | What it covers |
|------|---------------|
| `test_clean_removes_mentions` | @mentions stripped from text |
| `test_clean_removes_urls` | URLs stripped from text |
| `test_clean_removes_empty_rows` | Rows that become empty after cleaning are dropped |
| `test_clean_lowercase` | Text is lowercased |
| `test_build_features_shape` | Vectorizer output dimensions are correct |
| `test_build_features_stratified` | Train/test split preserves class distribution |
| `test_metrics_json_written` | `metrics.json` is written with correct keys |
