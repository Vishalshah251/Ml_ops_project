"""
Customer Support Ticket Classification with Data Drift Detection
MLOps Pipeline: Load → Preprocess → Feature Engineering → Train → Evaluate
"""

import logging
import sys

from config import DATA_PATH, MAX_FEATURES, RANDOM_SEED, MODEL_TYPE, MAX_ITER, C, TOP_N_COMPANIES
from data.loader import load_dataset
from preprocessing.cleaner import preprocess
from features.vectorizer import build_features
from model.trainer import train, save_artifacts
from model.evaluator import evaluate
from tracking.mlflow_logger import start_run, log_results, end_run, promote_best_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger(__name__)


def run_pipeline(filepath: str = DATA_PATH, max_features: int = MAX_FEATURES):
    """End-to-end training pipeline with MLflow experiment tracking."""
    params = {
        "max_features": max_features,
        "random_seed": RANDOM_SEED,
        "model_type": MODEL_TYPE,
        "max_iter": MAX_ITER,
        "C": C,
        "top_n_companies": TOP_N_COMPANIES,
    }

    start_run(params)
    try:
        df = load_dataset(filepath)
        df = preprocess(df)
        X_train, X_test, y_train, y_test, vectorizer = build_features(df, max_features)
        model = train(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        save_artifacts(model, vectorizer)

        # Pass sample data so MLflow can infer and store the model's input/output schema
        log_results(model, metrics, X_sample=X_test[:100], y_sample=y_test[:100])
    finally:
        end_run()

    # Promote best run to 'champion' alias in Model Registry
    promote_best_model(metric="accuracy")

    return model, vectorizer


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    run_pipeline(csv_path)
