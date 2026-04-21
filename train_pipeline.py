"""
Customer Support Ticket Classification with Data Drift Detection
MLOps Pipeline: Load → Preprocess → Feature Engineering → Train → Evaluate
"""

import sys
import time

from config import DATA_PATH, MAX_FEATURES, RANDOM_SEED, MODEL_TYPE, MAX_ITER, C, TOP_N_COMPANIES
from data.loader import load_dataset
from preprocessing.cleaner import preprocess
from features.vectorizer import build_features
from model.trainer import train, save_artifacts
from model.evaluator import evaluate
from tracking.mlflow_logger import start_run, log_results, end_run, promote_best_model
from logger import get_logger

log = get_logger("train")


def run_pipeline(filepath: str = DATA_PATH, max_features: int = MAX_FEATURES):
    params = {
        "max_features": max_features,
        "random_seed": RANDOM_SEED,
        "model_type": MODEL_TYPE,
        "max_iter": MAX_ITER,
        "C": C,
        "top_n_companies": TOP_N_COMPANIES,
    }

    log.info("Pipeline started | file=%s max_features=%d", filepath, max_features)
    pipeline_start = time.time()

    start_run(params)
    try:
        t0 = time.time()
        df = load_dataset(filepath)
        log.info("Data loaded | rows=%d elapsed=%.2fs", len(df), time.time() - t0)

        t0 = time.time()
        df = preprocess(df)
        log.info("Preprocessing done | rows_after=%d elapsed=%.2fs", len(df), time.time() - t0)

        t0 = time.time()
        X_train, X_test, y_train, y_test, vectorizer = build_features(df, max_features)
        log.info(
            "Features built | train=%d test=%d features=%d elapsed=%.2fs",
            X_train.shape[0], X_test.shape[0], X_train.shape[1], time.time() - t0,
        )

        t0 = time.time()
        model = train(X_train, y_train)
        log.info("Model trained | type=%s elapsed=%.2fs", MODEL_TYPE, time.time() - t0)

        metrics = evaluate(model, X_test, y_test)
        log.info(
            "Evaluation done | accuracy=%.4f f1=%.4f precision=%.4f recall=%.4f",
            metrics["accuracy"], metrics["f1"], metrics["precision"], metrics["recall"],
        )

        save_artifacts(model, vectorizer)
        log.info("Artifacts saved")

        log_results(model, metrics, X_sample=X_test[:100], y_sample=y_test[:100])
    except Exception:
        log.exception("Pipeline failed")
        raise
    finally:
        end_run()

    promote_best_model(metric="accuracy")
    log.info("Pipeline complete | total=%.2fs", time.time() - pipeline_start)

    return model, vectorizer


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    run_pipeline(csv_path)
