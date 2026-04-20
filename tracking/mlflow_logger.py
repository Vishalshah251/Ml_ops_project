import logging
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
from config import EXPERIMENT_NAME, MODEL_PATH, VECTORIZER_PATH, TOP_N_COMPANIES, DATA_PATH

log = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "twitter-support-classifier"


def start_run(params: dict, tags: dict = None):
    """Set experiment and start an MLflow run, logging params and tags."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    run = mlflow.start_run()

    mlflow.set_tags({
        "project": "customer-support-classification",
        "dataset": DATA_PATH,
        "top_n_companies": TOP_N_COMPANIES,
        "framework": "scikit-learn",
        **(tags or {}),
    })

    mlflow.log_params(params)
    log.info("MLflow run started: %s", run.info.run_id)
    return run


def log_results(model, metrics: dict, X_sample=None, y_sample=None):
    """Log metrics, model signature, sklearn model, and local artifact files."""
    mlflow.log_metrics(metrics)

    # Infer model signature from sample data for schema validation on serving
    if X_sample is not None and y_sample is not None:
        signature = infer_signature(X_sample, model.predict(X_sample))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )
    else:
        mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    mlflow.log_artifact(str(MODEL_PATH))
    mlflow.log_artifact(str(VECTORIZER_PATH))
    log.info("Metrics and artifacts logged to MLflow.")


def promote_best_model(metric: str = "accuracy"):
    """Compare all runs and promote the best one to 'champion' alias in Model Registry."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        log.warning("Experiment not found: %s", EXPERIMENT_NAME)
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        log.warning("No runs found in experiment.")
        return

    best_run = runs[0]
    best_score = best_run.data.metrics.get(metric, 0)
    log.info("Best run: %s | %s=%.4f", best_run.info.run_id, metric, best_score)

    # Find the model version registered from this run
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    for v in versions:
        if v.run_id == best_run.info.run_id:
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", v.version)
            log.info(
                "Promoted version %s (run=%s) to alias 'champion'",
                v.version, best_run.info.run_id[:8],
            )
            return

    log.warning("No registered model version found for best run.")


def end_run():
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()
    log.info("MLflow run complete. Run ID: %s", run_id)
