import logging
import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from config import RANDOM_SEED, MODEL_TYPE, MAX_ITER, C, MODEL_PATH, VECTORIZER_PATH

log = logging.getLogger(__name__)


def train(X_train, y_train):
    """Train classifier based on MODEL_TYPE in config."""
    if MODEL_TYPE == "linearsvc":
        log.info("Training LinearSVC (C=%s, max_iter=%d)...", C, MAX_ITER)
        base = LinearSVC(C=C, max_iter=MAX_ITER, random_state=RANDOM_SEED, class_weight="balanced")
        # Wrap in CalibratedClassifierCV to enable predict_proba for confidence scores
        model = CalibratedClassifierCV(base, cv=3)
    else:
        log.info("Training LogisticRegression (C=%s, max_iter=%d)...", C, MAX_ITER)
        model = LogisticRegression(
            solver="saga", C=C, max_iter=MAX_ITER,
            random_state=RANDOM_SEED, class_weight="balanced",
        )

    model.fit(X_train, y_train)
    log.info("Training complete.")
    return model


def save_artifacts(model, vectorizer):
    """Persist model and vectorizer to disk."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    log.info("Model saved to %s", MODEL_PATH)
    log.info("Vectorizer saved to %s", VECTORIZER_PATH)


def load_artifacts():
    """Load persisted model and vectorizer for inference."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    log.info("Artifacts loaded from %s", MODEL_PATH.parent)
    return model, vectorizer
