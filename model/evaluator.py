import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

log = logging.getLogger(__name__)


def evaluate(model, X_test, y_test) -> dict:
    """Compute and log classification metrics. Returns metrics dict."""
    y_pred = model.predict(X_test)
    labels = sorted(y_test.unique())

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    log.info("Accuracy : %.4f", acc)
    log.info("Precision: %.4f", precision)
    log.info("Recall   : %.4f", recall)
    log.info("F1-Score : %.4f", f1)
    log.info("Confusion Matrix:\n%s", np.array2string(cm))
    log.info("Classification Report:\n%s", classification_report(y_test, y_pred, zero_division=0))

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
