import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.cleaner import preprocess  # noqa: E402
from features.vectorizer import build_features  # noqa: E402


# ── Cleaner tests ──────────────────────────────────────────────────────────────
def test_clean_removes_mentions():
    df = pd.DataFrame({"text": ["@AppleSupport my phone is broken"], "company": ["AppleSupport"]})
    result = preprocess(df)
    assert "@" not in result["clean_text"].iloc[0]


def test_clean_removes_urls():
    df = pd.DataFrame({"text": ["check http://apple.com for help"], "company": ["AppleSupport"]})
    result = preprocess(df)
    assert "http" not in result["clean_text"].iloc[0]


def test_clean_removes_empty_rows():
    df = pd.DataFrame({"text": ["@user http://x.com", "real complaint here"], "company": ["A", "B"]})
    result = preprocess(df)
    assert len(result) == 1


def test_clean_lowercase():
    df = pd.DataFrame({"text": ["MY PHONE IS BROKEN"], "company": ["AppleSupport"]})
    result = preprocess(df)
    assert result["clean_text"].iloc[0] == result["clean_text"].iloc[0].lower()


# ── Vectorizer tests ───────────────────────────────────────────────────────────
def test_build_features_shape():
    df = pd.DataFrame({
        "clean_text": ["phone broken help", "refund my order please", "flight delayed again",
                       "music not playing", "internet down again"] * 20,
        "company": ["AppleSupport", "AmazonHelp", "Delta", "SpotifyCares", "comcastcares"] * 20,
    })
    X_train, X_test, y_train, y_test, vectorizer = build_features(df, max_features=100)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] <= 100


def test_build_features_stratified():
    df = pd.DataFrame({
        "clean_text": ["some text"] * 100,
        "company": (["A"] * 50 + ["B"] * 50),
    })
    _, _, y_train, y_test, _ = build_features(df, max_features=50)
    assert set(y_test.unique()) == {"A", "B"}


# ── Metrics file test ─────────────────────────────────────────────────────────
def test_metrics_json_written(tmp_path, monkeypatch):
    import json
    from model import evaluator
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    monkeypatch.setattr(evaluator, "METRICS_PATH", tmp_path / "metrics.json")

    df = pd.DataFrame({
        "clean_text": ["phone broken help", "refund my order please",
                       "music not playing", "internet down again"] * 30,
        "company": ["AppleSupport", "AmazonHelp", "SpotifyCares", "comcastcares"] * 30,
    })
    X_train, X_test, y_train, y_test, _ = build_features(df, max_features=50)
    model = CalibratedClassifierCV(LinearSVC(max_iter=100), cv=2)
    model.fit(X_train, y_train)

    evaluator.evaluate(model, X_test, y_test)

    assert (tmp_path / "metrics.json").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert "accuracy" in saved
    assert 0.0 <= saved["accuracy"] <= 1.0
