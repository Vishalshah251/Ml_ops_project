import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from config import MAX_FEATURES, TEST_SIZE, RANDOM_SEED, LABEL_COLUMN

log = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, max_features: int = MAX_FEATURES):
    """Fit TF-IDF on clean_text and return train/test splits + vectorizer."""
    log.info("Fitting TF-IDF vectorizer (max_features=%d)...", max_features)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),   # unigrams + bigrams improve short-text classification
        min_df=3,
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    log.info("Train: %d samples | Test: %d samples", X_train.shape[0], X_test.shape[0])
    return X_train, X_test, y_train, y_test, vectorizer
