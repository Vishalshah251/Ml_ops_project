import re
import logging
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

log = logging.getLogger(__name__)

_NLTK_RESOURCES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def _ensure_nltk_resources():
    for resource in _NLTK_RESOURCES:
        nltk.download(resource, quiet=True)


def _clean(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    # Remove Twitter @mentions and URLs — they leak company identity into features
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


_lemmatizer: WordNetLemmatizer | None = None
_stop_words: set | None = None


def _get_nltk_objects():
    global _lemmatizer, _stop_words
    if _lemmatizer is None:
        _ensure_nltk_resources()
        _lemmatizer = WordNetLemmatizer()
        _stop_words = set(stopwords.words("english"))
    return _lemmatizer, _stop_words


def clean_text(text: str) -> str:
    """Clean a single tweet string. Safe to call from the API at inference time."""
    lemmatizer, stop_words = _get_nltk_objects()
    return _clean(str(text), lemmatizer, stop_words)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full text cleaning pipeline to the 'text' column."""
    lemmatizer, stop_words = _get_nltk_objects()

    log.info("Preprocessing %d text samples...", len(df))
    df = df.copy()
    df["clean_text"] = df["text"].apply(lambda x: _clean(str(x), lemmatizer, stop_words))
    # Drop rows where cleaning produced empty strings
    before = len(df)
    df = df[df["clean_text"].str.strip().ne("")].reset_index(drop=True)
    if before - len(df):
        log.info("Dropped %d empty rows after cleaning.", before - len(df))
    log.info("Preprocessing complete.")
    return df
