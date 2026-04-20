import logging
import pandas as pd
from config import DATA_PATH, TEXT_COLUMN, LABEL_COLUMN, TOP_N_COMPANIES

log = logging.getLogger(__name__)


def load_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Twitter Customer Support dataset.

    Supports two formats:
      - Raw twcs.csv (7 columns): joins inbound tweets with responding company
      - Pre-labeled CSV (2 columns: text, company): loads directly
    """
    log.info("Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath, dtype={"response_tweet_id": str})

    # ── Pre-labeled format (text + company columns already present) ──────────
    if TEXT_COLUMN in df.columns and LABEL_COLUMN in df.columns:
        labeled = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna().reset_index(drop=True)
        log.info("Pre-labeled dataset detected.")

    # ── Raw twcs format — join inbound tweets with responding company ─────────
    else:
        if "inbound" not in df.columns:
            raise ValueError("Unrecognized CSV format. Expected 'text'+'company' or raw twcs columns.")

        outbound = df[~df["inbound"]][["tweet_id", "author_id"]].rename(columns={"author_id": LABEL_COLUMN})
        inbound = df[df["inbound"]][["tweet_id", TEXT_COLUMN, "response_tweet_id"]].dropna(
            subset=["response_tweet_id"]
        ).copy()

        inbound["response_tweet_id"] = (
            inbound["response_tweet_id"].str.split(",").str[0].str.strip()
        )
        inbound["response_tweet_id"] = pd.to_numeric(inbound["response_tweet_id"], errors="coerce")
        inbound = inbound.dropna(subset=["response_tweet_id"])
        inbound["response_tweet_id"] = inbound["response_tweet_id"].astype(int)

        merged = inbound.merge(outbound, left_on="response_tweet_id", right_on="tweet_id", how="inner")
        labeled = merged[[TEXT_COLUMN, LABEL_COLUMN]].dropna().reset_index(drop=True)
        log.info("Raw twcs format detected — joined inbound tweets with company responses.")

    # ── Restrict to top-N companies ───────────────────────────────────────────
    top_companies = labeled[LABEL_COLUMN].value_counts().head(TOP_N_COMPANIES).index
    labeled = labeled[labeled[LABEL_COLUMN].isin(top_companies)].reset_index(drop=True)

    log.info("Dataset shape: %s", labeled.shape)
    log.info("Class distribution:\n%s", labeled[LABEL_COLUMN].value_counts().to_string())

    return labeled
