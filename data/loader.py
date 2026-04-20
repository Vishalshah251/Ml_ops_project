import logging
import pandas as pd
from config import DATA_PATH, TEXT_COLUMN, LABEL_COLUMN, TOP_N_COMPANIES

log = logging.getLogger(__name__)


def load_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Twitter Customer Support dataset (twcs.csv).

    Strategy:
      - inbound=True rows are customer tweets (input text)
      - label = author_id of the company that replied to each tweet
      - Join inbound tweets to outbound replies via response_tweet_id
      - Restrict to top TOP_N_COMPANIES for a balanced classification task
    """
    log.info("Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath, dtype={"response_tweet_id": str, "in_response_to_tweet_id": str})

    # Build outbound lookup: tweet_id -> company author_id
    outbound = (
        df[~df["inbound"]][["tweet_id", "author_id"]]
        .rename(columns={"author_id": LABEL_COLUMN})
    )

    # Inbound customer tweets that have at least one response
    inbound = df[df["inbound"]][["tweet_id", TEXT_COLUMN, "response_tweet_id"]].dropna(
        subset=["response_tweet_id"]
    )

    # response_tweet_id may be comma-separated; take the first one
    inbound = inbound.copy()
    inbound["response_tweet_id"] = (
        inbound["response_tweet_id"].str.split(",").str[0].str.strip()
    )
    inbound["response_tweet_id"] = pd.to_numeric(inbound["response_tweet_id"], errors="coerce")
    inbound = inbound.dropna(subset=["response_tweet_id"])
    inbound["response_tweet_id"] = inbound["response_tweet_id"].astype(int)

    # Join to get label
    labeled = inbound.merge(outbound, left_on="response_tweet_id", right_on="tweet_id", how="inner")
    labeled = labeled[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

    # Restrict to top-N companies
    top_companies = labeled[LABEL_COLUMN].value_counts().head(TOP_N_COMPANIES).index
    labeled = labeled[labeled[LABEL_COLUMN].isin(top_companies)].reset_index(drop=True)

    log.info("Labeled dataset shape: %s", labeled.shape)
    log.info("Class distribution:\n%s", labeled[LABEL_COLUMN].value_counts().to_string())

    return labeled
