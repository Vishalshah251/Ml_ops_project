from pathlib import Path

# Paths
DATA_PATH = "twcs.csv"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"

# Dataset
# Twitter Customer Support dataset (twcs.csv):
#   - inbound=True rows are customer tweets (input text)
#   - label is derived by joining each inbound tweet to the company that responded
TEXT_COLUMN = "text"
LABEL_COLUMN = "company"
TOP_N_COMPANIES = 10

# Feature engineering
MAX_FEATURES = 15000
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Model — LinearSVC is faster and more accurate than LogisticRegression on sparse TF-IDF
MODEL_TYPE = "linearsvc"   # options: "linearsvc", "logreg"
C = 0.5
MAX_ITER = 2000

# MLflow
EXPERIMENT_NAME = "twitter-customer-support-classification"
