"""Project-wide constants: seeds, tickers, and all filesystem paths."""

from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Target companies
TARGET_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "KO", "PEP"]

# Filesystem layout
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_FMP_DIR = DATA_DIR / "raw" / "fmp"
# Layout: data/transcripts/{TICKER}_transcripts/{TICKER}_{YEAR}_Q{N}.txt
EXPANDED_TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
RAW_KALSHI_DIR = DATA_DIR / "raw" / "kalshi"
RAW_NEWS_DIR = DATA_DIR / "raw" / "news"
PROCESSED_DIR = DATA_DIR / "processed"
TRANSCRIPTS_DIR = PROCESSED_DIR / "transcripts"
SPLITS_DIR = PROCESSED_DIR / "splits"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = OUTPUTS_DIR / "logs"
MODELS_DIR = OUTPUTS_DIR / "models"

# External APIs
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Data quality
MIN_TRANSCRIPT_WORDS = 1000  # transcripts shorter than this are excluded
