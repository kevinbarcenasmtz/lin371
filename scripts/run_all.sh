#!/usr/bin/env bash
# run_all.sh — Reproduce all results from a clean clone.
#
# Prerequisites:
#   pip install -r requirements.txt
#   cp .env.example .env && fill in FMP_API_KEY
#   Drop KO .txt transcripts into data/raw/fmp/KO/
#
# Usage: bash scripts/run_all.sh
set -euo pipefail

echo "=== Stage 1: FMP transcripts (Mayank) ==="
# FMP transcript collection is Mayank's responsibility.
# He delivers files to data/raw/fmp/ before this script is run.
# See scripts/pull_fmp_transcripts.py for the reference template.
echo "  -> Assumes transcripts are already in data/raw/fmp/ (delivered by Mayank)"

echo "=== Stage 1: Pull Kalshi markets ==="
# KO series ticker confirmed from local price-history CSV.
# For AAPL/MSFT/GOOGL/META/AMZN: run --search earnings to discover, then fill in below.
python scripts/pull_kalshi_markets.py --ticker KO --series-ticker KXEARNINGSMENTIONKO
# python scripts/pull_kalshi_markets.py --ticker AAPL --series-ticker <SERIES>
# python scripts/pull_kalshi_markets.py --ticker MSFT --series-ticker <SERIES>
# python scripts/pull_kalshi_markets.py --ticker GOOGL --series-ticker <SERIES>
# python scripts/pull_kalshi_markets.py --ticker META --series-ticker <SERIES>
# python scripts/pull_kalshi_markets.py --ticker AMZN --series-ticker <SERIES>

echo "=== Stage 2: Build dataset ==="
python scripts/build_dataset.py

echo "=== Stage 3: Run baselines ==="
python scripts/run_baselines.py

echo "=== Stage 3: Run classical models ==="
python scripts/run_classical.py --model all

echo "=== Stage 4: Fine-tune DistilBERT ==="
python scripts/run_distilbert.py

echo "=== Done. Results in outputs/results/ ==="
