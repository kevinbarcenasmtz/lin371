#!/usr/bin/env bash
# run_all.sh — Reproduce every reported result from a clean clone.
#
# Prerequisites:
#   pip install -r requirements.txt
#   FMP_API_KEY in .env (only needed if re-pulling FMP transcripts;
#                       Mayank's transcript drop ships in data/raw/transcripts/)
#
# Stages:
#   1. Pull Kalshi markets for all earnings-mention series
#   2. Flatten cached Kalshi JSON into outputs/results/kalshi_markets_flat.csv
#   3. Build labels.csv + train/dev/test splits
#   4. Run baseline classifiers (majority, consensus, buy-all-yes, hist_freq)
#   5. Run classical models (LogReg, Decision Tree) — full features + ablation
#   6. Fine-tune DistilBERT in the aggressive regime that actually learns
#   7. Regenerate figures from notebooks
#
# Usage: bash scripts/run_all.sh
set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "=== Stage 1: Pull Kalshi markets (all earnings-mention series) ==="
# Hits the Kalshi public API for every KXEARNINGSMENTION* / KXMENTIONEARN*
# series. Cached JSON drops into data/raw/kalshi/. Skip with --skip-kalshi
# if data/raw/kalshi/ is already populated (gitignored, but Kevin's box has it).
"$PYTHON" scripts/pull_kalshi_all.py

echo "=== Stage 2: Flatten Kalshi JSON to wide CSV ==="
"$PYTHON" scripts/flatten_kalshi.py

echo "=== Stage 3: Build labels.csv + cross-company splits ==="
# Cleans raw transcripts (data/raw/transcripts/ + data/raw/fmp/), joins with
# Kalshi markets, computes hist_rate from strictly-prior transcripts (date-
# based; see FIX_LEAKAGE.md), and writes data/processed/{labels.csv,splits/}.
"$PYTHON" scripts/build_dataset.py

echo "=== Stage 4: Baselines (majority, consensus, buy-all-yes, hist_freq) ==="
"$PYTHON" scripts/run_baselines.py

echo "=== Stage 5: Classical models — full features + implied_prob ablation ==="
"$PYTHON" scripts/run_classical.py --model all
"$PYTHON" scripts/run_classical.py --model all --drop-implied-prob

echo "=== Stage 6: DistilBERT (aggressive regime — the one that learns) ==="
# Conservative regime (lr=2e-5, epochs=3) collapses to majority class on
# this 751-row train split; the report includes it as methodology evidence
# only. Aggressive regime is the canonical reported run.
"$PYTHON" scripts/run_distilbert.py --lr 5e-5 --epochs 5

echo "=== Stage 7: Regenerate figures from notebooks ==="
# Produces outputs/results/figures/*.png used in the slides + final report.
jupyter nbconvert --to notebook --execute --inplace notebooks/02_kalshi_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_error_analysis.ipynb

echo "=== Done. Inspect outputs/results/ for tables, figures, experiments.jsonl ==="
