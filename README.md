# Kalshi Earnings Mentions Bot

**Course:** LIN 371 — NLP, Spring 2026, UT Austin
**Team:** Kevin Barcenas (pipeline + classification) — Mayank Gulecha (FMP transcript collection + parallel classification experiments)
**Deadlines:** Presentation Apr 22 · Report May 1 (hard May 3)

## What it does

Predicts whether specific target words will be spoken on public-company
earnings calls, then simulates trading on Kalshi binary earnings-mention
prediction markets. The project is **classification-focused** — an
earlier plan to run an n-gram language modeling track in parallel was
dropped on 2026-04-21; the team agreed to go full classification.

**Target dataset:** 93 companies × 1,090 settled Kalshi markets from the
Jan–Apr 2026 earnings season (Q4 2025 / Q1 2026), paired with 7,211
cleaned earnings transcripts across 88 tickers. Cross-company split
(70/15/15), seeded at `RANDOM_SEED = 42`. Splits are **754 train /
161 dev / 175 test rows** across **65 / 13 / 15 tickers**.

## Setup

Requires **Python 3.10 or newer** (developed on 3.12.3 and 3.13.9).

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
cp .env.example .env             # fill in FMP_API_KEY only if re-pulling FMP
```

`requirements.txt` pins everything needed by `src/`, `scripts/`, and
all six notebooks (including `seaborn`, `matplotlib`, `scipy`,
`jupyter` for figures and error analysis; `lightgbm` for Mayank's
Stage 1 model in notebook 05; `cryptography` for the Kalshi RSA-PSS
auth flow in notebook 07).

## Reproducibility scope

The code in `src/`, `scripts/`, and notebooks `01–03` reproduces every
**classification result** in the final report end-to-end from a clean
clone (Tables 2 / Section 5.1–5.4 — baselines, LogReg, Decision Tree,
DistilBERT). The fastest path is `bash scripts/run_all.sh` after
`pip install -r requirements.txt`.

The **LightGBM + Platt-scaling pipeline** in notebooks `04 / 05 / 07`
(Section 5.5 of the report — Brier 0.1902, AUC 0.778, $0.754 simulated
P&L) is **not fully reproducible from a clean clone**. The notebooks
are committed for inspection, but the intermediate CSVs, model
pickles, and transcript drops they consume and produce live on
Mayank's local machine and are not in the repo. Details and exactly
which artifacts are missing are documented in [§6 below](#6-lightgbm--platt-scaling-pipeline-mayank).

## End-to-end reproduction (Kevin's pipeline)

The fastest path: run `bash scripts/run_all.sh` from a clean clone after
`pip install -r requirements.txt`. It executes every stage below.

If you want to step through manually:

### 1. Data — Kalshi markets

```bash
python scripts/pull_kalshi_all.py            # pull all earnings-mention series → data/raw/kalshi/
python scripts/flatten_kalshi.py             # → outputs/results/kalshi_markets_flat.csv
```

`pull_kalshi_all.py` hits the public Kalshi Trade API v2 (no auth) for
every `KXEARNINGSMENTION*` and `KXMENTIONEARN*` series; `flatten_kalshi.py`
walks the cached per-event JSON and writes the wide CSV that downstream
stages consume.

> **Point-in-time snapshot.** The accompanying data zip ships a frozen
> `data/raw/kalshi/` cache from **2026-04-12**, which is what the
> reported 1,090-market dataset was built from. Re-running
> `pull_kalshi_all.py` against the live API today will pull strictly
> more markets (the API keeps adding new earnings seasons), so the
> downstream row counts and reported numbers will drift. To reproduce
> the paper's numbers exactly, use the cached JSON in the zip and skip
> this stage.

### 2. Data — FMP transcripts (Mayank)

Mayank delivers earnings-call transcripts; Kevin does not run the FMP
pull script. Two drop locations are recognized automatically by
`src/data.py:_iter_raw_transcripts`:

- `data/raw/transcripts/{TICKER}_transcripts/{TICKER}_{YEAR}_Q{N}.txt` — Mayank's expanded corpus
- `data/raw/fmp/{TICKER}/...` — older single-ticker drops (KO local files, FMP JSON cache)

`scripts/pull_fmp_transcripts.py` is a reference template only.

### 3. Build labels + splits

```bash
python scripts/build_dataset.py
```

Cleans raw transcripts, computes `hist_rate` from the row's ticker's
strictly-prior transcripts (filtered by real call date, not by
`(year, quarter)` tuple — this filter is what prevents fiscal-vs-
calendar quarter leakage), assigns companies to train/dev/test, and
writes `data/processed/labels.csv` plus
`data/processed/splits/{train,dev,test}.csv`.

### 4. Train + evaluate

```bash
# Baselines (majority, consensus, buy-all-yes, hist_freq)
python scripts/run_baselines.py

# Classical models (LogReg + Decision Tree)
python scripts/run_classical.py --model all
python scripts/run_classical.py --model all --drop-implied-prob   # ablation

# DistilBERT — aggressive regime is the canonical reported run.
# Conservative defaults (lr=2e-5, ep=3) collapsed to a majority-class
# classifier on this train split; the aggressive regime is the one
# discussed in §3.2 of the final report.
python scripts/run_distilbert.py --lr 5e-5 --epochs 5
python scripts/run_distilbert.py --smoke-test                     # tiny subset, 1 epoch
```

Each run appends to `outputs/results/experiments.jsonl` and regenerates
`outputs/results/results_table.md`. Test set is evaluated exactly once
per run, after hyperparameter selection on dev.

The `--uncertain` flag is implemented on `run_baselines.py` and
`run_classical.py` for evaluation on the uncertain band only
($0.20 \le \mathtt{implied\_prob} \le 0.80$). On the current snapshot
the uncertain band is small (14 test rows), so the per-band metrics
are noisy and not reported in the final paper.

### 5. Figures

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_kalshi_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_error_analysis.ipynb
```

`02` produces the five `kalshi_*.png` dataset figures; `03` produces the
two confusion-matrix figures, the per-ticker error plot, and the
DistilBERT-prob-vs-implied scatter. Output goes to
`outputs/results/figures/`. Both notebooks must run after
`scripts/run_distilbert.py` so `outputs/models/distilbert/test_probs.npy`
exists.

DistilBERT was fine-tuned on the UTCS `pytorch-cuda` SSH box; the
fine-tune itself is launched via `scripts/run_distilbert.py` (the
script runs unmodified on any CUDA-capable machine, falls back to
CPU if `torch.cuda.is_available()` is False).

### 6. LightGBM + Platt-scaling pipeline (Mayank)

Mayank's pipeline runs out of `notebooks/` and produces the §5.5
numbers in the final report (Brier 0.1902, AUC 0.778, $0.754
simulated P&L across 49 bets). Two stages: LightGBM gradient boosting
on transcript-history features, then Platt scaling against live
market prices for calibration.

The pipeline is staged across three notebooks:

- `04_FMP_Data_Collection.ipynb` — pulls raw FMP transcripts into
  `notebooks/transcripts/{TICKER}_transcripts/`. Requires `FMP_API_KEY`
  in `.env`. Cell 7 is the canonical pull; cells 1–5 are earlier
  drafts.
- `05_LightGBM_Model_Training.ipynb` — Stage 1 (LightGBM) + Stage 2
  (Platt scaling). Reads `model_data/features_expanded.csv` and
  `candlestick_data/daily_features_summary.csv`; writes
  `notebooks/model_output/{stage1_lgb.pkl, stage2_lr.pkl, trading_config.json, test_predictions.csv}`.
- `07_Trading_Recommender.ipynb` — live-only. Hits the Kalshi
  authenticated API (requires `KALSHI_KEY_ID` + `kalshi_key.key`, see
  `.env.example`) to score active markets. Not deterministic; the
  §5.5 paper numbers come from notebook 05's held-out test split,
  not from 07.

#### Reproducibility caveat

**Notebooks 04 / 05 / 07 do not reproduce from a clean clone.** The
following artifacts are not in this repository:

| Artifact | Used by | Status |
| --- | --- | --- |
| `notebooks/transcripts/{TICKER}_transcripts/*.txt` | 04 → 05 | Generated by 04 from a paid FMP API key |
| `notebooks/model_data/features_expanded.csv` | 05 (Stage 1 input) | Built by Mayank locally; no producer script committed |
| `notebooks/candlestick_data/daily_features_summary.csv` | 05 (Stage 2 input) | Built by Mayank locally; no producer script committed |
| `notebooks/model_output/stage1_lgb.pkl` | 07 | Output of 05 |
| `notebooks/model_output/stage2_lr.pkl` | 07 | Output of 05 |
| `notebooks/model_output/trading_config.json` | 07 | Output of 05 |
| `notebooks/model_output/test_predictions.csv` | 07 | Output of 05 |
| Kalshi RSA private key (`kalshi_key.key`) | 07 | Per-user secret; not committed by design |

The notebooks themselves are committed in their last-executed state
so graders can read the cell outputs and verify the code matches the
report. Re-executing requires Mayank's local data drops plus a paid
FMP key. The classification track in §5.1–§5.4 of the report is
fully reproducible via `scripts/run_all.sh` and is not affected by
this caveat.

The LightGBM pipeline pins are in `requirements.txt` (`lightgbm` for
training, `cryptography` for the Kalshi RSA-PSS auth flow in notebook 07).

## Final results (cross-company split, post leakage + label fixes)

Dev/test on the cross-company split (161 dev rows, 175 test rows).
All rows are **post-leakage-fix** (transcript date filter that closed
a fiscal-vs-calendar quarter leak) **and post `_company_label` fix**
(recovers the proper ticker for the seven `KXMENTIONEARN*` series:
AIR, CCL, CHWY, DAL, NKE, SFD, WGO). Both fixes are described in §4
of the final report.

Sorted by test F1-macro descending:

| Model                                       | Dev F1-macro | Test F1-macro | Test ROI/trade |
| ------------------------------------------- | -----------: | ------------: | -------------: |
| Decision Tree — full features (depth=3)     |       0.9564 |    **0.9476** |   **−$0.0198** |
| Kalshi consensus                            |       0.9499 |        0.9408 |       −$0.0434 |
| LogReg — full features (C=10.0)             |       0.9437 |        0.9408 |       −$0.0434 |
| LogReg — no `implied_prob` (C=1.0)          |       0.5962 |        0.5813 |       −$0.0278 |
| Historical frequency (θ=0.15)               |       0.6347 |        0.5605 |       −$0.0410 |
| Decision Tree — no `implied_prob` (depth=5) |       0.5639 |        0.5497 |       −$0.0303 |
| DistilBERT — text-only (lr=5e-5, epochs=5)  |       0.5925† |       0.5182 |       −$0.0191 |
| Majority class                              |       0.3534 |        0.3705 |       −$0.0297 |
| Buy-all-Yes                                 |       0.3534 |        0.3705 |       −$0.0297 |

†DistilBERT dev F1 above is the Apr 21 number on the old 155-row dev split.
Dev probabilities were not re-transferred from the GPU box, so only the
test record (computed locally from `test_probs.npy`) is post-fix.

**Headline:** The Kalshi consensus is *near-perfect* (test F1 0.9408)
but no longer saturates the test set the way it did on the Apr 21 split,
because the new test composition includes a handful of mid-priced
markets where the 0.5 threshold tips wrong. The Decision Tree on full
features now **beats consensus on both test F1-macro and ROI**
(0.9476 / −$0.0198 vs 0.9408 / −$0.0434), recovering 1 fewer false
positive than consensus + scoring at the higher-value end of the
implied-probability spectrum. LogReg at C=10.0 ties consensus exactly.

The ablation (drop `implied_prob`) collapses every classical model to
the 0.55–0.58 range — confirming that `implied_prob` is doing nearly
all the work for the full-feature models. **DistilBERT text-only
underperforms the LogReg text-only ablation by ~6 F1 points** on the
new test split (0.5182 vs 0.5813). The pre-fix narrative — DistilBERT
beating LogReg text-only by ~4 F1 — does not survive the
`_company_label` fix; the rebalanced split assigns DistilBERT a harder
test set and the larger transcript coverage gives the classical
TF-IDF + `hist_rate` features a stronger floor. DistilBERT has the
**best ROI of any model** (−$0.0191) — its predictions sit near 0.5,
so it stakes less per trade and loses less when wrong on a mid-priced
market.

**Note on DistilBERT hyperparameters.** The run we report uses
`lr=5e-5, epochs=5`, which is more aggressive than standard BERT
fine-tune defaults. The standard defaults (`lr=2e-5, epochs=3`)
previously produced a degenerate majority-class classifier on the
Apr 21 split; the retune disambiguated "model undertrained" from
"task unlearnable text-only". With `load_best_model_at_end=True` the
reported test metrics correspond to the best-dev-F1 checkpoint inside
a 5-epoch run. See §3.2 of the final report for the full methodology.

## Repo layout

```
src/             core library (clients, data pipeline, features, models, evaluation)
scripts/         CLI entry points per pipeline stage (one-command run_all.sh at the top)
notebooks/       exploratory analysis + figure regeneration
data/raw/        cached API responses (gitignored; FMP content from Mayank)
data/processed/  cleaned transcripts, labels.csv, train/dev/test splits (gitignored)
outputs/         experiments.jsonl, results tables, figures (model checkpoints gitignored)
```

## Writeup

The canonical writeup is the compiled PDF of the final report,
submitted separately to Canvas. This repository ships the code and
this README so the classification numbers in the report can be
reproduced end-to-end.
