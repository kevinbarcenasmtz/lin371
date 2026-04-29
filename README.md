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
all five notebooks (including `seaborn`, `matplotlib`, `scipy`,
`jupyter` for figures and error analysis; `lightgbm` for Mayank's
Stage 1 model in notebook 05; `cryptography` for the Kalshi RSA-PSS
auth flow in notebook 07).

## End-to-end reproduction

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
strictly-prior transcripts (date-based; see `FIX_LEAKAGE.md`), assigns
companies to train/dev/test, and writes `data/processed/labels.csv` plus
`data/processed/splits/{train,dev,test}.csv`.

### 4. Train + evaluate

```bash
# Baselines (majority, consensus, buy-all-yes, hist_freq)
python scripts/run_baselines.py

# Classical models (LogReg + Decision Tree)
python scripts/run_classical.py --model all
python scripts/run_classical.py --model all --drop-implied-prob   # ablation

# DistilBERT — aggressive regime is the canonical reported run.
# Conservative defaults (lr=2e-5, ep=3) collapsed to majority-class on
# the Apr 21 train split; documented in §4.5.1 of the results draft.
python scripts/run_distilbert.py --lr 5e-5 --epochs 5
python scripts/run_distilbert.py --smoke-test                     # tiny subset, 1 epoch
```

Each run appends to `outputs/results/experiments.jsonl` and regenerates
`outputs/results/results_table.md`. Test set is evaluated exactly once
per run, after hyperparameter selection on dev.

The `--uncertain` flag is implemented on `run_baselines.py` and
`run_classical.py` but produces empty or near-empty test evaluations at
the current snapshot time — see `report/models_results_draft.md` §4.4.2.

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

DistilBERT was fine-tuned on the UTCS `pytorch-cuda` SSH box; setup
notes (pip install, WinSCP, tmux) are in `SESSION_HANDOFF.md`.

### 6. LightGBM + Platt-scaling pipeline (Mayank)

Mayank's pipeline runs out of `notebooks/` and produces the §5.5
numbers in the final report (Brier 0.1902, AUC 0.778, $0.754
simulated P&L across 49 bets). Two stages: LightGBM gradient boosting
on transcript-history features, then Platt scaling against live
market prices for calibration.

```bash
# 1. Pull raw FMP transcripts into notebooks/transcripts/{TICKER}_transcripts/.
#    Requires FMP_API_KEY in .env. Cell 7 pulls the canonical 100+ ticker
#    list; cells 1–5 are earlier draft cells and can be skipped.
jupyter nbconvert --to notebook --execute --inplace notebooks/04_FMP_Data_Collection.ipynb

# 2. TODO (Mayank): Stage 1 reads `model_data/features_expanded.csv` and
#    `candlestick_data/daily_features_summary.csv`. Neither file is
#    currently in the repo — the feature-engineering pipeline that
#    produces them needs to be committed before notebook 05 can be
#    re-run from a clean clone.

# 3. Train LightGBM (Stage 1) + Platt scaling (Stage 2). Writes pickled
#    models, threshold config, and test predictions to notebooks/model_output/.
jupyter nbconvert --to notebook --execute --inplace notebooks/05_LightGBM_Model_Training.ipynb

# 4. (Live only) Pull active Kalshi markets and produce trading signals.
#    Requires KALSHI_KEY_ID + kalshi_key.key (see .env.example). The §5.5
#    paper numbers come from notebook 05's held-out test split, NOT from
#    notebook 07, so this step is optional for reproducing the report.
jupyter nbconvert --to notebook --execute --inplace notebooks/07_Trading_Recommender.ipynb
```

The LightGBM pipeline pins are in `requirements.txt` (`lightgbm` for
training, `cryptography` for the Kalshi RSA-PSS auth flow in notebook 07).

## Current results (P4 complete, 2026-04-28 post `_company_label` fix)

Dev/test on the cross-company split (161 dev rows, 175 test rows). All
rows are **post-leakage-fix** (see `FIX_LEAKAGE.md`) **and post the
2026-04-28 `_company_label` fix** that recovered the proper ticker for
the seven `KXMENTIONEARN*` series (AIR, CCL, CHWY, DAL, NKE, SFD, WGO);
this changes which markets land in dev vs. test, which is why split
sizes differ from the Apr 21 numbers. See
`report/models_results_draft.md` §3.5–§3.6 for both fixes.

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
a 5-epoch run. See `report/models_results_draft.md` §4.5 for the full
methodology.

## Repo layout

```
src/             core library (clients, data pipeline, features, models, evaluation)
scripts/         CLI entry points per pipeline stage (one-command run_all.sh at the top)
data/raw/        cached API responses (gitignored; FMP content from Mayank)
data/processed/  cleaned transcripts, labels.csv, train/dev/test splits
outputs/         model checkpoints (gitignored), experiments.jsonl, results tables, figures
report/          draft Markdown sections + final_report.tex
notebooks/       exploratory analysis + figure regeneration
```

## Documents

- `IMPLEMENTATION.md` — full plan, dated 2026-04-12 (some sections — e.g.
  Mayank's n-gram track — are historical; see `SESSION_HANDOFF.md` for
  the current direction).
- `FIX_LEAKAGE.md` — diagnosis and fix for the (year, quarter)-tuple
  leakage bug caught on 2026-04-21.
- `SESSION_HANDOFF.md` — latest resume note, current as of 2026-04-21.
- `outputs/results/data_collection_audit.md` — Kalshi data collection
  rationale, expansion to 93 companies, integrity checks.
- `report/data_section_draft.md` — data section of the final report
  (git-ignored).
- `report/models_results_draft.md` — methods + results sections of the
  final report (git-ignored).
