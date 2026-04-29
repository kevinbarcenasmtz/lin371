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
cleaned earnings transcripts across 82 tickers. Cross-company split
(70/15/15), seeded at `RANDOM_SEED = 42`.

## Setup

Requires **Python 3.10 or newer** (developed on 3.12.3 and 3.13.9).

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
cp .env.example .env             # fill in FMP_API_KEY only if re-pulling FMP
```

`requirements.txt` pins everything needed by `src/`, `scripts/`, and the
three notebooks (including `seaborn`, `matplotlib`, `scipy`, `jupyter`).

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
# Conservative defaults (lr=2e-5, ep=3) collapse to majority-class on
# this 751-row train split; documented in §4.5.1 of the results draft.
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

## Current results (P4 complete, 2026-04-21 post-leakage-fix)

Dev/test on the cross-company split (155 dev rows, 184 test rows). All
rows are **post-fix** (see `FIX_LEAKAGE.md` for the bug and
`report/models_results_draft.md` §3.5 / §4.5.5 for disclosure).

Sorted by test F1-macro descending:

| Model                                       | Dev F1-macro | Test F1-macro | Test ROI/trade |
| ------------------------------------------- | -----------: | ------------: | -------------: |
| Kalshi consensus                            |       0.9803 |    **1.0000** |       −$0.0194 |
| LogReg — full features (C=0.01)             |       0.9869 |    **1.0000** |       −$0.0194 |
| Decision Tree — full features (depth=10)    |       0.9739 |        0.9941 |       −$0.0195 |
| DistilBERT — text-only (lr=5e-5, epochs=5)  |       0.5925 |        0.5560 |       −$0.0293 |
| LogReg — no `implied_prob`                  |       0.5800 |        0.5146 |       −$0.0302 |
| Historical frequency (θ=0.25)               |       0.5806 |        0.5071 |       −$0.0303 |
| Decision Tree — no `implied_prob` (depth=5) |       0.6125 |        0.4998 |       −$0.0295 |
| Majority class                              |       0.3621 |        0.3887 |       −$0.0267 |
| Buy-all-Yes                                 |       0.3621 |        0.3887 |       −$0.0267 |

**Headline:** Kalshi's pre-call consensus is effectively correct at the
0.5 threshold, so classification accuracy saturates at ~1.0 for any
model that sees `implied_prob`. The ablation removes `implied_prob` and
shows classical models collapse to roughly the hist_freq baseline —
TF-IDF + historical prior alone is not much above majority.
DistilBERT text-only beats the classical text-only ablation by ~4 F1
points (0.556 vs LogReg-no-`implied_prob` 0.515). Surprisingly, this
edge is *larger* post-fix than pre-fix (the pre-fix 0.527 was pulled
down by a degenerate positive-collapse that the fix corrected — see
§4.5.5 for the pre/post comparison). Every model loses money on ROI
because fees eat the tiny payouts on extreme-priced markets. See
`report/models_results_draft.md` §4.3–§4.5 for the full discussion.

**Note on DistilBERT hyperparameters.** The run we report uses
`lr=5e-5, epochs=5`, which is more aggressive than standard BERT
fine-tune defaults. The standard defaults (`lr=2e-5, epochs=3`)
produced a degenerate majority-class classifier on this dataset; the
retune disambiguated "model undertrained" from "task unlearnable
text-only". Dev F1 peaked at epoch 3 (0.5925) and decayed monotonically
thereafter, so with `load_best_model_at_end=True` the reported test
metrics correspond to a 3-epoch-trained model with 5-epoch patience.
See `report/models_results_draft.md` §4.5 for the full methodology.

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
