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

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
cp .env.example .env             # fill in FMP_API_KEY if re-pulling FMP
```

## Data collection

### Mayank (FMP transcripts)

Mayank collects earnings call transcripts from FMP and delivers them to
`data/raw/fmp/` and `data/raw/transcripts/`. Kevin does not run the FMP
pull script. See `scripts/pull_fmp_transcripts.py` for the reference
template/interface.

### Kevin (Kalshi markets)

```bash
python scripts/pull_kalshi_markets.py --search earnings
python scripts/pull_kalshi_markets.py --ticker KO --series-ticker KXEARNINGSMENTIONKO
```

The flattened market dataset lives at `outputs/results/kalshi_markets_flat.csv`
(1,090 settled markets).

## Reproduce the current results

Data already cleaned and labeled (`data/processed/labels.csv` + splits):

```bash
# Main classification table (baselines + classical)
python scripts/run_baselines.py
python scripts/run_classical.py --model all

# Ablation: drop implied_prob from the numeric feature block
python scripts/run_classical.py --model all --drop-implied-prob
```

Each run appends to `outputs/results/experiments.jsonl` and regenerates
`outputs/results/results_table.md`.

The `--uncertain` flag is implemented on both run scripts but produces
empty or near-empty test evaluations at the current snapshot time
— see `report/models_results_draft.md` §4.4.2.

DistilBERT fine-tune (runs on the UTCS `pytorch-cuda` SSH box; see
`SESSION_HANDOFF.md` for the pip install + WinSCP steps):

```bash
# Aggressive regime — the one that actually learns (§4.5.3)
python scripts/run_distilbert.py --lr 5e-5 --epochs 5

# Conservative regime — standard BERT defaults; reported in §4.5.1
# as methodology evidence only (flatlines at majority-class baseline)
python scripts/run_distilbert.py
```

## Current results (through P4, 2026-04-21 post-leakage-fix)

Dev/test on the cross-company split (155 dev rows, 184 test rows).
All rows are **post-fix** (see `FIX_LEAKAGE.md` for the bug and
`report/models_results_draft.md` §3.5 / §4.5.5 for disclosure).

| Model                         | Dev F1-macro | Test F1-macro | Test ROI/trade |
| ----------------------------- | -----------: | ------------: | -------------: |
| Majority class                |       0.3621 |        0.3887 |       −$0.0267 |
| Buy-all-Yes                   |       0.3621 |        0.3887 |       −$0.0267 |
| Historical frequency (θ=0.25) |       0.5806 |        0.5071 |       −$0.0303 |
| Kalshi consensus              |       0.9803 |    **1.0000** |       −$0.0194 |
| LogReg (C=0.01, full features) |      0.9869 |    **1.0000** |       −$0.0194 |
| Decision Tree (d=10, full features) | 0.9739 |        0.9941 |       −$0.0195 |
| LogReg — no `implied_prob`    |       0.5800 |        0.5146 |       −$0.0302 |
| Decision Tree — no `implied_prob` (d=5) | 0.6125 |   0.4998 |       −$0.0295 |
| DistilBERT (text-only, lr=5e-5, ep=5) | 0.5925 | 0.5560 | −$0.0293 |

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
scripts/         CLI entry points per pipeline stage
data/raw/        cached API responses (gitignored; FMP content from Mayank)
data/processed/  cleaned transcripts, labels.csv, train/dev/test splits
outputs/         model checkpoints (gitignored), experiments.jsonl, results tables, figures
report/          draft Markdown sections + final_report.tex
notebooks/       exploratory analysis
```

## Documents

- `IMPLEMENTATION.md` — full plan, dated 2026-04-12 (some sections — e.g.
  Mayank's n-gram track — are historical; see `SESSION_HANDOFF.md` for
  the current direction).
- `SESSION_HANDOFF.md` — latest resume note, current as of 2026-04-21.
- `report/data_section_draft.md` — data section of the final report
  (git-ignored).
- `report/models_results_draft.md` — methods + results sections of the
  final report (git-ignored).
