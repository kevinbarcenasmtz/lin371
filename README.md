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
`data/raw/fmp/` and `data/transcripts/`. Kevin does not run the FMP pull
script. See `scripts/pull_fmp_transcripts.py` for the reference
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

DistilBERT (`scripts/run_distilbert.py`) is planned for P4 and not yet
implemented.

## Current results (through P3 + follow-ups, 2026-04-21)

Dev/test on the cross-company split (155 dev rows, 184 test rows):

| Model                         | Dev F1-macro | Test F1-macro | Test ROI/trade |
| ----------------------------- | -----------: | ------------: | -------------: |
| Majority class                |       0.3621 |        0.3887 |       −$0.0267 |
| Buy-all-Yes                   |       0.3621 |        0.3887 |       −$0.0267 |
| Historical frequency (θ=0.25) |       0.5806 |        0.5071 |       −$0.0303 |
| Kalshi consensus              |       0.9803 |    **1.0000** |       −$0.0194 |
| LogReg (C=0.01, full features) |      0.9869 |    **1.0000** |       −$0.0194 |
| Decision Tree (d=10, full features) | 0.9739 |        0.9941 |       −$0.0195 |
| LogReg — no `implied_prob`    |       0.5800 |        0.5146 |       −$0.0302 |
| Decision Tree — no `implied_prob` | 0.6125  |        0.4998 |       −$0.0295 |
| DistilBERT                    |            – |             – |              – |

**Headline:** Kalshi's pre-call consensus is effectively correct at the
0.5 threshold, so classification accuracy saturates at ~1.0 for any
model that sees `implied_prob`. The ablation removes `implied_prob` and
shows classical models collapse to roughly the hist_freq baseline —
TF-IDF + historical prior alone is not much above majority. Every model
loses money on ROI because fees eat the tiny payouts on extreme-priced
markets. See `report/models_results_draft.md` §4.3–§4.4 for the full
discussion.

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
