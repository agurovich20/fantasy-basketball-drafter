# Fantasy Basketball Drafter — FPPG projection model

Projection model for a 14-team ESPN head-to-head **points** league (2026-27).
Predicts next-season fantasy points per game (FPPG) with strict walk-forward
validation, a sealed holdout, and ESPN preseason projections as a benchmark.

## League scoring (single source of truth: `src/config.py`)

```
FP = 2·FGM − FGA + FTM − FTA + 3PM + REB + 2·AST + 4·STL + 4·BLK − 2·TO + PTS
```

## Layout

| module | role |
|---|---|
| `src/config.py` | paths, seed, **scoring rule**, season splits |
| `src/data.py` | download + clean ESPN box scores, basketball-reference, rosters/drafts; name matching; 2008/09 backfill |
| `src/features.py` | pooled (player, season) modeling table; hist + ctx features; leakage test |
| `src/features_ext.py` | experimental roster-change features and player-type clusters |
| `src/splits.py` | walk-forward / holdout / production splits |
| `src/models.py` | Marcel + last-season baselines, Ridge/RF/XGB/LGBM, target specs (a/b/c), metrics |
| `src/espn.py` | ESPN preseason projections benchmark (evaluation only) |
| `src/backtest.py` | ⏸1 data summary, ⏸2 walk-forward leaderboard, ⏸3 holdout |
| `src/pre_freeze.py` | pre-freeze experiments (leakage probe, roster features, clusters, selection, blend) |
| `src/project.py` | refit frozen spec on 2011-2026 and project 2027 |

## Setup

System Python here is 3.14 with no pip, so the environment is managed with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.14
uv pip install -r requirements.txt
```

**libgomp:** lightgbm needs `libgomp.so.1`, which is not installed system-wide.
Place a copy at `vendor/libgomp.so.1` (any `libgomp.so.1`, e.g. from a torch
wheel's `torch/lib/`); `src/config.py` preloads it on import.

## Run

```bash
.venv/bin/python -m src.backtest      # ⏸1 + ⏸2 (downloads data on first run)
.venv/bin/python -m src.pre_freeze    # pre-freeze experiments
# after a spec is frozen to reports/frozen_spec.json and approved:
.venv/bin/python -m src.project       # writes output/projections_2027.csv
.venv/bin/python -m src.espn refresh2027   # re-pull ESPN 2027 on draft day
```

## Status

Walk-forward leader: **xgb · spec (b) residual-over-Marcel · hist+ctx**
(WF MAE 3.965 vs Marcel 4.18). Production predictor blends `0.55·model +
0.45·ESPN`. Holdout (2024-2026) scored only once, after the spec is frozen.

Out of scope for now: availability (games-played) model, rookie projections,
draft strategy.
