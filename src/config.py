"""Central configuration: paths, seeds, the league scoring rule, and the
walk-forward / holdout / production season splits.

The fantasy scoring rule lives here and NOWHERE else. Every module imports
``fantasy_points`` / ``SCORING`` from this file.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
OUTPUT = ROOT / "output"
VENDOR = ROOT / "vendor"

for _p in (DATA_RAW, DATA_INTERIM, REPORTS, OUTPUT):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 20262027

# --------------------------------------------------------------------------
# OpenMP shim.  lightgbm (and xgboost on some builds) need libgomp.so.1 which
# is not installed system-wide here; we vendor a copy and preload it with
# RTLD_GLOBAL *before* those libraries import.  Importing src.config first
# therefore makes ``python -m src.backtest`` work out of the box.
# --------------------------------------------------------------------------
def _preload_libgomp() -> None:
    import ctypes

    cand = VENDOR / "libgomp.so.1"
    if cand.exists():
        try:
            ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_libgomp()

# --------------------------------------------------------------------------
# League scoring  (14-team ESPN head-to-head POINTS)
# --------------------------------------------------------------------------
# FP = 2*FGM - 1*FGA + 1*FTM - 1*FTA + 1*3PM + 1*REB + 2*AST + 4*STL + 4*BLK
#      - 2*TO + 1*PTS
#
# Keys are the ESPN player-box-score column names.
SCORING: dict[str, float] = {
    "field_goals_made": 2.0,
    "field_goals_attempted": -1.0,
    "free_throws_made": 1.0,
    "free_throws_attempted": -1.0,
    "three_point_field_goals_made": 1.0,
    "rebounds": 1.0,
    "assists": 2.0,
    "steals": 4.0,
    "blocks": 4.0,
    "turnovers": -2.0,
    "points": 1.0,
}


def fantasy_points(df: pd.DataFrame) -> pd.Series:
    """Fantasy points for each row of an ESPN-schema box-score frame.

    Missing counting stats are treated as 0 so that a played row with a
    stray NaN still scores.  This is the ONLY definition of league scoring.
    """
    total = pd.Series(0.0, index=df.index, dtype="float64")
    for col, w in SCORING.items():
        total = total + w * pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return total


# --------------------------------------------------------------------------
# ESPN constants
# --------------------------------------------------------------------------
REGULAR_SEASON_TYPE = 2

# Box-score years we have on disk.  ESPN labels a season by the year it ends.
BOX_YEARS = list(range(2010, 2027))  # 2009-10 ... 2025-26

# Seasons backfilled from basketball-reference Per Game (box scores start 2010,
# so 2011/2012 targets would otherwise have data-edge missing lag seasons).
BACKFILL_SEASONS = [2008, 2009]
BACKFILL_SEASON_GAMES = 82  # both were full 82-game seasons (team-games denom)

# 2026-27 players ruled out for the whole season.  They stay in the projection
# pool but are removed from the opening roster used for team-context features,
# mirroring historical players who missed a whole season (zero box rows).
RULED_OUT_2027 = {
    6430: "Jimmy Butler III",
    4871141: "Ariel Hukporti",
    5105571: "Henri Veesaar",
    4606840: "Jamir Watkins",
    5157066: "Johnny Furphy",
}

# --------------------------------------------------------------------------
# Modeling-table rules
# --------------------------------------------------------------------------
MIN_GAMES = 20            # minimum games in target season t to be a train/eval row
N_LAG = 3                 # number of prior seasons used in history features
LAST_N_GAMES = 25         # "last N games" window inside a season
DRAFT_POOL_SIZE = 200     # top-N by Marcel projection for draft-pool metrics

# --------------------------------------------------------------------------
# Season splits  (target seasons; models train on target seasons <= boundary)
# --------------------------------------------------------------------------
TRAIN_START = 2011        # earliest target season used anywhere

# Walk-forward CV: (train target seasons inclusive range, validate target season)
WALK_FORWARD_FOLDS = [
    ((2011, 2014), 2015),
    ((2011, 2015), 2016),
    ((2011, 2016), 2017),
    ((2011, 2017), 2018),
    ((2011, 2018), 2019),
    ((2011, 2019), 2020),
    ((2011, 2020), 2021),
    ((2011, 2021), 2022),
    ((2011, 2022), 2023),
]

# Final holdout (do NOT touch until frozen_spec.json exists)
HOLDOUT_FOLDS = [
    ((2011, 2023), 2024),
    ((2011, 2024), 2025),
    ((2011, 2025), 2026),
]

# Production: refit on everything, project the next season
PRODUCTION_FOLD = ((2011, 2026), 2027)

VALIDATION_SEASONS = [v for (_, v) in WALK_FORWARD_FOLDS]
HOLDOUT_SEASONS = [v for (_, v) in HOLDOUT_FOLDS]

FROZEN_SPEC_PATH = REPORTS / "frozen_spec.json"
CHANGELOG_PATH = REPORTS / "changelog.md"


def train_seasons(bounds: tuple[int, int]) -> list[int]:
    """Inclusive list of target seasons used for training a fold."""
    lo, hi = bounds
    return list(range(lo, hi + 1))
