"""ESPN preseason fantasy projections (evaluation-only benchmark).

Source: ESPN public fantasy API (no login):
  https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{YEAR}/
      segments/0/leaguedefaults/1?view=kona_player_info

Each player carries a `stats` list; the preseason full-season projection is the
entry with statSourceId=1 and statSplitTypeId=0.  We ignore ESPN's own applied
scoring and recompute fantasy points with OUR league formula from the projected
stat TOTALS, then FPPG = FP / projected games played.

Projections are available only for seasons >= 2018 (earlier years 404).
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

API = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/"
       "{year}/segments/0/leaguedefaults/1?view=kona_player_info")
RAW_DIR = C.DATA_RAW / "espn_proj"

# ESPN basketball stat id -> ESPN box-score (scoring) column name.
# Confirmed by matching statSourceId=0 totals to our box-score totals.
STAT_MAP = {
    "0": "points",
    "13": "field_goals_made",
    "14": "field_goals_attempted",
    "15": "free_throws_made",
    "16": "free_throws_attempted",
    "17": "three_point_field_goals_made",
    "6": "rebounds",
    "3": "assists",
    "2": "steals",
    "1": "blocks",
    "11": "turnovers",
}
GP_STAT = "42"     # projected games played
MIN_STAT = "40"    # projected total minutes

# A projection is only usable if every scoring total is present.  (In 2020 ESPN
# stored projections in a partial per-game form missing PTS/REB/AST/..., so we
# cannot score them with our formula -- that whole season is dropped.)
ESSENTIAL = ["0", "6", "3", "2", "1", "11", "13", "14", "15", "16", "17"]
MIN_USABLE_PER_SEASON = 50

# Seasons that are NOT clean preseason projections and must not be benchmarked:
#   2020 -> stored in partial per-game form (cannot be scored).
#   2023 -> a rest-of-season snapshot saved mid-season (~40-game baseline); it
#           tracks the current season, not prior (corr(proj,current) 0.96 >
#           corr(proj,prior) 0.86; Markkanen "proj" 37 vs prior 27), so using
#           it would leak half a season into the benchmark and blend.
EXCLUDED_SEASONS = {2020, 2023}

ESPN_SEASONS = list(range(2018, 2028))  # 2018 .. 2027 (2017 and earlier 404)


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def fetch_season(year: int, limit: int = 1500,
                 force: bool = False) -> dict | None:
    """Download one season (skip if cached unless force).  Returns JSON/None."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"espn_proj_{year}.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text())

    import requests
    flt = {"players": {"limit": limit, "offset": 0,
                       "sortPercOwned": {"sortAsc": False, "sortPriority": 1}}}
    headers = {"User-Agent": "Mozilla/5.0",
               "X-Fantasy-Filter": json.dumps(flt)}
    r = requests.get(API.format(year=year), headers=headers, timeout=90)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    dest.write_bytes(r.content)
    return r.json()


def fetch_all() -> list[int]:
    got = []
    for y in ESPN_SEASONS:
        d = fetch_season(y)
        if d is not None and d.get("players"):
            got.append(y)
    return got


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------
def _line(player: dict, src: int, year: int) -> dict | None:
    for s in player.get("stats", []):
        if (s.get("statSourceId") == src and s.get("statSplitTypeId") == 0
                and s.get("seasonId") == year):
            return s.get("stats") or {}
    return None


def _fp_from_totals(stats: dict) -> float:
    fp = 0.0
    for sid, col in STAT_MAP.items():
        fp += C.SCORING[col] * float(stats.get(sid, 0.0) or 0.0)
    return fp


def _load(year: int) -> list:
    dest = RAW_DIR / f"espn_proj_{year}.json"
    if not dest.exists():
        return []
    return json.loads(dest.read_text()).get("players", [])


def projections(year: int) -> pd.DataFrame:
    """Preseason projected FPPG per player for one season.

    Only players whose projected line carries every scoring total are kept, so
    FP is never undercounted from a partial stat dict.
    """
    rows = []
    for p in _load(year):
        pl = p["player"]
        st = _line(pl, 1, year)
        if not st or not all(k in st for k in ESSENTIAL):
            continue
        gp = float(st.get(GP_STAT, 0.0) or 0.0)
        if gp <= 0:
            continue
        fp = _fp_from_totals(st)
        rows.append({"athlete_id": int(p["id"]), "season": year,
                     "name": pl.get("fullName"),
                     "espn_proj_gp": gp,
                     "espn_proj_fp_total": fp,
                     "espn_proj_fppg": fp / gp})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def usable_seasons() -> tuple:
    """Clean preseason seasons with enough complete projections to benchmark."""
    out = []
    for y in ESPN_SEASONS:
        if y in EXCLUDED_SEASONS:
            continue
        if len(projections(y)) >= MIN_USABLE_PER_SEASON:
            out.append(y)
    return tuple(out)


def actuals(year: int) -> pd.DataFrame:
    """ESPN actual season totals (statSourceId=0) -- used only for validation."""
    rows = []
    for p in _load(year):
        pl = p["player"]
        st = _line(pl, 0, year)
        if not st:
            continue
        rec = {"athlete_id": int(p["id"]), "season": year,
               "name": pl.get("fullName"),
               "espn_actual_gp": float(st.get(GP_STAT, 0.0) or 0.0)}
        for sid, col in STAT_MAP.items():
            rec[f"espn_{col}"] = float(st.get(sid, 0.0) or 0.0)
        rows.append(rec)
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def all_projections() -> pd.DataFrame:
    frames = [projections(y) for y in usable_seasons()]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def refresh_2027() -> int:
    """Re-pull 2027 (ESPN keeps updating through preseason).  Run on draft day.

    Overwrites the cached 2027 JSON and returns the number of complete
    projections now available.
    """
    fetch_season(2027, force=True)
    n = len(projections(2027))
    print(f"re-pulled ESPN 2027 projections: {n} complete lines")
    return n


if __name__ == "__main__":
    import sys
    if "refresh2027" in sys.argv:
        refresh_2027()
