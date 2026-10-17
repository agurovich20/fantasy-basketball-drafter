"""Data acquisition and cleaning.

Builds, from the raw ESPN box scores and basketball-reference tables:

* ``player_season``  : one row per (athlete_id, season) of actually-played
                       regular-season games, with ESPN rate stats plus the
                       matched basketball-reference advanced columns.
* ``opening_roster`` : one row per (athlete_id, season) giving the team on the
                       player's FIRST box-score row that season (DNP rows
                       included) -- the opening-night roster, leak-free.
* ``player_bio``     : one row per athlete_id with birth_date and debut season.
* ``dnp``            : the did-not-play rows (kept separately).
* ``rosters_2027``   : the 2026-27 opening rosters (ESPN).

Downloads are cached under data/raw/ and skipped if present.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
_BOX_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_nba_player_boxscores/player_box_{year}.parquet"
)
_ADV_URL = (
    "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/"
    "master/Data/Advanced.csv"
)
_CAREER_URL = (
    "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/"
    "master/Data/Player%20Career%20Info.csv"
)
_PER_GAME_URL = (
    "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/"
    "master/Data/Player%20Per%20Game.csv"
)
_ROSTERS_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_nba_rosters/rosters_2027.parquet"
)
_DRAFT_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_nba_draft/draft_2027.parquet"
)


def _download(url: str, dest) -> None:
    if dest.exists():
        return
    import requests

    print(f"  downloading {dest.name} ...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def download_all() -> None:
    """Fetch every raw file we need, skipping any already on disk."""
    for y in C.BOX_YEARS:
        _download(_BOX_URL.format(year=y), C.DATA_RAW / f"player_box_{y}.parquet")
    _download(_ADV_URL, C.DATA_RAW / "Advanced.csv")
    _download(_CAREER_URL, C.DATA_RAW / "Player_Career_Info.csv")
    _download(_PER_GAME_URL, C.DATA_RAW / "Player_Per_Game.csv")
    _download(_ROSTERS_URL, C.DATA_RAW / "rosters_2027.parquet")
    _download(_DRAFT_URL, C.DATA_RAW / "draft_2027.parquet")


# --------------------------------------------------------------------------
# Name normalization
# --------------------------------------------------------------------------
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]")
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Lower-case, strip accents, drop punctuation and Jr./Sr./II/III/IV/V."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # letters that do not NFKD-decompose to ASCII
    for k, v in {"ı": "i", "ø": "o", "đ": "d", "ð": "d", "ł": "l",
                 "ß": "ss", "æ": "ae", "œ": "oe", "þ": "th"}.items():
        s = s.replace(k, v)
    # drop periods/apostrophes entirely so "J.R." -> "jr" (matches ESPN "JR")
    s = s.replace(".", "").replace("'", "").replace("`", "").replace("-", " ")
    s = _PUNCT_RE.sub(" ", s)
    s = _SUFFIX_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------
# Raw box scores
# --------------------------------------------------------------------------
_STAT_COLS = [
    "minutes", "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls", "points",
]


@lru_cache(maxsize=1)
def load_box_raw() -> pd.DataFrame:
    """All regular-season box-score rows, 2010-2026, DNP rows included."""
    frames = []
    for y in C.BOX_YEARS:
        f = C.DATA_RAW / f"player_box_{y}.parquet"
        df = pd.read_parquet(f)
        df = df[df["season_type"] == C.REGULAR_SEASON_TYPE].copy()
        frames.append(df)
    box = pd.concat(frames, ignore_index=True)
    box = box[box["athlete_id"].notna()].copy()
    box["season"] = box["season"].astype(int)
    box["athlete_id"] = box["athlete_id"].astype("int64")
    for c in _STAT_COLS:
        box[c] = pd.to_numeric(box[c], errors="coerce")
    box["game_date"] = pd.to_datetime(box["game_date"], errors="coerce")
    box["did_not_play"] = box["did_not_play"].fillna(False).astype(bool)
    box["starter"] = box["starter"].fillna(False).astype(bool)
    box["norm_name"] = box["athlete_display_name"].map(normalize_name)
    box["fp"] = C.fantasy_points(box)
    return box


def played_rows(box: pd.DataFrame) -> pd.DataFrame:
    """Rows where the player actually appeared (not DNP, minutes recorded)."""
    m = (~box["did_not_play"]) & box["minutes"].notna()
    return box[m]


def dnp_rows(box: pd.DataFrame) -> pd.DataFrame:
    """Did-not-play rows, with their injury/other reason."""
    cols = ["season", "game_id", "game_date", "athlete_id",
            "athlete_display_name", "team_abbreviation", "reason"]
    return box.loc[box["did_not_play"], cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Player-season aggregation
# --------------------------------------------------------------------------
def _team_games(box: pd.DataFrame) -> pd.DataFrame:
    """Distinct regular-season games per (team_abbreviation, season)."""
    pr = played_rows(box)
    tg = (pr.groupby(["season", "team_abbreviation"])["game_id"]
            .nunique().rename("team_games").reset_index())
    return tg


def player_season_stats(box: pd.DataFrame) -> pd.DataFrame:
    """One row per (athlete_id, season) from played regular-season games."""
    pr = played_rows(box).copy()

    # primary team that season = team the player logged the most games for
    tm_counts = (pr.groupby(["athlete_id", "season", "team_abbreviation"])
                   .size().rename("n").reset_index())
    primary = (tm_counts.sort_values(["athlete_id", "season", "n"])
                        .groupby(["athlete_id", "season"]).tail(1)
                        [["athlete_id", "season", "team_abbreviation"]]
                        .rename(columns={"team_abbreviation": "primary_team"}))

    g = pr.groupby(["athlete_id", "season"])
    agg = g.agg(
        name=("athlete_display_name", "last"),
        norm_name=("norm_name", "last"),
        games=("game_id", "nunique"),
        starts=("starter", "sum"),
        mp=("minutes", "sum"),
        fgm=("field_goals_made", "sum"),
        fga=("field_goals_attempted", "sum"),
        fg3m=("three_point_field_goals_made", "sum"),
        fg3a=("three_point_field_goals_attempted", "sum"),
        ftm=("free_throws_made", "sum"),
        fta=("free_throws_attempted", "sum"),
        reb=("rebounds", "sum"),
        ast=("assists", "sum"),
        stl=("steals", "sum"),
        blk=("blocks", "sum"),
        tov=("turnovers", "sum"),
        pts=("points", "sum"),
        fp_total=("fp", "sum"),
        fp_std=("fp", "std"),
    ).reset_index()

    agg = agg.merge(primary, on=["athlete_id", "season"], how="left")
    tg = _team_games(box)
    agg = agg.merge(
        tg.rename(columns={"team_abbreviation": "primary_team"}),
        on=["season", "primary_team"], how="left",
    )

    # rate stats
    agg["fppg"] = agg["fp_total"] / agg["games"]
    agg["mpg"] = agg["mp"] / agg["games"]
    agg["fp_per_min"] = agg["fp_total"] / agg["mp"].replace(0, np.nan)
    agg["start_rate"] = agg["starts"] / agg["games"]
    agg["games_share"] = (agg["games"] / agg["team_games"]).clip(upper=1.0)

    # per-36
    mp = agg["mp"].replace(0, np.nan)
    for src, dst in [("pts", "pts"), ("reb", "reb"), ("ast", "ast"),
                     ("stl", "stl"), ("blk", "blk"), ("tov", "tov"),
                     ("fga", "fga"), ("fta", "fta"), ("fg3a", "fg3a")]:
        agg[f"p36_{dst}"] = agg[src] / mp * 36.0
    agg["fg_pct"] = agg["fgm"] / agg["fga"].replace(0, np.nan)
    agg["ft_pct"] = agg["ftm"] / agg["fta"].replace(0, np.nan)

    # last-25-game FPPG / MPG
    last = _last_n_game_rates(pr, C.LAST_N_GAMES)
    agg = agg.merge(last, on=["athlete_id", "season"], how="left")
    return agg


def _last_n_game_rates(pr: pd.DataFrame, n: int) -> pd.DataFrame:
    pr = pr.sort_values(["athlete_id", "season", "game_date", "game_id"])
    out = []
    for (aid, season), grp in pr.groupby(["athlete_id", "season"], sort=False):
        tail = grp.tail(n)
        ng = len(tail)
        out.append((aid, season,
                    tail["fp"].sum() / ng,
                    tail["minutes"].sum() / ng))
    return pd.DataFrame(out, columns=["athlete_id", "season",
                                      "fppg_last25", "mpg_last25"])


# --------------------------------------------------------------------------
# Opening rosters (leak-free: team identity only, DNP rows included)
# --------------------------------------------------------------------------
def opening_rosters(box: pd.DataFrame) -> pd.DataFrame:
    """Team on each player's FIRST box-score row of the season (DNP included)."""
    b = box.sort_values(["athlete_id", "season", "game_date", "game_id"])
    first = b.groupby(["athlete_id", "season"]).head(1)
    return first[["athlete_id", "season", "team_abbreviation",
                  "athlete_display_name", "norm_name"]].rename(
        columns={"team_abbreviation": "team"}).reset_index(drop=True)


# --------------------------------------------------------------------------
# Basketball-reference
# --------------------------------------------------------------------------
_ADV_KEEP = [
    "per", "ts_percent", "usg_percent", "ast_percent", "orb_percent",
    "drb_percent", "trb_percent", "stl_percent", "blk_percent",
    "tov_percent", "bpm", "age", "pos",
]


@lru_cache(maxsize=1)
def load_advanced() -> pd.DataFrame:
    """basketball-reference Advanced table, combined (NTM) rows for traded
    players, keyed by (season, norm_name)."""
    adv = pd.read_csv(C.DATA_RAW / "Advanced.csv")
    adv = adv[adv["lg"] == "NBA"].copy()
    adv["season"] = adv["season"].astype(int)
    # For a player with multiple rows in a season (traded), keep the combined
    # "2TM"/"3TM"/... row; otherwise the single team row.
    adv["is_combined"] = adv["team"].astype(str).str.match(r"^\d+TM$")
    dup_ids = adv.loc[adv.duplicated(["player_id", "season"], keep=False),
                      ["player_id", "season"]].drop_duplicates()
    dup_key = set(map(tuple, dup_ids.to_numpy()))
    keep_mask = adv.apply(
        lambda r: (r["player_id"], r["season"]) not in dup_key or r["is_combined"],
        axis=1,
    )
    adv = adv[keep_mask].copy()
    adv["norm_name"] = adv["player"].map(normalize_name)
    cols = ["season", "norm_name", "player", "player_id"] + _ADV_KEEP
    return adv[cols].reset_index(drop=True)


@lru_cache(maxsize=1)
def load_career() -> pd.DataFrame:
    """Player career info: birth_date and debut (first) season."""
    ci = pd.read_csv(C.DATA_RAW / "Player_Career_Info.csv")
    ci["birth_date"] = pd.to_datetime(ci["birth_date"], errors="coerce")
    ci = ci.rename(columns={"from": "debut_season"})
    return ci[["player_id", "player", "birth_date", "debut_season", "ht_in_in"]]


# basketball-reference per-game counting columns -> ESPN scoring keys
_PG_TO_ESPN = {
    "fg_per_game": "field_goals_made",
    "fga_per_game": "field_goals_attempted",
    "ft_per_game": "free_throws_made",
    "fta_per_game": "free_throws_attempted",
    "x3p_per_game": "three_point_field_goals_made",
    "trb_per_game": "rebounds",
    "ast_per_game": "assists",
    "stl_per_game": "steals",
    "blk_per_game": "blocks",
    "tov_per_game": "turnovers",
    "pts_per_game": "points",
}


@lru_cache(maxsize=1)
def load_per_game() -> pd.DataFrame:
    """basketball-reference Player Per Game, combined (NTM) rows kept."""
    pg = pd.read_csv(C.DATA_RAW / "Player_Per_Game.csv")
    pg = pg[pg["lg"] == "NBA"].copy()
    pg["season"] = pg["season"].astype(int)
    pg["is_combined"] = pg["team"].astype(str).str.match(r"^\d+TM$")
    dup = pg.loc[pg.duplicated(["player_id", "season"], keep=False),
                 ["player_id", "season"]].drop_duplicates()
    dup_key = set(map(tuple, dup.to_numpy()))
    keep = pg.apply(
        lambda r: (r["player_id"], r["season"]) not in dup_key or r["is_combined"],
        axis=1,
    )
    pg = pg[keep].copy()
    pg["norm_name"] = pg["player"].map(normalize_name)
    return pg.reset_index(drop=True)


def backfill_player_seasons(pid2aid: pd.Series, seasons=None) -> pd.DataFrame:
    """Build player-season rows for `seasons` from the Per Game table, mapped
    to ESPN athlete_id via pid2aid (bbref player_id -> athlete_id).

    FPPG / MPG / FP-per-min / per-36 / shooting are filled from per-game stats
    using the same league scoring; last-25-game rates and FP std are left NaN.
    Rows carry is_bref_backfill = 1.
    """
    seasons = seasons or C.BACKFILL_SEASONS
    pg = load_per_game()
    pg = pg[pg["season"].isin(seasons)].copy()
    pg["athlete_id"] = pg["player_id"].map(pid2aid)
    pg = pg[pg["athlete_id"].notna()].copy()
    pg["athlete_id"] = pg["athlete_id"].astype("int64")

    # per-game fantasy points: apply the scoring weights to per-game counts
    espn_like = pg.rename(columns=_PG_TO_ESPN)
    fppg = C.fantasy_points(espn_like)

    out = pd.DataFrame({
        "athlete_id": pg["athlete_id"].to_numpy(),
        "season": pg["season"].to_numpy(),
        "name": pg["player"].to_numpy(),
        "norm_name": pg["norm_name"].to_numpy(),
        "games": pg["g"].to_numpy(),
        "starts": pg["gs"].to_numpy(),
        "mpg": pg["mp_per_game"].to_numpy(),
        "fppg": fppg.to_numpy(),
    })
    mpg = out["mpg"].replace(0, np.nan)
    out["fp_per_min"] = out["fppg"] / mpg
    out["mp"] = out["mpg"] * out["games"]
    out["fp_total"] = out["fppg"] * out["games"]
    out["start_rate"] = out["starts"] / out["games"]
    out["games_share"] = (out["games"] / C.BACKFILL_SEASON_GAMES).clip(upper=1.0)
    # per-36 from per-game / minutes-per-game
    for pgcol, dst in [("pts_per_game", "pts"), ("trb_per_game", "reb"),
                       ("ast_per_game", "ast"), ("stl_per_game", "stl"),
                       ("blk_per_game", "blk"), ("tov_per_game", "tov"),
                       ("fga_per_game", "fga"), ("fta_per_game", "fta"),
                       ("x3pa_per_game", "fg3a")]:
        out[f"p36_{dst}"] = pg[pgcol].to_numpy() / mpg.to_numpy() * 36.0
    out["fg_pct"] = pg["fg_percent"].to_numpy()
    out["ft_pct"] = pg["ft_percent"].to_numpy()
    # not available for backfilled seasons
    out["fppg_last25"] = np.nan
    out["mpg_last25"] = np.nan
    out["fp_std"] = np.nan
    out["is_bref_backfill"] = 1
    return out


def match_advanced(ps: pd.DataFrame, adv: pd.DataFrame):
    """Left-join player-season ESPN stats to advanced on (season, norm_name).

    Returns (merged, report_dict).
    """
    adv_idx = adv.drop_duplicates(["season", "norm_name"], keep="first")
    merged = ps.merge(
        adv_idx, on=["season", "norm_name"], how="left",
        suffixes=("", "_adv"),
    )
    matched = merged["player_id"].notna()
    report = {
        "espn_player_seasons": int(len(ps)),
        "matched": int(matched.sum()),
        "match_rate": float(matched.mean()),
    }
    # unmatched players with > 500 minutes
    unmatched = merged.loc[~matched & (merged["mp"] > 500),
                           ["season", "name", "mp", "games"]]
    report["unmatched_over_500min"] = (
        unmatched.sort_values("mp", ascending=False).reset_index(drop=True)
    )
    return merged, report


# --------------------------------------------------------------------------
# Player bio (athlete_id -> birth_date, debut season)
# --------------------------------------------------------------------------
def player_bio(matched: pd.DataFrame, career: pd.DataFrame) -> pd.DataFrame:
    """Majority-vote each athlete_id to a bbref player_id, then attach
    birth_date and debut season."""
    m = matched.loc[matched["player_id"].notna(),
                    ["athlete_id", "player_id"]]
    if len(m):
        vote = (m.groupby(["athlete_id", "player_id"]).size()
                  .rename("n").reset_index()
                  .sort_values(["athlete_id", "n"]))
        best = vote.groupby("athlete_id").tail(1)[["athlete_id", "player_id"]]
    else:
        best = pd.DataFrame(columns=["athlete_id", "player_id"])
    bio = best.merge(career, on="player_id", how="left")
    return bio[["athlete_id", "player_id", "birth_date", "debut_season",
                "ht_in_in"]]


# --------------------------------------------------------------------------
# 2027 rosters
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_rosters_2027() -> pd.DataFrame:
    r = pd.read_parquet(C.DATA_RAW / "rosters_2027.parquet")
    r = r.rename(columns={
        "athlete_id": "athlete_id",
        "team_abbreviation": "team",
        "full_name": "name",
        "date_of_birth": "dob",
        "experience_years": "experience_years",
    })
    r["athlete_id"] = r["athlete_id"].astype("int64")
    r["dob"] = pd.to_datetime(r["dob"], errors="coerce", utc=True).dt.tz_localize(None)
    r["norm_name"] = r["name"].map(normalize_name)
    keep = ["athlete_id", "team", "name", "norm_name", "age", "dob",
            "experience_years", "status_name"]
    return r[keep].reset_index(drop=True)
