"""Feature engineering: build the pooled (player, target-season) modeling table.

One row = (athlete_id, target season t).  Every feature uses only seasons
<= t-1 plus information knowable before opening night of t (team identity on
the opening roster, age, experience).  Rookies (no prior NBA season in the
last 3) are dropped.

The same machinery builds the 2027 production rows (no target columns).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C
from . import data as D

# Columns of a single player-season that get lagged into L1/L2/L3 blocks.
LAG_COLS = [
    # usage / production rates
    "fppg", "mpg", "fp_per_min", "games", "games_share", "start_rate",
    "fppg_last25", "mpg_last25",
    # per-36 box stats
    "p36_pts", "p36_reb", "p36_ast", "p36_stl", "p36_blk", "p36_tov",
    "p36_fga", "p36_fta", "p36_fg3a",
    # shooting
    "fg_pct", "ft_pct",
    # basketball-reference advanced
    "usg_percent", "ast_percent", "trb_percent", "stl_percent",
    "blk_percent", "tov_percent", "ts_percent", "bpm",
    # volatility
    "fp_std",
    # provenance (1 if this season came from the bbref 2008/2009 backfill)
    "is_bref_backfill",
]

HIST_PREFIXES = tuple(f"L{l}_" for l in range(1, C.N_LAG + 1))
CTX_COLS = ["changed_team", "net_fp_churn", "teammate_fppg_rank"]
AGE_COLS = ["age", "age_sq", "experience"]

TARGET_COLS = ["target_fppg", "target_mpg", "target_fppm", "target_games"]


class Base:
    """Container for the shared, cached source frames.

    retain_missers (Task-1 leakage probe): when True, players who posted
    1000+ FP the prior season but have zero box rows in season t are added
    back to their prior team's opening roster for t, so their production is
    NOT counted as vacated.
    """

    def __init__(self, retain_missers: bool = False) -> None:
        self.retain_missers = retain_missers
        self.box = D.load_box_raw()
        self.rosters_2027 = D.load_rosters_2027()
        self.player_season = self._build_player_season()
        self.bio = self._bio
        self.match_report = self._match_report
        # athlete_id -> height (inches) for clustering
        self._ht = self._bio.dropna(subset=["ht_in_in"]).set_index(
            "athlete_id")["ht_in_in"].to_dict()
        self.player_season["ht_in_in"] = (
            self.player_season["athlete_id"].map(self._ht))
        self.opening = self._build_opening_all()

    def _build_player_season(self) -> pd.DataFrame:
        ps = D.player_season_stats(self.box)
        ps["is_bref_backfill"] = 0
        adv = D.load_advanced()
        merged, report = D.match_advanced(ps, adv)
        self._match_report = report
        career = D.load_career()
        self._bio = D.player_bio(merged, career)

        # Backfill 2008/2009 player-seasons from bbref Per Game, mapped to ESPN
        # athlete_ids via the bio (player_id -> athlete_id) table.  Advanced
        # columns attach on (season, norm_name) just like the ESPN rows.
        pid2aid = (self._bio.dropna(subset=["player_id"])
                   .drop_duplicates("player_id")
                   .set_index("player_id")["athlete_id"])
        bf = D.backfill_player_seasons(pid2aid)
        bf = bf.merge(adv.drop(columns=["player"]).drop_duplicates(
            ["season", "norm_name"]), on=["season", "norm_name"], how="left")
        merged = pd.concat([merged, bf], ignore_index=True)
        self._n_backfill = int(len(bf))
        return merged

    def _build_opening_all(self) -> pd.DataFrame:
        """Opening rosters for every season, incl. 2027 from ESPN rosters.

        Players ruled out for all of 2026-27 are removed from the 2027 opening
        roster used for team-context features (they still get projected),
        mirroring historical whole-season absences.
        """
        opening = D.opening_rosters(self.box)[["athlete_id", "season", "team"]]
        r27 = self.rosters_2027[["athlete_id", "team"]].copy()
        r27["season"] = 2027
        r27 = r27[~r27["athlete_id"].isin(C.RULED_OUT_2027)]
        frames = [opening, r27[["athlete_id", "season", "team"]]]
        if self.retain_missers:
            frames.append(self.whole_season_missers()
                          [["athlete_id", "season", "team"]])
        return pd.concat(frames, ignore_index=True)

    def whole_season_missers(self) -> pd.DataFrame:
        """(athlete_id, season t, prior team) for players with 1000+ FP in t-1
        and zero box rows (played or DNP) in t."""
        ps = self.player_season
        present = self.box.groupby("season")["athlete_id"].agg(set).to_dict()
        prior_team = (D.opening_rosters(self.box)
                      .set_index(["athlete_id", "season"])["team"])
        rows = []
        for t in range(C.TRAIN_START, 2027):
            prev = ps[(ps["season"] == t - 1) & (ps["fp_total"] >= 1000)]
            cur = present.get(t, set())
            for aid in prev["athlete_id"]:
                if aid in cur:
                    continue
                try:
                    team = prior_team.loc[(aid, t - 1)]
                except KeyError:
                    continue
                rows.append((aid, t, team))
        return pd.DataFrame(rows, columns=["athlete_id", "season", "team"])


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------
def _attach_lags(cand: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    """Attach L1..L3 history blocks (season t-lag) with missing indicators."""
    block = ps[["athlete_id", "season"] + LAG_COLS]
    for lag in range(1, C.N_LAG + 1):
        shifted = block.copy()
        shifted["season"] = shifted["season"] + lag  # label as target season
        ren = {c: f"L{lag}_{c}" for c in LAG_COLS}
        shifted = shifted.rename(columns=ren)
        cand = cand.merge(shifted, on=["athlete_id", "season"], how="left")
        cand[f"L{lag}_missing"] = cand[f"L{lag}_fppg"].isna().astype(int)
    return cand


def _churn_and_rank(base: Base) -> pd.DataFrame:
    """Per (season, team, athlete_id): opening-roster churn and teammate rank.

    * net_fp_churn = prior-season FP vacated by departed players
                     - prior-season FP added by arriving players.
    * teammate_fppg_rank = player's rank (1 = best) by prior-season FPPG
      among the opening-roster teammates of his team this season.
    Uses only prior-season production and current roster identity -> leak-free.
    """
    ps = base.player_season
    fp_tot = ps.set_index(["athlete_id", "season"])["fp_total"]
    fppg = ps.set_index(["athlete_id", "season"])["fppg"]

    opening = base.opening
    members = {k: set(v) for k, v in
               opening.groupby(["season", "team"])["athlete_id"]}

    rows = []
    for (season, team), cur in members.items():
        prev = members.get((season - 1, team), set())
        left = prev - cur
        arrived = cur - prev
        vacated = float(sum(fp_tot.get((a, season - 1), 0.0) or 0.0 for a in left))
        added = float(sum(fp_tot.get((a, season - 1), 0.0) or 0.0 for a in arrived))
        net = vacated - added

        # rank teammates by prior-season FPPG (NaN -> treated as -inf, last)
        prio = {a: fppg.get((a, season - 1), np.nan) for a in cur}
        ordered = sorted(
            cur, key=lambda a: (prio[a] if pd.notna(prio[a]) else -np.inf),
            reverse=True,
        )
        rank = {a: i + 1 for i, a in enumerate(ordered)}
        for a in cur:
            r = rank[a] if pd.notna(prio[a]) else np.nan
            rows.append((season, a, net, r))

    return pd.DataFrame(rows, columns=["season", "athlete_id",
                                       "net_fp_churn", "teammate_fppg_rank"])


def _attach_ctx(cand: pd.DataFrame, base: Base) -> pd.DataFrame:
    # team this season (already on cand as 'team'); team last season
    opening = base.opening.copy()
    prev = opening.rename(columns={"team": "team_prev"})
    prev["season"] = prev["season"] + 1
    cand = cand.merge(prev[["athlete_id", "season", "team_prev"]],
                      on=["athlete_id", "season"], how="left")
    changed = (cand["team"] != cand["team_prev"])
    changed = changed.where(cand["team_prev"].notna(), np.nan)
    cand["changed_team"] = changed.astype("float")

    cr = _churn_and_rank(base)
    cand = cand.merge(cr, on=["season", "athlete_id"], how="left")
    return cand


def _attach_age(cand: pd.DataFrame, base: Base) -> pd.DataFrame:
    bio = base.bio.set_index("athlete_id")
    bd = cand["athlete_id"].map(bio["birth_date"])
    debut = cand["athlete_id"].map(bio["debut_season"])

    # 2027 fallback from ESPN roster dob / experience
    r27 = base.rosters_2027.set_index("athlete_id")
    is27 = cand["season"] == 2027
    bd = bd.where(bd.notna() | ~is27, cand["athlete_id"].map(r27["dob"]))

    feb1 = cand["season"].map(lambda s: pd.Timestamp(year=int(s), month=2, day=1))
    age = (feb1 - bd).dt.days / 365.25
    cand["age"] = age
    cand["age_sq"] = age ** 2
    exp = cand["season"] - debut
    # roster experience fallback for 2027
    exp27 = cand["athlete_id"].map(r27["experience_years"])
    exp = exp.where(exp.notna(), exp27)
    cand["experience"] = exp
    return cand


# --------------------------------------------------------------------------
# Public builders
# --------------------------------------------------------------------------
def _finalize(cand: pd.DataFrame) -> pd.DataFrame:
    """Drop rookies (no history in last 3 seasons)."""
    has_hist = cand[[f"L{l}_missing" for l in range(1, C.N_LAG + 1)]].sum(axis=1) \
        < C.N_LAG
    return cand[has_hist].reset_index(drop=True)


def build_modeling_table(base: Base | None = None) -> pd.DataFrame:
    """Historical pooled table, target seasons 2011-2026, >= MIN_GAMES games."""
    base = base or Base()
    ps = base.player_season
    cand = ps[(ps["games"] >= C.MIN_GAMES)
              & ps["season"].between(C.TRAIN_START, 2026)].copy()
    cand = cand[["athlete_id", "season", "name", "fppg", "mpg",
                 "fp_per_min", "games"]].rename(columns={
                     "fppg": "target_fppg", "mpg": "target_mpg",
                     "fp_per_min": "target_fppm", "games": "target_games"})
    # opening-night team for the target season
    opening = base.opening[["athlete_id", "season", "team"]]
    cand = cand.merge(opening, on=["athlete_id", "season"], how="left")

    cand = _attach_lags(cand, ps)
    cand = _attach_ctx(cand, base)
    cand = _attach_age(cand, base)
    cand = _attach_roster_change(cand, base)
    return _finalize(cand)


def _attach_roster_change(cand: pd.DataFrame, base: Base) -> pd.DataFrame:
    """Attach experimental roster-change columns and player height (cached)."""
    from . import features_ext as FX
    if getattr(base, "_roster_change", None) is None:
        base._roster_change = FX.build_roster_change(base)
    cand = cand.merge(base._roster_change, on=["athlete_id", "season"],
                      how="left")
    cand["ht_in_in"] = cand["athlete_id"].map(base._ht)
    return cand


def build_production_table(base: Base | None = None) -> pd.DataFrame:
    """2027 rows: everyone on rosters_2027 with NBA history (no target)."""
    base = base or Base()
    ps = base.player_season
    r = base.rosters_2027.copy()
    r["season"] = 2027
    cand = r[["athlete_id", "season", "name", "team"]].copy()
    for c in TARGET_COLS:
        cand[c] = np.nan

    cand = _attach_lags(cand, ps)
    cand = _attach_ctx(cand, base)
    cand = _attach_age(cand, base)
    cand = _attach_roster_change(cand, base)
    return _finalize(cand)


def feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Return feature columns for 'hist', 'hist_ctx' or 'hist_ctx_roster'."""
    from . import features_ext as FX
    hist = [c for c in df.columns
            if c.startswith(HIST_PREFIXES)] + AGE_COLS
    if feature_set == "hist":
        return hist
    if feature_set == "hist_ctx":
        return hist + CTX_COLS
    if feature_set == "hist_ctx_roster":
        return hist + CTX_COLS + [c for c in FX.ROSTER_COLS if c in df.columns]
    raise ValueError(feature_set)


# --------------------------------------------------------------------------
# Leakage test
# --------------------------------------------------------------------------
def leakage_test(base: Base | None = None, season: int = 2020,
                 seed: int = C.SEED) -> dict:
    """Perturb every season-t stat and assert no feature for target t changes.

    Features for target t may only use seasons <= t-1 and opening-roster /
    bio identity, so scrambling season-t player-season stats must leave every
    feature column of the season-t rows untouched (the target may change).
    """
    base = base or Base()
    tbl0 = build_modeling_table(base)

    # perturb season-t stats in a copy of base
    rng = np.random.default_rng(seed)
    base2 = Base.__new__(Base)
    base2.box = base.box
    ps2 = base.player_season.copy()
    numeric = [c for c in ps2.columns
               if c not in ("athlete_id", "season", "name", "norm_name",
                            "player", "player_id", "primary_team")
               and pd.api.types.is_numeric_dtype(ps2[c])]
    mask = ps2["season"] == season
    for c in numeric:
        ps2[c] = ps2[c].astype("float64")
        ps2.loc[mask, c] = rng.normal(100, 10, size=int(mask.sum()))
    base2.player_season = ps2
    base2.opening = base.opening
    base2.bio = base.bio
    base2.rosters_2027 = base.rosters_2027
    base2.match_report = base.match_report
    tbl1 = build_modeling_table(base2)

    feats = [c for c in feature_columns(tbl0, "hist_ctx")]
    a = tbl0[tbl0["season"] == season].set_index("athlete_id")
    b = tbl1[tbl1["season"] == season].set_index("athlete_id")
    # Perturbing season-t games changes which players clear the target-season
    # games filter (that is target-side, not a feature); compare the features
    # on the athletes present in both builds.
    common = a.index.intersection(b.index)
    a, b = a.loc[common].sort_index(), b.loc[common].sort_index()

    changed = []
    for c in feats:
        va, vb = a[c].to_numpy(float), b[c].to_numpy(float)
        if not np.allclose(va, vb, equal_nan=True):
            changed.append(c)

    target_changed = not np.allclose(
        a["target_fppg"].to_numpy(float), b["target_fppg"].to_numpy(float),
        equal_nan=True,
    )
    return {
        "season": season,
        "n_rows": int(len(common)),
        "n_features_checked": len(feats),
        "features_that_changed": changed,
        "passed": len(changed) == 0,
        "target_changed_as_expected": bool(target_changed),
    }
