"""Experimental pre-freeze features (evaluated on walk-forward folds only):

  * roster-change flows   -> build_roster_change(base)   [leak-free, precomputed]
  * player-type clusters  -> ClusterFeature                [fold-local]

All inputs come from opening rosters and prior-season stats; nothing reads the
target season's box score.  Incoming rookies are credited expected minutes /
usage by draft-pick range, from a prior (2010-2014) scouting table.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import data as D


# --------------------------------------------------------------------------
# Position groups
# --------------------------------------------------------------------------
def pos_group(pos) -> str:
    if not isinstance(pos, str) or not pos:
        return "UNK"
    first = pos.replace("-", " ").split()[0].upper()
    if first in ("PG", "SG", "G"):
        return "G"
    if first in ("SF", "PF", "F"):
        return "F"
    if first in ("C",):
        return "C"
    return "UNK"


# --------------------------------------------------------------------------
# Draft pick lookup (normalized name -> overall pick)  &  rookie priors
# --------------------------------------------------------------------------
def draft_pick_by_name() -> dict:
    """Earliest overall draft pick per normalized name across all drafts."""
    out = {}
    for f in sorted(glob.glob(str(C.DATA_RAW / "draft" / "draft_*.parquet"))):
        d = pd.read_parquet(f)
        if "overall_pick" not in d.columns:
            continue
        for _, r in d.iterrows():
            nm = D.normalize_name(r.get("athlete_display_name"))
            pk = r.get("overall_pick")
            if nm and pd.notna(pk):
                out[nm] = min(pk, out.get(nm, 999))
    return out


def _pick_bucket(pick: float) -> str:
    if pick is None or not np.isfinite(pick):
        return "undrafted"
    if pick <= 3:
        return "p1_3"
    if pick <= 9:
        return "p4_9"
    if pick <= 14:
        return "p10_14"
    if pick <= 30:
        return "p15_30"
    if pick <= 60:
        return "p31_60"
    return "undrafted"


def rookie_priors(ps: pd.DataFrame, picks: dict) -> dict:
    """Expected rookie minutes & usage-possessions by pick bucket.

    Built from debut (first-season) player-seasons in 2010-2014 ONLY -- before
    any validation season -- so it is a fixed, leak-free scouting prior.
    """
    debut = (ps.sort_values("season").groupby("athlete_id").head(1)
               .rename(columns={"season": "debut_season"}))
    debut = debut[debut["debut_season"].between(2010, 2014)].copy()
    debut["bucket"] = debut["norm_name"].map(
        lambda n: _pick_bucket(picks.get(n)))
    debut["usagepos"] = debut["fga"] + 0.44 * debut["fta"] + debut["tov"]
    g = debut.groupby("bucket").agg(exp_min=("mp", "mean"),
                                    exp_usagepos=("usagepos", "mean"))
    base = {"exp_min": debut["mp"].mean(),
            "exp_usagepos": debut["usagepos"].mean()}
    return {"table": g.to_dict("index"), "base": base}


# --------------------------------------------------------------------------
# Roster-change features
# --------------------------------------------------------------------------
_TC_COLS = ["tc_usg", "tc_ast", "tc_reb", "tc_blk", "rank_usg", "rank_fppg"]
_NET_COLS = ["rc_net_min", "rc_net_usagepos", "rc_net_ast", "rc_net_reb",
             "rc_net_min_pos", "rc_net_usagepos_pos", "rc_net_ast_pos",
             "rc_net_reb_pos"]
ROSTER_COLS = _NET_COLS + _TC_COLS


def build_roster_change(base) -> pd.DataFrame:
    ps = base.player_season.copy()
    ps["usagepos"] = ps["fga"] + 0.44 * ps["fta"] + ps["tov"]
    ps["pgrp"] = ps["pos"].map(pos_group) if "pos" in ps.columns else "UNK"

    # prior-season per-player quantities as fast dict lookups keyed (aid, season)
    qcols = ["mp", "usagepos", "ast", "reb", "usg_percent", "ast_percent",
             "trb_percent", "blk_percent", "fppg", "pgrp"]
    QD = {col: dict(zip(zip(ps["athlete_id"], ps["season"]), ps[col]))
          for col in qcols}

    picks = draft_pick_by_name()
    rp = rookie_priors(ps, picks)
    tg = D._team_games(base.box)  # (season, team_abbreviation) -> team_games
    tg_map = {(s, t): g for s, t, g in
              tg[["season", "team_abbreviation", "team_games"]].to_numpy()}

    opening = base.opening
    members = {k: list(v) for k, v in
               opening.groupby(["season", "team"])["athlete_id"]}
    name_by_id = (ps.sort_values("season").groupby("athlete_id")["norm_name"]
                    .last().to_dict())

    def q(aid, s, col):
        v = QD[col].get((aid, s), np.nan)
        if col == "pgrp":
            return v if isinstance(v, str) else None
        return v if (v is not None and pd.notna(v)) else np.nan

    rows = []
    for (season, team), cur in members.items():
        prev = members.get((season - 1, team), [])
        cur_set, prev_set = set(cur), set(prev)
        left = prev_set - cur_set
        arrived = cur_set - prev_set
        denom = tg_map.get((season - 1, team), 82) or 82

        # team-level net flows (per prior-season team game)
        def flow(ids_left, ids_arr, col, arr_rookie_key=None):
            vac = np.nansum([q(a, season - 1, col) for a in ids_left])
            add = 0.0
            for a in ids_arr:
                v = q(a, season - 1, col)
                if np.isnan(v):  # rookie / no prior season -> expected by pick
                    if arr_rookie_key:
                        bucket = _pick_bucket(picks.get(name_by_id.get(a)))
                        v = rp["table"].get(bucket, rp["base"]).get(
                            arr_rookie_key, 0.0)
                    else:
                        v = 0.0
                add += 0.0 if np.isnan(v) else v
            return (vac - add) / denom

        net_min = flow(left, arrived, "mp", "exp_min")
        net_usg = flow(left, arrived, "usagepos", "exp_usagepos")
        net_ast = flow(left, arrived, "ast", None)
        net_reb = flow(left, arrived, "reb", None)

        # per-player: position-restricted flows + teammate competition
        for p in cur:
            pg = q(p, season - 1, "pgrp")
            pg = pg if isinstance(pg, str) else None
            lft = [a for a in left if q(a, season - 1, "pgrp") == pg] if pg else []
            arr = [a for a in arrived if q(a, season - 1, "pgrp") == pg] if pg else []
            nmin_p = flow(lft, arr, "mp", "exp_min")
            nusg_p = flow(lft, arr, "usagepos", "exp_usagepos")
            nast_p = flow(lft, arr, "ast", None)
            nreb_p = flow(lft, arr, "reb", None)

            mates = [a for a in cur if a != p]
            mm = [(a, q(a, season - 1, "mp")) for a in mates]
            mm = [(a, m) for a, m in mm if np.isfinite(m) and m > 0]
            mm.sort(key=lambda x: x[1], reverse=True)
            top = mm[:9]
            if top:
                wts = np.array([m for _, m in top])
                def wavg(col):
                    vals = np.array([q(a, season - 1, col) for a, _ in top])
                    ok = np.isfinite(vals)
                    return (np.average(vals[ok], weights=wts[ok])
                            if ok.any() else np.nan)
                tc_usg, tc_ast = wavg("usg_percent"), wavg("ast_percent")
                tc_reb, tc_blk = wavg("trb_percent"), wavg("blk_percent")
            else:
                tc_usg = tc_ast = tc_reb = tc_blk = np.nan

            # rank on team by prior usage and prior FPPG (1 = highest)
            usg_self = q(p, season - 1, "usg_percent")
            fppg_self = q(p, season - 1, "fppg")
            usg_all = [q(a, season - 1, "usg_percent") for a in cur]
            fppg_all = [q(a, season - 1, "fppg") for a in cur]
            rank_usg = (1 + sum(1 for v in usg_all if np.isfinite(v) and v > usg_self)
                        if np.isfinite(usg_self) else np.nan)
            rank_fppg = (1 + sum(1 for v in fppg_all if np.isfinite(v) and v > fppg_self)
                         if np.isfinite(fppg_self) else np.nan)

            rows.append((p, season, net_min, net_usg, net_ast, net_reb,
                         nmin_p, nusg_p, nast_p, nreb_p,
                         tc_usg, tc_ast, tc_reb, tc_blk, rank_usg, rank_fppg))

    cols = ["athlete_id", "season"] + _NET_COLS + _TC_COLS
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------
# Player-type clusters (fold-local)
# --------------------------------------------------------------------------
STYLE_COLS = ["p36_pts", "p36_reb", "p36_ast", "p36_stl", "p36_blk",
              "p36_tov", "p36_fg3a", "p36_fta", "usg_percent", "ast_percent",
              "trb_percent", "stl_percent", "blk_percent", "ht_in_in"]


class ClusterFeature:
    """Fit k-means or GMM on 500+ minute training player-seasons (style stats),
    then assign a cluster id to any season's lag-1 style vector."""

    def __init__(self, k: int, method: str = "kmeans"):
        self.k, self.method = k, method
        self.scaler = StandardScaler()
        self.model = None
        self.medians = None

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[[c for c in STYLE_COLS]].to_numpy(float)

    def fit(self, ps_train: pd.DataFrame) -> "ClusterFeature":
        sub = ps_train[ps_train["mp"] >= 500].copy()
        X = self._X(sub)
        self.medians = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), self.medians, X)
        Xs = self.scaler.fit_transform(X)
        if self.method == "kmeans":
            self.model = KMeans(self.k, n_init=10, random_state=C.SEED).fit(Xs)
        else:
            self.model = GaussianMixture(self.k, covariance_type="diag",
                                         random_state=C.SEED).fit(Xs)
        return self

    def assign(self, style_df: pd.DataFrame) -> np.ndarray:
        X = style_df[[c for c in STYLE_COLS]].to_numpy(float)
        X = np.where(np.isnan(X), self.medians, X)
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)
