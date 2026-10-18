"""Games-played (availability) model.

Target: share of team games played in season t = games_played / team_games.
Players with box rows in t-1 and t+1 but none in t get target 0 (kept, not
dropped).  Features use seasons <= t-1 only.  Same walk-forward / holdout folds
as the FPPG model.  Time-boxed: if no candidate beats ESPN projected games on
the walk-forward folds, keep ESPN.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config as C      # preloads libgomp for lightgbm
from . import data as D
from . import espn as E
import lightgbm as lgb

# --------------------------------------------------------------------------
# Injury-category mapping (checked in order; first match wins)
# --------------------------------------------------------------------------
INJURY_MAP = [
    ("rest_load",       ["LOAD MANAGEMENT", "MAINTENANCE", r"\bREST\b", "REST "]),
    ("soft_tissue_leg", ["CALF", "HAMSTRING", "ACHILLES", "GROIN", "QUAD",
                         "THIGH", "ADDUCTOR", "HIP FLEXOR", "LEG"]),
    ("knee",            ["KNEE", "PATELLA", "MENISCUS", "ACL", "MCL", "PCL"]),
    ("ankle_foot",      ["ANKLE", "FOOT", "HEEL", "TOE", "PLANTAR",
                         "METATARSAL", "ARCH"]),
    ("back_hip",        ["BACK", "HIP", "SPINE", "LUMBAR", "SACR", "SCIATIC",
                         "DISC"]),
    ("other_injury",    ["SHOULDER", "WRIST", "HAND", "FINGER", "THUMB",
                         "CONCUSSION", "ILLNESS", "ABDOMIN", "OBLIQUE", "ELBOW",
                         "NECK", "RIB", "EYE", "FACIAL", "NOSE", "JAW",
                         "DENTAL", "FLU", "FEVER", "MIGRAINE", "SURGERY",
                         "STRAIN", "SPRAIN", "SORE", "SORENESS", "INJURY",
                         "CONTUSION", "FRACTURE", "TENDON", "HERNIA",
                         "BRUISE", "LACERATION", "SPASM", "VIRUS", "INFECTION",
                         "RESPIRATORY", "NERVE"]),
]
# ignored (not injury): COACH'S DECISION, PERSONAL, SUSPEND, NOT WITH TEAM,
# HEALTH & SAFETY / PROTOCOL (COVID), G LEAGUE, TWO-WAY, TRADE, INACTIVE.
INJ_CATS = [c for c, _ in INJURY_MAP]


def categorize(reason) -> str | None:
    if not isinstance(reason, str):
        return None
    r = reason.upper()
    if "COACH" in r:
        return None
    for cat, kws in INJURY_MAP:
        for kw in kws:
            if kw.startswith(r"\b"):
                if re.search(kw, r):
                    return cat
            elif kw in r:
                return cat
    return None


def mapping_table() -> pd.DataFrame:
    rows = []
    for cat, kws in INJURY_MAP:
        rows.append({"category": cat,
                     "keywords": ", ".join(k.replace(r"\b", "").strip()
                                           for k in kws)})
    rows.append({"category": "(ignored: non-injury)",
                 "keywords": "COACH'S DECISION, PERSONAL, SUSPENDED, NOT WITH "
                             "TEAM, HEALTH & SAFETY PROTOCOLS, G LEAGUE, "
                             "TWO-WAY, TRADE, INACTIVE"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Feature / target table
# --------------------------------------------------------------------------
GFEATURES = [
    "gshare_l1", "gshare_l2", "gshare_l3", "consec70",
    "inj_soft_tissue_leg", "inj_knee", "inj_ankle_foot", "inj_back_hip",
    "inj_other_injury", "inj_rest_load",
    "age", "ht_in_in", "wt", "experience",
    "l1_mpg", "l1_usg", "l1_start_rate", "changed_team",
]


def _injury_counts_by_season(box) -> dict:
    """{(athlete_id, season, category): games_missed}."""
    dnp = D.dnp_rows(box).copy()
    dnp["cat"] = dnp["reason"].map(categorize)
    dnp = dnp[dnp["cat"].notna()]
    g = dnp.groupby(["athlete_id", "season", "cat"]).size()
    return g.to_dict()


def build_games_table(base, production_season=None) -> pd.DataFrame:
    ps = base.player_season
    box = base.box
    opening = base.opening

    gshare = ps.set_index(["athlete_id", "season"])["games_share"].to_dict()
    games = ps.set_index(["athlete_id", "season"])["games"].to_dict()
    mpg = ps.set_index(["athlete_id", "season"])["mpg"].to_dict()
    usg = ps.set_index(["athlete_id", "season"])["usg_percent"].to_dict()
    startr = ps.set_index(["athlete_id", "season"])["start_rate"].to_dict()
    tgames = {(s, t): g for s, t, g in
              D._team_games(box)[["season", "team_abbreviation", "team_games"]]
              .to_numpy()}
    primary = ps.set_index(["athlete_id", "season"])["primary_team"].to_dict()
    injc = _injury_counts_by_season(box)
    box_present = {s: set(v) for s, v in
                   box.groupby("season")["athlete_id"].agg(set).items()}
    bio = base.bio.set_index("athlete_id")
    ht = base._ht
    wt = bio["wt"].to_dict()
    debut = bio["debut_season"].to_dict()
    bd = bio["birth_date"].to_dict()
    name = (ps.sort_values("season").groupby("athlete_id")["name"].last().to_dict())
    open_team = {(a, s): t for a, s, t in
                 opening[["athlete_id", "season", "team"]].to_numpy()}

    def consec70(aid, t):
        c = 0
        for s in range(t - 1, t - 12, -1):
            if games.get((aid, s), 0) >= 70:
                c += 1
            else:
                break
        return c

    def row(aid, season, team_games_t, actual_games, target, is_bridge):
        feat = {
            "athlete_id": aid, "season": season, "name": name.get(aid),
            "team_games_t": team_games_t, "actual_games": actual_games,
            "target_share": target, "is_bridge": int(is_bridge),
            "gshare_l1": gshare.get((aid, season - 1), np.nan),
            "gshare_l2": gshare.get((aid, season - 2), np.nan),
            "gshare_l3": gshare.get((aid, season - 3), np.nan),
            "consec70": consec70(aid, season),
            "l1_mpg": mpg.get((aid, season - 1), np.nan),
            "l1_usg": usg.get((aid, season - 1), np.nan),
            "l1_start_rate": startr.get((aid, season - 1), np.nan),
            "ht_in_in": ht.get(aid, np.nan), "wt": wt.get(aid, np.nan),
        }
        for cat in INJ_CATS:
            feat[f"inj_{cat}"] = sum(injc.get((aid, season - l, cat), 0)
                                     for l in (1, 2, 3))
        b = bd.get(aid)
        feat["age"] = ((pd.Timestamp(year=season, month=2, day=1) - b).days / 365.25
                       if pd.notna(b) else np.nan)
        feat["experience"] = (season - debut.get(aid)
                              if pd.notna(debut.get(aid)) else np.nan)
        tt = open_team.get((aid, season))
        tp = open_team.get((aid, season - 1))
        feat["changed_team"] = (float(tt != tp) if (tt is not None
                                and tp is not None) else np.nan)
        return feat

    rows = []
    seasons = ([production_season] if production_season
               else range(C.TRAIN_START, 2027))
    for t in seasons:
        if production_season:
            ids = base.rosters_2027["athlete_id"].tolist()
        else:
            played = box_present.get(t, set())
            bridge = (box_present.get(t - 1, set())
                      & box_present.get(t + 1, set())) - played
            ids = played | bridge
        for aid in ids:
            # need >=1 of last 3 seasons on record (non-rookie)
            if not any((aid, t - l) in games for l in (1, 2, 3)):
                continue
            if production_season:
                tg = C.BACKFILL_SEASON_GAMES
                rows.append(row(aid, t, tg, np.nan, np.nan, False))
            elif (aid, t) in games:              # played in t
                g = games[(aid, t)]
                tg = tgames.get((t, primary.get((aid, t))), 82) or 82
                rows.append(row(aid, t, tg, g, g / tg, False))
            else:                                # bridged -> target 0
                tg = 82
                rows.append(row(aid, t, tg, 0.0, 0.0, True))
    df = pd.DataFrame(rows)
    df["team"] = df.apply(lambda r: open_team.get((r["athlete_id"], r["season"]),
                                                  ""), axis=1)
    return df


# --------------------------------------------------------------------------
# Candidate predictors (fold-local)
# --------------------------------------------------------------------------
CLEAN_ESPN = set(E.ESPN_SEASONS) - E.EXCLUDED_SEASONS      # 2018,19,21,22,24,25,26


def _xy(train, evl):
    imp = SimpleImputer(strategy="median").fit(train[GFEATURES])
    sc = StandardScaler().fit(imp.transform(train[GFEATURES]))
    Xtr = sc.transform(imp.transform(train[GFEATURES]))
    Xev = sc.transform(imp.transform(evl[GFEATURES]))
    return Xtr, Xev


def pred_baseline(train, evl):
    g = evl[["gshare_l1", "gshare_l2", "gshare_l3"]].to_numpy(float)
    w = np.where(np.isnan(g), 0.0, g)          # games-weight ~ share
    num = np.nansum(np.where(np.isnan(g), 0.0, g) * w, axis=1)
    den = w.sum(axis=1)
    mu = train["target_share"].mean()
    out = np.where(den > 0, num / np.where(den > 0, den, 1), mu)
    return {"share": np.clip(out, 0, 1)}


def pred_kmeans(train, evl, k):
    Xtr, Xev = _xy(train, evl)
    km = KMeans(k, n_init=10, random_state=C.SEED).fit(Xtr)
    lab = km.labels_
    y = train["target_share"].to_numpy(float)
    stats_ = {}
    for c in range(k):
        yc = y[lab == c]
        if len(yc):
            stats_[c] = (yc.mean(), np.quantile(yc, .1), np.quantile(yc, .9))
    ev = km.predict(Xev)
    m = np.array([stats_.get(c, (y.mean(), np.quantile(y, .1), np.quantile(y, .9)))
                  for c in ev])
    return {"share": np.clip(m[:, 0], 0, 1),
            "p10": np.clip(m[:, 1], 0, 1), "p90": np.clip(m[:, 2], 0, 1)}


def pred_knn(train, evl, k, want_comps=False):
    Xtr, Xev = _xy(train, evl)
    nn = NearestNeighbors(n_neighbors=min(k, len(train))).fit(Xtr)
    _, idx = nn.kneighbors(Xev)
    y = train["target_share"].to_numpy(float)
    names = train["name"].to_numpy()
    seas = train["season"].to_numpy()
    med = np.median(y[idx], axis=1)
    p10 = np.quantile(y[idx], .1, axis=1)
    p90 = np.quantile(y[idx], .9, axis=1)
    out = {"share": np.clip(med, 0, 1), "p10": np.clip(p10, 0, 1),
           "p90": np.clip(p90, 0, 1)}
    if want_comps:
        out["comps"] = [[f"{names[j]} {int(seas[j])}" for j in idx[i][:3]]
                        for i in range(len(idx))]
    return out


def pred_lgbm(train, evl):
    Xtr, Xev = _xy(train, evl)
    y = train["target_share"].to_numpy(float)
    preds = {}
    for a, key in [(0.1, "p10"), (0.5, "share"), (0.9, "p90")]:
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300,
                              learning_rate=0.03, num_leaves=15,
                              min_child_samples=30, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8,
                              random_state=C.SEED, verbose=-1)
        m.fit(Xtr, y)
        preds[key] = np.clip(m.predict(Xev), 0, 1)
    return preds


def espn_share(evl, season):
    """ESPN projected games -> share; None if not a clean ESPN season."""
    if season not in CLEAN_ESPN:
        return None
    gp = E.projections(season).set_index("athlete_id")["espn_proj_gp"].to_dict()
    share = evl["athlete_id"].map(gp).to_numpy(float) / evl["team_games_t"].to_numpy(float)
    return np.clip(share, 0, 1)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _metrics(evl, share, p10=None, p90=None):
    tg = evl["team_games_t"].to_numpy(float)
    actual = evl["actual_games"].to_numpy(float)
    pg = np.asarray(share) * tg
    mask = ~np.isnan(pg) & ~np.isnan(actual)
    mae = float(np.mean(np.abs(pg[mask] - actual[mask])))
    # spearman within top 200 by prior minutes
    rel = evl["l1_mpg"].fillna(0).to_numpy()
    top = np.argsort(-rel)[:200]
    tm = top[~np.isnan(pg[top]) & ~np.isnan(actual[top])]
    rho = float(spearmanr(pg[tm], actual[tm]).statistic) if len(tm) > 5 else np.nan
    out = {"mae": mae, "rho200": rho}
    if p10 is not None:
        lo, hi = np.asarray(p10) * tg, np.asarray(p90) * tg
        out["coverage"] = float(np.mean((actual[mask] >= lo[mask]) &
                                         (actual[mask] <= hi[mask])))
        pb = 0.0
        for q, tau in [(lo, .1), (pg, .5), (hi, .9)]:
            d = actual[mask] - q[mask]
            pb += np.mean(np.maximum(tau * d, (tau - 1) * d))
        out["pinball"] = float(pb / 3)
    return out


# --------------------------------------------------------------------------
# Candidate registry
# --------------------------------------------------------------------------
def _candidates(train, evl, season):
    c = {}
    c["baseline_3yr"] = pred_baseline(train, evl)
    for k in (6, 10, 15):
        c[f"kmeans_{k}"] = pred_kmeans(train, evl, k)
    for k in (25, 50):
        c[f"knn_{k}"] = pred_knn(train, evl, k)
    c["lgbm_quantile"] = pred_lgbm(train, evl)
    es = espn_share(evl, season)
    if es is not None:
        c["espn"] = {"share": es}
    return c


def _rows_for(tbl, lo, hi):
    return tbl[tbl["season"].between(lo, hi)]


def run_wf(tbl):
    print("\n" + "=" * 78 + "\nWALK-FORWARD: games-played candidates\n" + "=" * 78)
    agg = {}
    for (lo, hi), ev in C.WALK_FORWARD_FOLDS:
        train = _rows_for(tbl, lo, hi)
        evl = tbl[tbl["season"] == ev]
        for name, p in _candidates(train, evl, ev).items():
            m = _metrics(evl, p["share"], p.get("p10"), p.get("p90"))
            agg.setdefault(name, []).append({"season": ev, **m})
    rows = []
    for name, lst in agg.items():
        d = pd.DataFrame(lst)
        n_all = len(d)
        rec = {"candidate": name, "folds": n_all,
               "MAE_games": round(d["mae"].mean(), 2),
               "rho200": round(d["rho200"].mean(), 3)}
        if "coverage" in d:
            rec["coverage80"] = round(d["coverage"].mean(), 3)
            rec["pinball"] = round(d["pinball"].mean(), 2)
        # MAE on the clean ESPN WF seasons only (for the ESPN comparison)
        clean = d[d["season"].isin(CLEAN_ESPN)]
        rec["MAE_cleanWF"] = round(clean["mae"].mean(), 2) if len(clean) else np.nan
        rows.append(rec)
    lb = pd.DataFrame(rows).sort_values("MAE_games")
    print(lb.to_string(index=False))
    return lb


def decide(lb):
    espn_clean = lb.loc[lb["candidate"] == "espn", "MAE_cleanWF"]
    espn_mae = float(espn_clean.iloc[0]) if len(espn_clean) else np.inf
    models = lb[lb["candidate"] != "espn"].dropna(subset=["MAE_cleanWF"])
    best = models.sort_values("MAE_cleanWF").iloc[0]
    print(f"\n[Decision] best model on clean WF seasons = {best['candidate']} "
          f"(MAE {best['MAE_cleanWF']}) vs ESPN (MAE {espn_mae})")
    if best["MAE_cleanWF"] < espn_mae:
        print(f"  -> {best['candidate']} BEATS ESPN; use it (+ optional blend).")
        return best["candidate"], espn_mae
    print("  -> NO candidate beats ESPN on the clean walk-forward seasons; "
          "KEEP ESPN's projected games.")
    return None, espn_mae


# --------------------------------------------------------------------------
# Holdout (report once) + 2027 output
# --------------------------------------------------------------------------
def holdout_report(tbl):
    print("\n" + "=" * 78 + "\nHOLDOUT (2024-2026): kept ESPN vs best model (report once)\n"
          + "=" * 78)
    rows = []
    for (lo, hi), ev in C.HOLDOUT_FOLDS:
        train = _rows_for(tbl, lo, hi)
        evl = tbl[tbl["season"] == ev]
        lg = pred_lgbm(train, evl)
        es = espn_share(evl, ev)
        rows.append({"season": ev,
                     "ESPN_MAE": round(_metrics(evl, es)["mae"], 2),
                     "lgbm_MAE": round(_metrics(evl, lg["share"])["mae"], 2)})
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    print(f"  mean: ESPN {d['ESPN_MAE'].mean():.2f}  vs  lgbm {d['lgbm_MAE'].mean():.2f}")
    return d


# manual return-date overrides (approx games played, ~82-game season)
MANUAL_GAMES = {
    "jimmy butler": 32,      # ~Jan/Feb return
    "shaedon sharpe": 22,    # ~late Feb
    "mark williams": 28,     # ~Feb
    # Brandon Ingram: TBD -> left to ESPN
}


def project_2027_games(base, tbl):
    """proj_games (ESPN kept + manual overrides), games_p10/p90 and top-3 comps."""
    train = tbl                                   # all 2011-2026 rows
    prod = build_games_table(base, production_season=2027)
    knn = pred_knn(train, prod, 50, want_comps=True)
    espn_gp = E.projections(2027).set_index("athlete_id")["espn_proj_gp"].to_dict()

    proj = prod["athlete_id"].map(espn_gp).to_numpy(float)
    knn_games = knn["share"] * 82.0
    proj = np.where(np.isnan(proj), knn_games, proj)   # fallback: KNN comps
    low = (knn["share"] - knn["p10"]) * 82.0
    high = (knn["p90"] - knn["share"]) * 82.0

    names_norm = prod["name"].map(D.normalize_name)
    for i, nm in enumerate(names_norm):
        if nm in MANUAL_GAMES:
            proj[i] = MANUAL_GAMES[nm]

    out = pd.DataFrame({
        "athlete_id": prod["athlete_id"].to_numpy(),
        "proj_games": np.clip(np.round(proj, 0), 0, 82),
        "games_p10": np.clip(np.round(proj - low, 0), 0, 82),
        "games_p90": np.clip(np.round(proj + high, 0), 0, 82),
        "comps": ["; ".join(c) for c in knn["comps"]],
    })
    return out


def main():
    from . import features as F
    print("[Injury-category mapping]")
    print(mapping_table().to_string(index=False))
    base = F.Base()
    tbl = build_games_table(base)
    lb = run_wf(tbl)
    lb.to_csv(C.REPORTS / "games_wf_leaderboard.csv", index=False)
    winner, espn_mae = decide(lb)
    holdout_report(tbl)
    out = project_2027_games(base, tbl)
    out.to_csv(C.REPORTS / "games_proj_2027.csv", index=False)
    print(f"\n[2027 games] wrote reports/games_proj_2027.csv ({len(out)} players)")
    print("  manual overrides:", MANUAL_GAMES)
    return {"winner": winner, "lb": lb, "tbl": tbl, "base": base, "out": out}


if __name__ == "__main__":
    main()
