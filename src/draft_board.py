"""Build output/draft_board_2027.csv for Google Sheets.

  python -m src.draft_board          # rebuild from cached ESPN data
  python -m src.draft_board refresh  # re-pull ESPN projections+ADP, rebuild

espn_rank note: ESPN's draftRanksByRankType exposes only STANDARD (8-category)
and ROTO -- NOT a points-league rank -- so espn_rank is the fallback: rank by
ESPN projected total points (FPPG x GP) rescored in our league.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import espn as E
from . import features as F
from . import features_ext as FX
from . import models as M
from . import project as PROJ
from . import splits as S

SLOT_POS = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C"}
REPLACEMENT_RANK = 150        # 150th-best player sets the VORP baseline R
TOP_N = 300
PROD_SEASON = 2027

# Known injuries for 2026-27 (override ESPN status; our information).
KNOWN_INJURY = {
    "jimmy butler": "out (season)",
    "ariel hukporti": "out (season)",
    "jamir watkins": "out (season)",
    "johnny furphy": "out (season)",
    "kyrie irving": "knee (ACL) - ramp up",
    "brandon ingram": "injured",
    "shaedon sharpe": "injured",
    "mark williams": "injured",
}


def _espn_meta(year=PROD_SEASON) -> pd.DataFrame:
    """Per-player ESPN metadata: STANDARD rank, ADP, positions, injury."""
    rows = []
    for p in E._load(year):
        pl = p["player"]
        slots = pl.get("eligibleSlots") or []
        pos = "/".join(SLOT_POS[s] for s in sorted(slots) if s in SLOT_POS)
        own = pl.get("ownership") or {}
        adp = own.get("averageDraftPosition")
        std = (pl.get("draftRanksByRankType") or {}).get("STANDARD", {})
        rows.append({
            "athlete_id": int(p["id"]),
            "positions": pos or None,
            "adp": adp if (adp and adp > 0) else np.nan,
            "espn_std_rank": std.get("rank") or np.nan,
            "injuryStatus": pl.get("injuryStatus"),
        })
    return pd.DataFrame(rows)


def _cluster_labeler(base):
    """Fit k-means(6) on <=2026 style; label the 6 clusters with DISTINCT names
    by their RELATIVE centroid profiles (robust to absolute thresholds)."""
    ps = base.player_season
    tr = ps[(ps["season"] <= 2026) & (ps["mp"] >= 500)]
    cf = FX.ClusterFeature(6, "kmeans").fit(tr)
    cur = tr[tr[FX.STYLE_COLS].notna().all(axis=1)].copy()
    cur["clu"] = cf.assign(cur)
    prof = cur.groupby("clu")[["p36_pts", "p36_reb", "p36_ast", "p36_fg3a",
                               "usg_percent", "ht_in_in", "blk_percent"]].mean()

    id2label, used = {}, set()
    # two tallest clusters are bigs; stretch vs rim by 3PA rate
    bigs = prof["ht_in_in"].sort_values(ascending=False).index[:2].tolist()
    if prof.loc[bigs[0], "p36_fg3a"] >= prof.loc[bigs[1], "p36_fg3a"]:
        id2label[bigs[0]], id2label[bigs[1]] = "Stretch big", "Rim-running big"
    else:
        id2label[bigs[0]], id2label[bigs[1]] = "Rim-running big", "Stretch big"
    used.update(bigs)
    rest = [c for c in prof.index if c not in used]
    rp = prof.loc[rest]
    for metric, lab in [("p36_ast", "Lead guard"),
                        ("usg_percent", "Primary scorer"),
                        ("p36_fg3a", "3&D wing")]:
        pick = rp[~rp.index.isin(used)][metric].idxmax()
        id2label[pick] = lab
        used.add(pick)
    for c in rest:                               # the remaining one
        if c not in used:
            id2label[c] = "Connector wing"
    return cf, id2label


def _assign_types(base, board_ids):
    """player_type from each player's 2026 style (blank if none)."""
    cf, id2label = _cluster_labeler(base)
    ps = base.player_season
    s26 = ps[(ps["season"] == 2026)].set_index("athlete_id")
    s26 = s26[s26[FX.STYLE_COLS].notna().all(axis=1)]
    out = {}
    if len(s26):
        clu = cf.assign(s26)
        for aid, c in zip(s26.index, clu):
            out[aid] = id2label.get(c, "")
    return out


def _tiers(vorp: np.ndarray, natural_breaks: int = 14,
           max_tier_size: int = 30) -> np.ndarray:
    """Tier breaks at the ~`natural_breaks` clearest VORP drops, plus a forced
    break if a tier exceeds max_tier_size (so the flat bottom isn't one blob)."""
    gaps = -np.diff(vorp)                      # sorted desc -> nonneg gaps
    thr = np.sort(gaps)[::-1][min(natural_breaks, len(gaps) - 1)]
    tier = np.ones(len(vorp), dtype=int)
    t, count = 1, 1
    for i in range(1, len(vorp)):
        if gaps[i - 1] > thr or count >= max_tier_size:
            t += 1; count = 0
        tier[i] = t; count += 1
    return tier


def build() -> pd.DataFrame:
    frozen = json.loads(C.FROZEN_SPEC_PATH.read_text())
    spec = {"family": frozen["family"], "spec": frozen["spec"],
            "feature_set": frozen["feature_set"], "params": dict(frozen["params"])}
    hl = spec["params"].get("half_life")
    spec["params"]["half_life"] = (float(hl) if hl not in (None, "inf", "Infinity")
                                   else np.inf)
    w = frozen.get("blend", {}).get("weight_ours", 0.55)

    base = F.Base()
    table = F.build_modeling_table(base)
    prod = F.build_production_table(base)

    # model projection (xgb refit on 2011-2026)
    train = S.rows_for_seasons(table, C.train_seasons(C.PRODUCTION_FOLD[0]))
    model = M.predict_with_spec(spec, train, prod, C.PRODUCTION_FOLD[0][1])
    model_map = dict(zip(prod["athlete_id"], np.clip(model, 0, None)))
    prod_idx = prod.set_index("athlete_id")

    esp = E.projections(PROD_SEASON)          # complete ESPN projections
    meta = _espn_meta()
    ros = base.rosters_2027.set_index("athlete_id")
    types = _assign_types(base, esp["athlete_id"].tolist())

    df = esp.merge(meta, on="athlete_id", how="left")
    df["espn_fppg"] = df["espn_proj_fppg"]
    df["proj_games"] = df["espn_proj_gp"]
    df["model_fppg"] = df["athlete_id"].map(model_map)
    # blend: 0.55*ours+0.45*ESPN; ESPN-only where no model (rookies)
    bl = w * df["model_fppg"] + (1 - w) * df["espn_fppg"]
    df["blend_fppg"] = bl.where(df["model_fppg"].notna(), df["espn_fppg"])
    df["season_pts"] = df["blend_fppg"] * df["proj_games"]

    # VORP with replacement = 150th-best player's blend FPPG (ranked by value)
    order = df.sort_values("season_pts", ascending=False).reset_index(drop=True)
    R = float(order.loc[min(REPLACEMENT_RANK - 1, len(order) - 1), "blend_fppg"])
    df["vorp"] = df["proj_games"] * (df["blend_fppg"] - R)

    # ranks
    df = df.sort_values("vorp", ascending=False).reset_index(drop=True)
    df["model_rank"] = np.arange(1, len(df) + 1)
    df["espn_value"] = df["espn_fppg"] * df["proj_games"]
    df["espn_rank"] = df["espn_value"].rank(ascending=False, method="first").astype(int)
    df["rank_diff"] = df["espn_rank"] - df["model_rank"]
    df["flag"] = np.where(df["rank_diff"] >= 15, "TARGET",
                          np.where(df["rank_diff"] <= -15, "FADE", ""))

    # identity / meta
    df["player"] = df["name"]
    df["team"] = df["athlete_id"].map(ros["team"]).fillna("")
    df["age"] = df["athlete_id"].map(
        lambda a: prod_idx["age"].get(a) if a in prod_idx.index else np.nan)
    df["age"] = df["age"].fillna(df["athlete_id"].map(ros["age"]))
    ct = df["athlete_id"].map(
        lambda a: prod_idx["changed_team"].get(a) if a in prod_idx.index else np.nan)
    df["changed_team"] = np.where(ct == 1, "Y", np.where(ct == 0, "N", ""))
    df["player_type"] = df["athlete_id"].map(types).fillna("")

    # injury note: known list overrides ESPN status
    def injury(row):
        nm = D.normalize_name(row["player"])
        if nm in KNOWN_INJURY:
            return KNOWN_INJURY[nm]
        st = row.get("injuryStatus")
        if st == "OUT":
            return "out"
        if st == "DAY_TO_DAY":
            return "day-to-day"
        return ""
    df["injury_note"] = df.apply(injury, axis=1)

    # bands (bootstrap holdout errors around the blend projection)
    resid = pd.read_csv(C.REPORTS / "holdout_residuals.csv")
    p10, p90 = PROJ._residual_bands(df["blend_fppg"].to_numpy(float), resid)
    df["fppg_p10"] = np.clip(p10, 0, None)
    df["fppg_p90"] = p90

    board = df.head(TOP_N).copy()
    board["tier"] = _tiers(board["vorp"].to_numpy(float))

    cols = ["model_rank", "espn_rank", "adp", "rank_diff", "flag", "player",
            "team", "positions", "age", "blend_fppg", "model_fppg", "espn_fppg",
            "fppg_p10", "fppg_p90", "proj_games", "season_pts", "vorp", "tier",
            "changed_team", "injury_note", "player_type"]
    out = board[cols].copy()
    for c in ["adp", "age", "blend_fppg", "model_fppg", "espn_fppg", "fppg_p10",
              "fppg_p90", "proj_games", "season_pts", "vorp"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(1)
    out["adp"] = out["adp"].map(lambda x: "" if pd.isna(x) else x)
    out["model_fppg"] = out["model_fppg"].map(lambda x: "" if pd.isna(x) else x)

    path = C.OUTPUT / "draft_board_2027.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"wrote {path}  ({len(out)} players)  VORP replacement R={R:.1f} "
          f"(150th-best blend FPPG)")
    print("espn_rank source: FALLBACK = ESPN projected total points "
          "(no points-league rank in draftRanksByRankType; only STANDARD/ROTO)")
    return out


def _show(out: pd.DataFrame):
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", None)
    show = ["model_rank", "espn_rank", "adp", "rank_diff", "flag", "player",
            "team", "positions", "age", "blend_fppg", "proj_games", "vorp",
            "tier", "injury_note", "player_type"]
    print("\n=== TOP 30 ===")
    print(out[show].head(30).to_string(index=False))
    tf = out[out["flag"] != ""]
    print(f"\n=== TARGET / FADE ({len(tf)}) ===")
    print(tf[show].to_string(index=False))


def refresh():
    print("Refreshing ESPN 2027 projections + ADP ...")
    E.fetch_season(2027, force=True)
    E.all_projections.cache_clear()
    out = build()
    _show(out)
    return out


def main():
    import sys
    if "refresh" in sys.argv:
        refresh()
    else:
        _show(build())


if __name__ == "__main__":
    main()
