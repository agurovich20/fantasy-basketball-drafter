"""End-to-end driver.  `python -m src.backtest` reruns everything.

  * data_summary()             -> Checkpoint 1
  * walk_forward_leaderboard() -> Checkpoint 2  (never reads the holdout)
  * holdout stage / production -> Checkpoint 3  (added later; gated on frozen_spec)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import espn as E
from . import features as F
from . import models as M
from . import splits as S
from scipy.stats import spearmanr

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)


# ==========================================================================
# Checkpoint 1: data summary
# ==========================================================================
def _fp_hand_check(box: pd.DataFrame) -> pd.DataFrame:
    want = [("Nikola Jokic", 2024), ("Victor Wembanyama", 2024),
            ("Stephen Curry", 2016)]
    rows = []
    pr = D.played_rows(box)
    for name, season in want:
        sub = pr[(pr["athlete_display_name"] == name) & (pr["season"] == season)]
        if sub.empty:
            continue
        g = sub.sort_values("fp", ascending=False).iloc[0]
        manual = (2 * g["field_goals_made"] - g["field_goals_attempted"]
                  + g["free_throws_made"] - g["free_throws_attempted"]
                  + g["three_point_field_goals_made"] + g["rebounds"]
                  + 2 * g["assists"] + 4 * g["steals"] + 4 * g["blocks"]
                  - 2 * g["turnovers"] + g["points"])
        rows.append({
            "player": name, "season": season, "date": g["game_date"].date(),
            "FGM-A": f"{int(g['field_goals_made'])}-{int(g['field_goals_attempted'])}",
            "FTM-A": f"{int(g['free_throws_made'])}-{int(g['free_throws_attempted'])}",
            "3PM": int(g["three_point_field_goals_made"]),
            "REB": int(g["rebounds"]), "AST": int(g["assists"]),
            "STL": int(g["steals"]), "BLK": int(g["blocks"]),
            "TO": int(g["turnovers"]), "PTS": int(g["points"]),
            "FP_manual": round(float(manual), 1), "FP_code": round(float(g["fp"]), 1),
            "match": bool(abs(manual - g["fp"]) < 1e-9)})
    return pd.DataFrame(rows)


def data_summary(base: F.Base | None = None):
    print("=" * 78 + "\nCHECKPOINT 1  --  DATA SUMMARY\n" + "=" * 78)
    D.download_all()
    base = base or F.Base()
    box, ps = base.box, base.player_season

    print("\n[1] Rows per season")
    played = D.played_rows(box)
    per = pd.DataFrame({
        "box_played_rows": played.groupby("season")["game_id"].count(),
        "player_seasons": ps.groupby("season")["athlete_id"].nunique()})
    table = F.build_modeling_table(base)
    per["modeling_rows"] = table.groupby("season")["athlete_id"].count()
    per = per.fillna(0).astype(int)
    print(per.to_string())
    print(f"  backfilled 2008/2009 player-seasons : {base._n_backfill}")
    print(f"  total modeling rows (2011-2026)     : {len(table):,}")
    print(f"  did-not-play rows kept              : {len(D.dnp_rows(box)):,}")

    print("\n[2] FP formula checked by hand (best game of each player)")
    hc = _fp_hand_check(box)
    print(hc.to_string(index=False))
    print(f"  all match: {bool(hc['match'].all())}")

    print("\n[3] Basketball-Reference match rate")
    rep = base.match_report
    print(f"  {rep['matched']:,}/{rep['espn_player_seasons']:,} = "
          f"{rep['match_rate']*100:.2f}%   unmatched >500 min: "
          f"{len(rep['unmatched_over_500min'])}")

    print("\n[4] Split tables")
    ss = S.split_summary(table)
    print(ss.to_string(index=False))

    print("\n[5] Leakage test")
    lk = F.leakage_test(base, season=2020)
    print(f"  features checked={lk['n_features_checked']}  "
          f"changed={lk['features_that_changed'] or 'NONE'}  passed={lk['passed']}")

    per.to_csv(C.REPORTS / "rows_per_season.csv")
    hc.to_csv(C.REPORTS / "fp_hand_check.csv", index=False)
    rep["unmatched_over_500min"].to_csv(
        C.REPORTS / "bbref_unmatched_over_500min.csv", index=False)
    ss.to_csv(C.REPORTS / "split_row_counts.csv", index=False)
    return table


# ==========================================================================
# Fold machinery
# ==========================================================================
@dataclass
class Fold:
    name: str
    train: pd.DataFrame
    evl: pd.DataFrame
    last_train_season: int
    marcel: M.Marcel
    pool_mask: np.ndarray


def _make_folds(table: pd.DataFrame, fold_defs) -> list[Fold]:
    folds = []
    for bounds, ev in fold_defs:
        tr = S.rows_for_seasons(table, C.train_seasons(bounds))
        evl = S.rows_for_seasons(table, [ev]).reset_index(drop=True)
        marcel = M.Marcel().fit(tr)
        mproj = marcel.predict(evl)
        pool = np.zeros(len(evl), dtype=bool)
        pool[np.argsort(-mproj)[:C.DRAFT_POOL_SIZE]] = True
        folds.append(Fold(f"{ev}", tr.reset_index(drop=True), evl,
                          bounds[1], marcel, pool))
    return folds


def _rolling_defs():
    # train y-3..y-1, validate y, for the same seasons the expanding CV scores
    return [((v - 3, v - 1), v) for v in C.VALIDATION_SEASONS]


# ==========================================================================
# Candidate evaluation
# ==========================================================================
@dataclass
class Result:
    cid: str
    family: str
    spec: str
    feature_set: str
    params: dict
    preds: list          # per-fold prediction arrays
    per_fold: list       # per-fold metric dicts
    mae_mean: float
    mae_std: float
    rmse_mean: float
    bias_mean: float
    pool_spearman: float
    pool_mae: float


def _aggregate(cid, family, spec, fs, params, folds, preds) -> Result:
    pf = [M.season_metrics(f.evl, p, f.pool_mask) for f, p in zip(folds, preds)]
    mae = np.array([m["mae"] for m in pf])
    return Result(
        cid, family, spec, fs, params, preds, pf,
        float(mae.mean()), float(mae.std(ddof=1)),
        float(np.mean([m["rmse"] for m in pf])),
        float(np.mean([m["bias"] for m in pf])),
        float(np.mean([m["pool_spearman"] for m in pf])),
        float(np.mean([m["pool_mae"] for m in pf])))


def _evaluate(cid, family, spec, fs, params, folds) -> Result:
    preds = [M.fit_predict(family, spec, fs, params, f.train, f.evl,
                           f.last_train_season, f.marcel) for f in folds]
    return _aggregate(cid, family, spec, fs, params, folds, preds)


def _baseline_results(folds) -> list[Result]:
    out = []
    ls = [M.last_season_pred(f.evl) for f in folds]
    out.append(_aggregate("baseline_last", "baseline", "-", "-", {}, folds, ls))
    mc = [f.marcel.predict(f.evl) for f in folds]
    out.append(_aggregate("baseline_marcel", "baseline", "-", "-", {}, folds, mc))
    return out


# ==========================================================================
# Checkpoint 2: walk-forward leaderboard
# ==========================================================================
def walk_forward_leaderboard(table: pd.DataFrame):
    print("\n" + "=" * 78 + "\nCHECKPOINT 2  --  WALK-FORWARD LEADERBOARD\n" + "=" * 78)
    folds = _make_folds(table, C.WALK_FORWARD_FOLDS)
    print(f"  {len(folds)} expanding folds, validate "
          f"{folds[0].name}..{folds[-1].name}")
    rng = np.random.default_rng(C.SEED)

    baselines = _baseline_results(folds)
    for b in baselines:
        print(f"    baseline {b.cid:18s} MAE {b.mae_mean:.3f} "
              f"(+/-{b.mae_std:.3f})  poolRho {b.pool_spearman:.3f}")

    # ---- runtime calibration: 1 config/family across all folds ----------
    print("\n[Calibration] timing one config per family to project total runtime")
    per_fit = {}
    for fam in M.FAMILIES:
        c = M.sample_configs(fam, 1, rng)[0]
        t0 = time.time()
        _evaluate(f"cal_{fam}", fam, "b", "hist_ctx", c, folds)
        per_fit[fam] = time.time() - t0  # time for 40-config-equivalent = *40
    # Stage A = 40 configs/family on spec b.  Stage B = best3 x 3 specs x 2 fs;
    # spec c fits two sub-models, so weight specs as a:1 b:1 c:2 -> factor 4/spec-fs row.
    est_a = sum(per_fit[f] * 40 for f in M.FAMILIES)
    specfs_factor = (1 + 1 + 2) * 2  # (a+b+c with c=2x) x 2 feature sets = 8 fits
    est_b = sum(per_fit[f] * 3 * specfs_factor for f in M.FAMILIES)
    print("  per-config time (9 folds): " +
          "  ".join(f"{f}={per_fit[f]:.1f}s" for f in M.FAMILIES))
    print(f"  PROJECTED TOTAL RUNTIME ~ Stage A {est_a/60:.1f} min + "
          f"Stage B {est_b/60:.1f} min = ~{(est_a+est_b)/60:.1f} min")

    # ---- Stage A: 40 configs/family, spec b + hist_ctx ------------------
    print("\n[Stage A] 40 random configs per family on spec (b) + hist+ctx")
    stage_a, best3 = {}, {}
    for fam in M.FAMILIES:
        t0 = time.time()
        cfgs = M.sample_configs(fam, 40, rng)
        res = [_evaluate(f"A_{fam}_{i}", fam, "b", "hist_ctx", c, folds)
               for i, c in enumerate(cfgs)]
        res.sort(key=lambda r: r.mae_mean)
        stage_a[fam] = res
        best3[fam] = res[:3]
        print(f"  {fam:5s}: best MAE {res[0].mae_mean:.3f} "
              f"(+/-{res[0].mae_std:.3f})   [{time.time()-t0:.0f}s]")

    # ---- Stage B: best 3 configs/family x all specs x feature sets ------
    print("\n[Stage B] best-3 configs/family across all target specs x feature sets")
    t0 = time.time()
    results: list[Result] = []
    for fam in M.FAMILIES:
        for rank, r in enumerate(best3[fam]):
            for spec in M.SPECS:
                for fs in M.FEATURE_SETS:
                    cid = f"{fam}|{spec}|{fs}|c{rank}"
                    results.append(_evaluate(cid, fam, spec, fs, r.params, folds))
    print(f"  evaluated {len(results)} candidates   [{time.time()-t0:.0f}s]")

    # ---- Ensemble: equal-weight average of the best 2 distinct candidates
    ranked = sorted(results, key=lambda r: r.mae_mean)
    top2 = ranked[:2]
    ens_preds = [(top2[0].preds[i] + top2[1].preds[i]) / 2 for i in range(len(folds))]
    ensemble = _aggregate(f"ensemble({top2[0].cid}+{top2[1].cid})",
                          "ensemble", "-", "-", {}, folds, ens_preds)
    results.append(ensemble)

    all_rows = baselines + results
    lb = _leaderboard_frame(all_rows).sort_values("MAE_mean").reset_index(drop=True)
    print("\n[Leaderboard] top 20 by mean walk-forward MAE (baselines marked *)")
    print(lb.head(20).to_string(index=False))
    lb.to_csv(C.REPORTS / "walk_forward_leaderboard.csv", index=False)
    _write_per_fold(all_rows, folds)

    # Persist every candidate's params + WF score so the holdout stage can
    # score each one on its own best walk-forward settings without re-searching.
    _persist_candidates(results)

    # ---- selection with 1-SE rule ---------------------------------------
    pick = _select(results, n_folds=len(folds))
    print(f"\n[Selection] best MAE = {ranked[0].cid} ({ranked[0].mae_mean:.3f});"
          f" 1-SE pick (simplest within 1 SE) = {pick.cid} ({pick.mae_mean:.3f})")

    # ---- rolling-window robustness for top 3 ----------------------------
    _rolling_robustness(table, ranked[:3])

    # ---- ESPN benchmark on walk-forward seasons (reference only) --------
    espn_comparison(folds, _result_to_spec(pick), "walk_forward")

    # ---- feature importance for the leader ------------------------------
    _leader_importance(ranked[0], folds)

    _propose_freeze(pick, ranked[0])
    return {"folds": folds, "results": results, "leaderboard": lb,
            "pick": pick, "best": ranked[0]}


def _leaderboard_frame(rows: list[Result]) -> pd.DataFrame:
    recs = []
    for r in rows:
        star = "*" if r.family == "baseline" else ""
        recs.append({
            "candidate": star + r.cid, "family": r.family, "spec": r.spec,
            "features": r.feature_set, "MAE_mean": round(r.mae_mean, 3),
            "MAE_std": round(r.mae_std, 3), "RMSE_mean": round(r.rmse_mean, 3),
            "bias": round(r.bias_mean, 3), "pool_rho": round(r.pool_spearman, 3),
            "pool_MAE": round(r.pool_mae, 3)})
    return pd.DataFrame(recs)


def _write_per_fold(rows: list[Result], folds):
    recs = []
    for r in rows:
        for f, m in zip(folds, r.per_fold):
            recs.append({"candidate": r.cid, "season": f.name,
                         "MAE": round(m["mae"], 3), "RMSE": round(m["rmse"], 3),
                         "bias": round(m["bias"], 3),
                         "pool_rho": round(m["pool_spearman"], 3)})
    pd.DataFrame(recs).to_csv(C.REPORTS / "walk_forward_per_fold.csv", index=False)


_COMPLEXITY = {"ridge": 1, "lgbm": 2, "xgb": 2, "rf": 3, "ensemble": 4}


def _complexity(r: Result) -> tuple:
    fam = _COMPLEXITY.get(r.family, 5)
    spec = {"a": 0, "b": 1, "c": 2, "-": 3}[r.spec]
    fs = {"hist": 0, "hist_ctx": 1, "-": 2}[r.feature_set]
    return (fam, fs, spec)


def _select(results: list[Result], n_folds: int) -> Result:
    best = min(results, key=lambda r: r.mae_mean)
    se = best.mae_std / np.sqrt(n_folds)
    within = [r for r in results if r.mae_mean <= best.mae_mean + se]
    return min(within, key=_complexity)


def _rolling_robustness(table, top3):
    print("\n[Rolling-window robustness] top-3 candidates, 3-season rolling train "
          "(not used for selection)")
    rfolds = _make_folds(table, _rolling_defs())
    recs = []
    for r in top3:
        preds = [M.fit_predict(r.family, r.spec, r.feature_set, r.params,
                               f.train, f.evl, f.last_train_season, f.marcel)
                 for f in rfolds]
        agg = _aggregate(r.cid, r.family, r.spec, r.feature_set, r.params,
                         rfolds, preds)
        recs.append({"candidate": r.cid, "expanding_MAE": round(r.mae_mean, 3),
                     "rolling_MAE": round(agg.mae_mean, 3),
                     "rolling_std": round(agg.mae_std, 3)})
    print(pd.DataFrame(recs).to_string(index=False))
    pd.DataFrame(recs).to_csv(C.REPORTS / "rolling_robustness.csv", index=False)


def _leader_importance(leader: Result, folds):
    print(f"\n[Leader feature importance] {leader.cid}")
    f = folds[-1]
    Xtr = M.build_X(f.train, leader.feature_set)
    Xev = M.build_X(f.evl, leader.feature_set)
    w = M.recency_weights(f.train, f.last_train_season,
                          leader.params.get("half_life", np.inf))
    try:
        if leader.spec == "b":
            y = f.train["target_fppg"].to_numpy(float) - f.marcel.predict(f.train)
        elif leader.spec == "a":
            y = f.train["target_fppg"].to_numpy(float)
        else:
            y = f.train["target_mpg"].to_numpy(float)
        est = M._fit_one(leader.family, leader.params, Xtr, y, w)
        if leader.family in ("xgb", "lgbm", "rf"):
            import shap
            base_model = est.named_steps["est"] if leader.family == "rf" else est
            Xshap = (est.named_steps["imp"].transform(Xev)
                     if leader.family == "rf" else Xev)
            if leader.family == "rf":
                Xshap = pd.DataFrame(Xshap, columns=Xtr.columns)
            expl = shap.TreeExplainer(base_model)
            sv = expl.shap_values(Xshap)
            imp = np.abs(sv).mean(axis=0)
        else:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(est, Xev,
                                        f.evl["target_fppg"].to_numpy(float),
                                        n_repeats=5, random_state=C.SEED)
            imp = pi.importances_mean
        top = (pd.Series(imp, index=Xtr.columns).sort_values(ascending=False)
               .head(15))
        print(top.round(4).to_string())
        top.to_csv(C.REPORTS / "leader_feature_importance.csv")
    except Exception as e:  # importance is diagnostic, never fatal
        print(f"  (importance skipped: {e})")


def _json_safe(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif v is None or isinstance(v, (int, float, str, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _persist_candidates(results: list[Result]):
    """Write all Stage-B candidates and the best config per (family,spec,fs)."""
    recs = [{"cid": r.cid, "family": r.family, "spec": r.spec,
             "feature_set": r.feature_set, "params": _json_safe(r.params),
             "mae_mean": r.mae_mean, "mae_std": r.mae_std,
             "rmse_mean": r.rmse_mean, "bias_mean": r.bias_mean,
             "pool_spearman": r.pool_spearman, "pool_mae": r.pool_mae}
            for r in results]
    (C.REPORTS / "candidates.json").write_text(json.dumps(recs, indent=2))

    best_by_type = {}
    for r in results:
        key = f"{r.family}|{r.spec}|{r.feature_set}"
        if key not in best_by_type or r.mae_mean < best_by_type[key]["mae_mean"]:
            best_by_type[key] = {"family": r.family, "spec": r.spec,
                                 "feature_set": r.feature_set,
                                 "params": _json_safe(r.params),
                                 "mae_mean": r.mae_mean, "cid": r.cid}
    (C.REPORTS / "best_by_type.json").write_text(
        json.dumps(best_by_type, indent=2))


def _result_to_spec(r: Result) -> dict:
    return {"family": r.family, "spec": r.spec,
            "feature_set": r.feature_set, "params": _json_safe(r.params),
            "cid": r.cid}


def _propose_freeze(pick: Result, best: Result):
    print("\n[Proposal] spec to freeze (awaiting approval; holdout untouched):")
    print(f"  family={pick.family} spec={pick.spec} features={pick.feature_set}")
    print(f"  params={pick.params}")
    print(f"  walk-forward MAE {pick.mae_mean:.3f} +/- {pick.mae_std:.3f}")
    proposal = {
        "family": pick.family, "spec": pick.spec,
        "feature_set": pick.feature_set, "params": pick.params,
        "walk_forward_mae_mean": pick.mae_mean,
        "walk_forward_mae_std": pick.mae_std,
        "note": "PROPOSED -- not yet frozen; awaiting user OK before holdout."}
    (C.REPORTS / "proposed_spec.json").write_text(json.dumps(proposal, indent=2))


# ==========================================================================
# Checkpoint 3: holdout  (gated: requires frozen_spec.json)
# ==========================================================================
NAMED_PLAYERS = ["Nikola Jokic", "Victor Wembanyama",
                 "Shai Gilgeous-Alexander", "Cade Cunningham",
                 "Jalen Johnson", "Amen Thompson", "Tyrese Haliburton",
                 "Jayson Tatum", "Joel Embiid", "LeBron James"]


def holdout_stage(table: pd.DataFrame, base: F.Base):
    if not C.FROZEN_SPEC_PATH.exists():
        raise SystemExit("frozen_spec.json does not exist -- freeze a spec "
                         "(after ⏸2 approval) before scoring the holdout.")
    frozen = json.loads(C.FROZEN_SPEC_PATH.read_text())
    print("\n" + "=" * 78 + "\nCHECKPOINT 3  --  HOLDOUT (2024-2026)\n" + "=" * 78)
    print(f"  frozen spec: {frozen['family']}|{frozen['spec']}|"
          f"{frozen['feature_set']}  params={frozen['params']}")

    folds = _make_folds(table, C.HOLDOUT_FOLDS)
    baselines = _baseline_results(folds)

    # frozen spec, refit per holdout fold
    fr_preds = [M.predict_with_spec(frozen, f.train, f.evl, f.last_train_season)
                for f in folds]
    frozen_res = _aggregate("FROZEN:" + frozen.get("cid", "spec"),
                            frozen["family"], frozen["spec"],
                            frozen["feature_set"], frozen["params"],
                            folds, fr_preds)

    print("\n[Per-season holdout: frozen spec vs baselines]")
    rows = []
    for i, f in enumerate(folds):
        for label, res in [("frozen", frozen_res), ("last", baselines[0]),
                           ("marcel", baselines[1])]:
            m = res.per_fold[i]
            rows.append({"season": f.name, "model": label,
                         "MAE": round(m["mae"], 3), "RMSE": round(m["rmse"], 3),
                         "bias": round(m["bias"], 3),
                         "pool_rho": round(m["pool_spearman"], 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    # every other candidate type, own best WF settings (comparison only)
    comp = []
    if (C.REPORTS / "best_by_type.json").exists():
        best_by_type = json.loads((C.REPORTS / "best_by_type.json").read_text())
        for key, spec in best_by_type.items():
            preds = [M.predict_with_spec(spec, f.train, f.evl, f.last_train_season)
                     for f in folds]
            r = _aggregate(key, spec["family"], spec["spec"],
                           spec["feature_set"], spec["params"], folds, preds)
            comp.append(r)
    comp_rows = _leaderboard_frame(baselines + [frozen_res] + comp)
    comp_rows = comp_rows.sort_values("MAE_mean").reset_index(drop=True)
    print("\n[Holdout leaderboard] frozen + all candidates (comparison only)")
    print(comp_rows.to_string(index=False))
    comp_rows.to_csv(C.REPORTS / "holdout_leaderboard.csv", index=False)

    # named players: predicted vs actual
    _named_player_table(folds, fr_preds)

    # ---- ESPN benchmark on holdout + blend + disagreement ---------------
    espn_comparison(folds, frozen, "holdout")
    wf_folds = _make_folds(table, C.WALK_FORWARD_FOLDS)
    w = fit_blend_weight(wf_folds, frozen)
    blend_recs = []
    for f in folds:
        if int(f.name) not in E.usable_seasons():
            continue
        df = _common_eval_df(f, frozen)
        y, wt = df["actual"].to_numpy(), df["games"].to_numpy()
        for label, pred in [("ours", df["ours"].to_numpy()),
                            ("espn", df["espn"].to_numpy()),
                            (f"blend(w={w:.2f})",
                             w * df["ours"].to_numpy() + (1 - w) * df["espn"].to_numpy())]:
            blend_recs.append({"season": f.name, "model": label, "n": len(df),
                               "MAE": round(M.w_mae(y, pred, wt), 3),
                               "RMSE": round(M.w_rmse(y, pred, wt), 3)})
    print("\n[Blend scored on holdout seasons]")
    print(pd.DataFrame(blend_recs).to_string(index=False))
    pd.DataFrame(blend_recs).to_csv(C.REPORTS / "blend_holdout.csv", index=False)

    disagreement_analysis(folds, frozen)

    # holdout residuals of the frozen spec (for projection bands)
    _save_holdout_residuals(folds, fr_preds)
    return {"frozen": frozen, "frozen_res": frozen_res}


def _named_player_table(folds, fr_preds):
    recs = []
    for f, pred in zip(folds, fr_preds):
        ev = f.evl.reset_index(drop=True)
        for nm in NAMED_PLAYERS:
            idx = ev.index[ev["name"] == nm]
            if len(idx):
                i = idx[0]
                recs.append({"season": f.name, "player": nm,
                             "pred_FPPG": round(float(pred[i]), 1),
                             "actual_FPPG": round(float(ev.loc[i, "target_fppg"]), 1),
                             "games": int(ev.loc[i, "target_games"])})
    df = pd.DataFrame(recs)
    print("\n[Named players: predicted vs actual, holdout seasons]")
    if len(df):
        print(df.to_string(index=False))
    df.to_csv(C.REPORTS / "holdout_named_players.csv", index=False)


def _save_holdout_residuals(folds, fr_preds):
    recs = []
    for f, pred in zip(folds, fr_preds):
        ev = f.evl.reset_index(drop=True)
        actual = ev["target_fppg"].to_numpy(float)
        for p, a in zip(pred, actual):
            recs.append({"pred": float(p), "actual": float(a),
                         "resid": float(a - p)})
    pd.DataFrame(recs).to_csv(C.REPORTS / "holdout_residuals.csv", index=False)


# ==========================================================================
# ESPN benchmark (evaluation only -- never used for model selection)
# ==========================================================================
def _common_eval_df(fold: Fold, spec: dict) -> pd.DataFrame:
    """Per-fold frame on players ESPN also projected, with every model's FPPG."""
    ev = fold.evl.reset_index(drop=True)
    df = pd.DataFrame({
        "athlete_id": ev["athlete_id"].to_numpy(),
        "name": ev["name"].to_numpy(),
        "season": int(fold.name),
        "actual": ev["target_fppg"].to_numpy(float),
        "games": ev["target_games"].to_numpy(float),
        "marcel": fold.marcel.predict(ev),
        "last": M.last_season_pred(ev),
        "ours": M.predict_with_spec(spec, fold.train, ev, fold.last_train_season),
    })
    e = E.all_projections()
    e = e[e["season"] == int(fold.name)][["athlete_id", "espn_proj_fppg"]]
    df = df.merge(e.rename(columns={"espn_proj_fppg": "espn"}),
                  on="athlete_id", how="inner")
    return df


def _matched_metrics(df: pd.DataFrame, col: str) -> dict:
    y, p, w = df["actual"].to_numpy(), df[col].to_numpy(), df["games"].to_numpy()
    pool = df.nlargest(min(C.DRAFT_POOL_SIZE, len(df)), "marcel").index
    rho = (float(spearmanr(df.loc[pool, col], df.loc[pool, "actual"]).statistic)
           if len(pool) > 2 else np.nan)
    return {"MAE": M.w_mae(y, p, w), "RMSE": M.w_rmse(y, p, w),
            "bias": M.w_bias(y, p, w), "pool_rho": rho}


def espn_comparison(folds, spec, title: str) -> pd.DataFrame:
    """Matched per-season ESPN vs Marcel vs last vs ours (same players)."""
    print(f"\n[ESPN comparison -- {title}]  games-weighted, on players ESPN "
          "projected (reference only)")
    recs = []
    for f in folds:
        if int(f.name) not in E.usable_seasons():
            continue
        df = _common_eval_df(f, spec)
        if len(df) < 10:
            continue
        for model in ["espn", "marcel", "last", "ours"]:
            m = _matched_metrics(df, model)
            recs.append({"season": f.name, "model": model, "n": len(df),
                         **{k: round(v, 3) for k, v in m.items()}})
    out = pd.DataFrame(recs)
    if len(out):
        print(out.to_string(index=False))
        avg = (out.groupby("model")[["MAE", "RMSE", "bias", "pool_rho"]]
                  .mean().round(3))
        print("  average across ESPN-covered seasons:")
        print(avg.to_string())
        out.to_csv(C.REPORTS / f"espn_comparison_{title}.csv", index=False)
    return out


def fit_blend_weight(wf_folds, spec) -> float:
    """w minimizing games-weighted MAE of w*ours+(1-w)*espn over WF seasons."""
    dfs = [_common_eval_df(f, spec) for f in wf_folds
           if int(f.name) in E.usable_seasons()]
    dfs = [d for d in dfs if len(d) >= 10]
    df = pd.concat(dfs, ignore_index=True)
    y, w = df["actual"].to_numpy(), df["games"].to_numpy()
    best_w, best = 0.5, np.inf
    for ww in np.linspace(0, 1, 101):
        pred = ww * df["ours"].to_numpy() + (1 - ww) * df["espn"].to_numpy()
        mae = M.w_mae(y, pred, w)
        if mae < best:
            best, best_w = mae, ww
    print(f"\n[Blend] fit on walk-forward ESPN seasons: "
          f"w*ours+(1-w)*espn  ->  w={best_w:.2f}  (WF MAE {best:.3f})")
    return float(best_w)


def disagreement_analysis(holdout_folds, spec, thresh: float = 3.0):
    print(f"\n[Disagreement analysis -- holdout]  |ours - espn| >= {thresh} FPPG")
    allrows = []
    for f in holdout_folds:
        if int(f.name) not in E.usable_seasons():
            continue
        df = _common_eval_df(f, spec)
        df["diff"] = df["ours"] - df["espn"]
        df["ours_err"] = (df["ours"] - df["actual"]).abs()
        df["espn_err"] = (df["espn"] - df["actual"]).abs()
        allrows.append(df)
        top = df.reindex(df["diff"].abs().sort_values(ascending=False).index).head(10)
        print(f"\n  {f.name}: 10 biggest disagreements")
        print(top[["name", "ours", "espn", "actual", "diff"]].round(1)
              .to_string(index=False))
    full = pd.concat(allrows, ignore_index=True)
    for label, sub in [("ours 3+ ABOVE espn", full[full["diff"] >= thresh]),
                       ("ours 3+ BELOW espn", full[full["diff"] <= -thresh])]:
        if not len(sub):
            continue
        oe, ee = sub["ours_err"].mean(), sub["espn_err"].mean()
        winner = "ours" if oe < ee else "espn"
        print(f"\n  {label}: n={len(sub)}  mean|ours-actual|={oe:.2f}  "
              f"mean|espn-actual|={ee:.2f}  -> {winner} closer by {abs(oe-ee):.2f}")
    full.to_csv(C.REPORTS / "espn_disagreement_holdout.csv", index=False)


def main():
    table = data_summary()
    walk_forward_leaderboard(table)
    # ⏸2 stop: holdout + production run only after the spec is frozen & approved.


if __name__ == "__main__":
    main()
