"""Pre-freeze experiments (walk-forward folds ONLY; holdout stays sealed).

Run:  python -m src.pre_freeze

  1. net_fp_churn leakage probe
  2. roster-change features
  3. player-type clusters
  4. paired-difference selection (ensemble vs xgb)
  5. production blend weight (fit on 2018/19/21/22 with the chosen spec)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import backtest as B
from . import config as C
from . import espn as E
from . import features as F
from . import features_ext as FX
from . import models as M

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)


def _specs() -> dict:
    bt = json.loads((C.REPORTS / "best_by_type.json").read_text())
    out = {}
    for key in ["xgb|b|hist_ctx", "rf|b|hist_ctx", "lgbm|b|hist_ctx"]:
        s = bt[key]
        p = dict(s["params"])
        hl = p.get("half_life")
        p["half_life"] = float(hl) if hl not in (None, "inf", "Infinity") else np.inf
        out[s["family"]] = {"family": s["family"], "spec": s["spec"],
                            "feature_set": s["feature_set"], "params": p}
    return out


# --------------------------------------------------------------------------
# Low-level predict with optional pre-built feature matrices
# --------------------------------------------------------------------------
def _predict(spec, fold, Xtr=None, Xev=None):
    family, sp, fs, params = (spec["family"], spec["spec"],
                              spec["feature_set"], spec["params"])
    if Xtr is None:
        Xtr, Xev = M.build_X(fold.train, fs), M.build_X(fold.evl, fs)
    w = M.recency_weights(fold.train, fold.last_train_season,
                          params.get("half_life", np.inf))
    if sp == "a":
        y = fold.train["target_fppg"].to_numpy(float)
        return M._fit_one(family, params, Xtr, y, w).predict(Xev)
    if sp == "b":
        resid = fold.train["target_fppg"].to_numpy(float) - fold.marcel.predict(fold.train)
        return fold.marcel.predict(fold.evl) + M._fit_one(family, params, Xtr, resid, w).predict(Xev)
    ymin = fold.train["target_mpg"].to_numpy(float)
    yrate = fold.train["target_fppm"].to_numpy(float)
    return (M._fit_one(family, params, Xtr, ymin, w).predict(Xev)
            * M._fit_one(family, params, Xtr, yrate, w).predict(Xev))


def _wmae(evl, pred, mask=None):
    y, w = evl["target_fppg"].to_numpy(float), evl["target_games"].to_numpy(float)
    if mask is not None:
        y, w, pred = y[mask], w[mask], np.asarray(pred)[mask]
    return float(np.average(np.abs(np.asarray(pred) - y), weights=w))


def _per_fold(folds, predfn, mask_fn=None):
    """Return (per-fold MAE array, per-fold preds)."""
    maes, preds = [], []
    for f in folds:
        p = predfn(f)
        preds.append(p)
        m = mask_fn(f) if mask_fn else None
        maes.append(_wmae(f.evl, p, m))
    return np.array(maes), preds


def _paired(folds, name_a, mae_a, name_b, mae_b):
    """Paired per-fold diff (a - b); negative => a better."""
    d = mae_a - mae_b
    se = d.std(ddof=1) / np.sqrt(len(d))
    a_better = int((mae_a < mae_b).sum())
    print(f"    {name_a} vs {name_b}: mean MAE {mae_a.mean():.3f} vs "
          f"{mae_b.mean():.3f}  |  mean diff {d.mean():+.3f} (SE {se:.3f})  |  "
          f"{name_a} better in {a_better}/{len(d)} folds")
    return d, a_better


# ==========================================================================
# Task 1: net_fp_churn leakage probe
# ==========================================================================
def task1_leakage(base, table, specs):
    print("\n" + "=" * 78 + "\n[1] net_fp_churn leakage probe\n" + "=" * 78)
    miss = base.whole_season_missers()
    print(f"  players w/ 1000+ prior FP and ZERO box rows in season t: {len(miss)}")

    base2 = F.Base(retain_missers=True)
    table2 = F.build_modeling_table(base2)
    folds1 = B._make_folds(table, C.WALK_FORWARD_FOLDS)
    folds2 = B._make_folds(table2, C.WALK_FORWARD_FOLDS)

    print("  re-scoring with missers retained on their team (churn recomputed):")
    flagged = []
    for fam in ["xgb", "rf"]:
        m1, _ = _per_fold(folds1, lambda f: _predict(specs[fam], f))
        m2, _ = _per_fold(folds2, lambda f: _predict(specs[fam], f))
        dz = m2.mean() - m1.mean()
        print(f"    {fam}: MAE {m1.mean():.3f} -> {m2.mean():.3f}  (change {dz:+.3f})")
        if abs(dz) > 0.02:
            flagged.append(fam)
    # ensemble
    e1, _ = _ensemble_per_fold(folds1, specs)
    e2, _ = _ensemble_per_fold(folds2, specs)
    de = e2.mean() - e1.mean()
    print(f"    ensemble: MAE {e1.mean():.3f} -> {e2.mean():.3f}  (change {de:+.3f})")
    if abs(de) > 0.02:
        flagged.append("ensemble")
    print(f"  >0.02 change flagged: {flagged or 'NONE'}")

    # 2027 confirm
    op27 = set(base.opening[base.opening["season"] == 2027]["athlete_id"])
    dropped = [n for a, n in C.RULED_OUT_2027.items() if a not in op27]
    print(f"  2027 team-context drops ruled-out: {dropped}")
    return {"missers": len(miss), "flagged": flagged}


def _ensemble_per_fold(folds, specs):
    mx, px = _per_fold(folds, lambda f: _predict(specs["xgb"], f))
    mr, pr = _per_fold(folds, lambda f: _predict(specs["rf"], f))
    preds = [(a + b) / 2 for a, b in zip(px, pr)]
    maes = np.array([_wmae(f.evl, p) for f, p in zip(folds, preds)])
    return maes, preds


# ==========================================================================
# Task 2: roster-change features
# ==========================================================================
def _roster_mask(f):
    ev = f.evl
    nm = ev["rc_net_min"].abs()
    thr = nm.quantile(0.75)
    changed = ev["changed_team"].fillna(0).to_numpy() == 1
    return (changed | (nm.to_numpy() >= thr))


def task2_roster(table, specs):
    print("\n" + "=" * 78 + "\n[2] roster-change features (hist_ctx -> hist_ctx_roster)\n" + "=" * 78)
    folds = B._make_folds(table, C.WALK_FORWARD_FOLDS)
    results = {}
    for fam in ["xgb", "rf"]:
        base_spec = specs[fam]
        rc_spec = dict(base_spec, feature_set="hist_ctx_roster")
        m_base, _ = _per_fold(folds, lambda f: _predict(base_spec, f))
        m_rc, _ = _per_fold(folds, lambda f: _predict(rc_spec, f))
        mb_sub, _ = _per_fold(folds, lambda f: _predict(base_spec, f), _roster_mask)
        mr_sub, _ = _per_fold(folds, lambda f: _predict(rc_spec, f), _roster_mask)
        print(f"  {fam}: OVERALL")
        _, better = _paired(folds, "hist_ctx_roster", m_rc, "hist_ctx", m_base)
        print(f"  {fam}: ROSTER-CHANGE SUBSET (changed team or top-quartile |net min|)")
        _, better_sub = _paired(folds, "hist_ctx_roster", mr_sub, "hist_ctx", mb_sub)
        gain = m_base.mean() - m_rc.mean()
        adopt = (better >= 7 and gain >= 0.02) or (better_sub >= 7 and m_rc.mean() <= m_base.mean() + 1e-9)
        print(f"    -> adopt for {fam}? {adopt}  (overall gain {gain:+.3f}, "
              f"better {better}/9, subset better {better_sub}/9)")
        results[fam] = adopt
    return results


# ==========================================================================
# Task 3: player-type clusters
# ==========================================================================
def _style_df(df):
    out = pd.DataFrame(index=df.index)
    for c in FX.STYLE_COLS:
        out[c] = df["ht_in_in"] if c == "ht_in_in" else df.get("L1_" + c)
    return out


def _cluster_augment(base, fold, k, method):
    ps = base.player_season
    tr_ps = ps[(ps["season"] <= fold.last_train_season) & (ps["mp"] >= 500)]
    cf = FX.ClusterFeature(k, method).fit(tr_ps)
    ctr = cf.assign(_style_df(fold.train))
    cev = cf.assign(_style_df(fold.evl))
    tr = pd.get_dummies(pd.Series(ctr), prefix="clu").astype(float)
    ev = pd.get_dummies(pd.Series(cev), prefix="clu").astype(float)
    ev = ev.reindex(columns=tr.columns, fill_value=0.0)
    tr.index = fold.train.index
    ev.index = fold.evl.index
    return tr, ev


def task3_clusters(base, table, specs):
    print("\n" + "=" * 78 + "\n[3] player-type clusters\n" + "=" * 78)
    folds = B._make_folds(table, C.WALK_FORWARD_FOLDS)
    spec = specs["xgb"]
    m_base, _ = _per_fold(folds, lambda f: _predict(spec, f))

    best = None
    for method in ["kmeans", "gmm"]:
        for k in [4, 6, 8, 10]:
            def pf(f, k=k, method=method):
                tr, ev = _cluster_augment(base, f, k, method)
                Xtr = pd.concat([M.build_X(f.train, "hist_ctx"), tr], axis=1)
                Xev = pd.concat([M.build_X(f.evl, "hist_ctx"), ev], axis=1)
                return _predict(spec, f, Xtr, Xev)
            m_c, _ = _per_fold(folds, pf)
            better = int((m_c < m_base).sum())
            gain = m_base.mean() - m_c.mean()
            print(f"  {method} k={k:2d}: MAE {m_c.mean():.3f} vs base {m_base.mean():.3f}"
                  f"  gain {gain:+.3f}  better {better}/9")
            if best is None or m_c.mean() < best[2]:
                best = (method, k, m_c.mean(), better, gain)
    method, k, mae, better, gain = best
    adopt = better >= 7 and gain >= 0.02
    print(f"  best: {method} k={k}  gain {gain:+.3f}  better {better}/9  -> adopt? {adopt}")
    if method and k:
        _name_clusters(base, method, k)
    return {"best": best, "adopt": adopt}


def _name_clusters(base, method, k):
    ps = base.player_season
    tr = ps[(ps["season"] <= 2026) & (ps["mp"] >= 500)]
    cf = FX.ClusterFeature(k, method).fit(tr)
    cur = ps[(ps["season"] == 2026) & (ps["mp"] >= 500)].copy()
    cur = cur[cur[FX.STYLE_COLS].notna().all(axis=1)]
    cur["clu"] = cf.assign(cur)
    print(f"\n  cluster profiles ({method} k={k}, 2025-26 examples):")
    for cid in sorted(cur["clu"].unique()):
        sub = cur[cur["clu"] == cid]
        prof = sub[["p36_pts", "p36_reb", "p36_ast", "p36_fg3a",
                    "usg_percent", "ht_in_in"]].mean().round(1).to_dict()
        ex = sub.sort_values("mp", ascending=False)["name"].head(5).tolist()
        print(f"    c{cid} (n={len(sub)}): {prof}")
        print(f"        e.g. {', '.join(ex)}")


# ==========================================================================
# Task 4: paired-difference selection
# ==========================================================================
def task4_selection(table, specs):
    print("\n" + "=" * 78 + "\n[4] paired selection: ensemble vs xgb\n" + "=" * 78)
    folds = B._make_folds(table, C.WALK_FORWARD_FOLDS)
    mx, _ = _per_fold(folds, lambda f: _predict(specs["xgb"], f))
    me, _ = _ensemble_per_fold(folds, specs)
    d, ens_better = _paired(folds, "ensemble", me, "xgb", mx)
    freeze = "ensemble" if ens_better >= 7 else "xgb"
    print(f"  ensemble beats xgb in {ens_better}/9 folds -> FREEZE: {freeze}")
    return freeze


# ==========================================================================
# Task 5: production blend weight
# ==========================================================================
def task5_blend(table, frozen_spec):
    print("\n" + "=" * 78 + "\n[5] production blend weight (fit on 2018/19/21/22)\n" + "=" * 78)
    wf = B._make_folds(table, C.WALK_FORWARD_FOLDS)
    seasons = [y for y in E.usable_seasons() if y < 2024]
    dfs = []
    for f in wf:
        if int(f.name) not in seasons:
            continue
        ev = f.evl.reset_index(drop=True)
        d = pd.DataFrame({"actual": ev["target_fppg"].to_numpy(float),
                          "games": ev["target_games"].to_numpy(float),
                          "athlete_id": ev["athlete_id"].to_numpy(),
                          "ours": _predict(frozen_spec, f)})
        e = E.projections(int(f.name))[["athlete_id", "espn_proj_fppg"]]
        d = d.merge(e.rename(columns={"espn_proj_fppg": "espn"}), on="athlete_id")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    y, w = df["actual"].to_numpy(), df["games"].to_numpy()
    best_w, best = 1.0, np.inf
    for ww in np.linspace(0, 1, 101):
        mae = float(np.average(np.abs(ww * df["ours"] + (1 - ww) * df["espn"] - y), weights=w))
        if mae < best:
            best, best_w = mae, ww
    print(f"  seasons used: {seasons}  n={len(df)}")
    print(f"  blend w(ours)={best_w:.2f}  espn_wt={1-best_w:.2f}  WF blend MAE={best:.3f}")
    return float(best_w)


def main():
    specs = _specs()
    base = F.Base()
    table = F.build_modeling_table(base)
    r1 = task1_leakage(base, table, specs)
    r2 = task2_roster(table, specs)
    r3 = task3_clusters(base, table, specs)
    freeze = task4_selection(table, specs)
    frozen_spec = specs["xgb"] if freeze == "xgb" else specs["xgb"]  # spec dict; ensemble handled at freeze time
    w = task5_blend(table, specs["xgb"])
    print("\n" + "=" * 78)
    print(f"SUMMARY: leakage_flagged={r1['flagged']}  roster_adopt={r2}  "
          f"cluster_adopt={r3['adopt']}  selection={freeze}  blend_w={w:.2f}")


if __name__ == "__main__":
    main()
