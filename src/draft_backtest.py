"""Mock-draft backtest with statistical inference (season-value edition).

Primary test: drafting with OUR out-of-fold projections vs ESPN's, against a
field that drafts by ADP (or ESPN projected value) with no randomness, measure
the edge in the team's actual best-10 SEASON total (with a waiver floor),
averaged over all 14 slots, per season.

Value conventions (fix set 1):
  * ranking value = projected FPPG x ESPN projected games played (ESPN GP for
    both sources, so the ranking gap is pure FPPG skill); rookies / no-history
    players use ESPN's value in every ranking.
  * actual value  = actual season fantasy-point TOTAL; whole-season absences = 0.
  * pool          = ESPN's projected players, top 250 by ESPN value (NOT the
    modeling table, whose 20-game filter drops injury busts).
  * waiver floor W = actual season total of the 15th-best UNDRAFTED player in
    that draft.  Weekly edge = season edge / 22.

Opponents (fix 2/3) draft by ADP when ownership.averageDraftPosition checks
out, else by ESPN projected value; randomized opponents perturb DRAFT SPOTS:
noisy rank = ESPN/ADP rank + Normal(0, sigma) picks.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from . import backtest as B
from . import config as C
from . import espn as E
from . import features as F
from . import models as M
from . import splits as S

TEAMS = 14
STARTERS = 10
ROSTER = 13
POOL = 250
WEEKS = 22
WF_ESPN = [2018, 2019, 2021, 2022]
HOLDOUT_ESPN = [2024, 2025, 2026]


def _frozen_spec() -> dict:
    s = json.loads(C.FROZEN_SPEC_PATH.read_text())
    p = dict(s["params"])
    hl = p.get("half_life")
    p["half_life"] = float(hl) if hl not in (None, "inf", "Infinity") else np.inf
    return {"family": s["family"], "spec": s["spec"],
            "feature_set": s["feature_set"], "params": p}


def _fold_for_season(season):
    for bounds, ev in (C.WALK_FORWARD_FOLDS + C.HOLDOUT_FOLDS):
        if ev == season:
            return bounds
    raise ValueError(season)


# --------------------------------------------------------------------------
# Season pool (ESPN-based; season-value)
# --------------------------------------------------------------------------
def build_pool(spec, base, table, season):
    esp = E.projections(season).copy()
    esp["espn_value"] = esp["espn_proj_fp_total"]            # FPPG x proj GP
    pool = esp.sort_values("espn_value", ascending=False).head(POOL).copy()

    bounds = _fold_for_season(season)
    train = S.rows_for_seasons(table, C.train_seasons(bounds))
    pred_tbl = F.build_predict_table(base, season, pool["athlete_id"].tolist())
    our = M.predict_with_spec(spec, train, pred_tbl, bounds[1])
    our_map = dict(zip(pred_tbl["athlete_id"].to_numpy(), our))
    pool["our_fppg"] = pool["athlete_id"].map(our_map)
    pool["our_value"] = (pool["our_fppg"] * pool["espn_proj_gp"]).fillna(
        pool["espn_value"])                                  # rookies -> ESPN
    # blend ranking = 0.55*ours + 0.45*ESPN FPPG (fallback ESPN where no model)
    blend_fppg = (0.55 * pool["our_fppg"] + 0.45 * pool["espn_proj_fppg"])
    pool["blend_value"] = (blend_fppg * pool["espn_proj_gp"]).fillna(
        pool["espn_value"])

    fp = (base.player_season[base.player_season["season"] == season]
          .set_index("athlete_id")["fp_total"].to_dict())
    pool["actual"] = pool["athlete_id"].map(lambda a: fp.get(a, 0.0)).fillna(0.0)

    use_adp = E.adp_valid(season)
    if use_adp:
        adp = E.adp(season).set_index("athlete_id")["adp"].to_dict()
        big = (max(adp.values()) + POOL) if adp else 9999.0
        pool["opp_key"] = pool["athlete_id"].map(lambda a: adp.get(a, big))
        pool["opp_ascending"] = True       # lower ADP drafted first
    else:
        pool["opp_key"] = -pool["espn_value"]
        pool["opp_ascending"] = True
    return pool.reset_index(drop=True), use_adp


# --------------------------------------------------------------------------
# Snake draft
# --------------------------------------------------------------------------
def _snake_team_order(teams, rounds):
    order = []
    for r in range(rounds):
        order.extend(range(teams) if r % 2 == 0 else range(teams - 1, -1, -1))
    return order


_TEAM_ORDER = _snake_team_order(TEAMS, ROSTER)


def _draft(opp_order, our_order, our_team):
    """Return (my pick indices, drafted boolean mask)."""
    n = len(opp_order)
    taken = np.zeros(n, dtype=bool)
    fp = op = 0
    mine = []
    for team in _TEAM_ORDER:
        if team == our_team:
            while taken[our_order[op]]:
                op += 1
            pick = our_order[op]; op += 1
        else:
            while taken[opp_order[fp]]:
                fp += 1
            pick = opp_order[fp]; fp += 1
        taken[pick] = True
        if team == our_team:
            mine.append(pick)
    return mine, taken


def _value(actual, mine, taken):
    und = np.sort(actual[~taken])[::-1]
    W = und[14] if len(und) > 14 else (und[-1] if len(und) else 0.0)
    v = np.maximum(actual[mine], W)
    return float(np.sort(v)[::-1][:STARTERS].sum())


def season_edge(actual, opp_order, our_order, espn_order):
    edges = []
    for slot in range(TEAMS):
        mo, to = _draft(opp_order, our_order, slot)
        me, te = _draft(opp_order, espn_order, slot)
        edges.append(_value(actual, mo, to) - _value(actual, me, te))
    return float(np.mean(edges))


def _orders(pool, treat="our_value"):
    opp = np.argsort(pool["opp_key"].to_numpy())             # ascending
    our = np.argsort(-pool[treat].to_numpy())
    espn = np.argsort(-pool["espn_value"].to_numpy())
    return opp, our, espn


def point_edge(pool, treat="our_value"):
    actual = pool["actual"].to_numpy(float)
    opp, our, espn = _orders(pool, treat)
    return season_edge(actual, opp, our, espn)


def per_slot_edges(pool, treat="our_value"):
    actual = pool["actual"].to_numpy(float)
    opp, our, espn = _orders(pool, treat)
    out = []
    for slot in range(TEAMS):
        mo, to = _draft(opp, our, slot)
        me, te = _draft(opp, espn, slot)
        out.append(_value(actual, mo, to) - _value(actual, me, te))
    return np.array(out)


# --------------------------------------------------------------------------
# Task 1/5: randomized opponents (noise in DRAFT SPOTS)
# --------------------------------------------------------------------------
def sim_spot_noise(pool, sigma, n_sims=300, seed=C.SEED):
    rng = np.random.default_rng(seed)
    actual = pool["actual"].to_numpy(float)
    opp, our, espn = _orders(pool)
    base_rank = np.empty(len(opp))
    base_rank[opp] = np.arange(len(opp))         # opp pick-spot of each player
    edges = []
    for _ in range(n_sims):
        noisy = base_rank + rng.normal(0, sigma, size=len(opp))
        opp_n = np.argsort(noisy)
        edges.append(season_edge(actual, opp_n, our, espn))
    e = np.array(edges)
    return e.mean(), e.std(ddof=1) / np.sqrt(n_sims)


# --------------------------------------------------------------------------
# Task 3: player bootstrap (main CI)
# --------------------------------------------------------------------------
def player_bootstrap(pool, n_boot=2000, seed=C.SEED, treat="our_value"):
    rng = np.random.default_rng(seed)
    n = len(pool)
    actual = pool["actual"].to_numpy(float)
    opp_key = pool["opp_key"].to_numpy(float)
    ourv = pool[treat].to_numpy(float)
    espnv = pool["espn_value"].to_numpy(float)
    edges = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        a = actual[idx]
        opp = np.argsort(opp_key[idx])
        our = np.argsort(-ourv[idx])
        espn = np.argsort(-espnv[idx])
        edges[b] = season_edge(a, opp, our, espn)
    lo, hi = np.quantile(edges, [0.025, 0.975])
    return {"mean": float(edges.mean()), "ci": (float(lo), float(hi)),
            "p_gt0": float((edges > 0).mean()), "draws": edges}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run(spec, base, table, seasons, n_boot=2000, treat="our_value",
        pools=None):
    rows, draws, adp_flags = [], {}, {}
    for s in seasons:
        pool, use_adp = (pools[s] if pools else build_pool(spec, base, table, s))
        adp_flags[s] = use_adp
        pe = point_edge(pool, treat)
        bs = player_bootstrap(pool, n_boot=n_boot, treat=treat)
        draws[s] = bs["draws"]
        rows.append({"season": s, "field": "ADP" if use_adp else "ESPN-val",
                     "edge_season": round(pe, 1),
                     "edge_week": round(pe / WEEKS, 2),
                     "ci_lo": round(bs["ci"][0], 1), "ci_hi": round(bs["ci"][1], 1),
                     "P(edge>0)": round(bs["p_gt0"], 3), "pool_n": len(pool)})
    summary = pd.DataFrame(rows)
    edges = np.array([r["edge_season"] for r in rows])
    n_pos = int((edges > 0).sum())
    sign_p = stats.binomtest(n_pos, len(edges), 0.5, alternative="greater").pvalue
    t = stats.ttest_1samp(edges, 0.0, alternative="greater")
    pooled = np.mean(np.vstack([draws[s] for s in seasons]), axis=0)
    return {"summary": summary, "edges": edges, "n_pos": n_pos,
            "sign_p": float(sign_p), "t_p": float(t.pvalue),
            "t_stat": float(t.statistic),
            "pooled_mean": float(pooled.mean()),
            "pooled_ci": (float(np.quantile(pooled, .025)),
                          float(np.quantile(pooled, .975))),
            "pooled_p": float((pooled > 0).mean()), "adp": adp_flags}


def _print(res, title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
    print(res["summary"].to_string(index=False))
    print(f"\n  season-level: {res['n_pos']}/{len(res['edges'])} positive  |  "
          f"sign test p={res['sign_p']:.3f}  |  paired-t p={res['t_p']:.3f} "
          f"(t={res['t_stat']:.2f})")
    print(f"  pooled edge {res['pooled_mean']:.1f} season pts "
          f"({res['pooled_mean']/WEEKS:.2f}/week)  95% CI "
          f"[{res['pooled_ci'][0]:.1f}, {res['pooled_ci'][1]:.1f}]  "
          f"P(edge>0)={res['pooled_p']:.3f}")


def main():
    spec = _frozen_spec()
    base = F.Base()
    table = F.build_modeling_table(base)
    seasons = WF_ESPN + (HOLDOUT_ESPN if C.FROZEN_SPEC_PATH.exists() else [])

    res = run(spec, base, table, seasons)
    _print(res, "PRIMARY: ours vs ESPN  (all 7 seasons)")
    res["summary"].to_csv(C.REPORTS / "draft_backtest_primary.csv", index=False)

    # Task 4: blend (0.55*ours + 0.45*ESPN) vs ESPN, holdout only (low power)
    if C.FROZEN_SPEC_PATH.exists():
        resb = run(spec, base, table, HOLDOUT_ESPN, treat="blend_value")
        _print(resb, "BLEND vs ESPN  (2024-2026 only -- LOW POWER, 3 seasons)")
        resb["summary"].to_csv(C.REPORTS / "draft_backtest_blend.csv", index=False)

    # secondary
    print("\n" + "=" * 78 + "\nSECONDARY (point est + sim/boot; no p-values)\n" + "=" * 78)
    print("  randomized opponents, noise in draft spots (mean edge, sim SE):")
    for s in seasons:
        pool, _ = build_pool(spec, base, table, s)
        parts = []
        for sg in (3, 6, 10):
            m, se = sim_spot_noise(pool, sg)
            parts.append(f"σ={sg}: {m:+.1f} (SE {se:.1f})")
        print(f"    {s}: " + "   ".join(parts))
    s = seasons[-1]
    pool, _ = build_pool(spec, base, table, s)
    print(f"\n  per-slot edges {s}: " + "  ".join(f"{v:+.0f}" for v in per_slot_edges(pool)))


if __name__ == "__main__":
    main()
