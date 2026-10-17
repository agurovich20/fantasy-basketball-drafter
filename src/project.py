"""Production step: refit the frozen spec on 2011-2026 and project 2027.

10th/90th-percentile bands come from bootstrapping the frozen spec's holdout
errors, pooled within projection-level bins so that high scorers get wider
bands than low scorers.

Run after the holdout stage (needs reports/holdout_residuals.csv):
    python -m src.project
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import models as M
from . import splits as S


def _residual_bands(proj: np.ndarray, resid_df: pd.DataFrame,
                    n_bins: int = 5, n_boot: int = 4000,
                    seed: int = C.SEED):
    """Bootstrap holdout residuals within projection bins -> (p10, p90)."""
    rng = np.random.default_rng(seed)
    hp = resid_df["pred"].to_numpy(float)
    hr = resid_df["resid"].to_numpy(float)
    edges = np.quantile(hp, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    hbin = np.clip(np.digitize(hp, edges[1:-1]), 0, n_bins - 1)
    pools = {b: hr[hbin == b] for b in range(n_bins)}
    allr = hr

    pbin = np.clip(np.digitize(proj, edges[1:-1]), 0, n_bins - 1)
    p10 = np.empty_like(proj, dtype=float)
    p90 = np.empty_like(proj, dtype=float)
    for i, (p, b) in enumerate(zip(proj, pbin)):
        pool = pools.get(b, allr)
        if len(pool) < 20:
            pool = allr
        draws = rng.choice(pool, size=n_boot, replace=True)
        samp = p + draws
        p10[i] = np.quantile(samp, 0.10)
        p90[i] = np.quantile(samp, 0.90)
    return p10, p90


def project_2027() -> pd.DataFrame:
    if not C.FROZEN_SPEC_PATH.exists():
        raise SystemExit("frozen_spec.json missing -- run ⏸2/⏸3 first.")
    resid_path = C.REPORTS / "holdout_residuals.csv"
    if not resid_path.exists():
        raise SystemExit("holdout_residuals.csv missing -- run the holdout "
                         "stage (⏸3) before projecting.")
    frozen = json.loads(C.FROZEN_SPEC_PATH.read_text())
    resid_df = pd.read_csv(resid_path)

    base = F.Base()
    table = F.build_modeling_table(base)
    prod = F.build_production_table(base)

    # refit frozen spec on all historical target seasons 2011-2026
    train = S.rows_for_seasons(table, C.train_seasons(C.PRODUCTION_FOLD[0]))
    last_train_season = C.PRODUCTION_FOLD[0][1]  # 2026
    xgb_proj = np.clip(M.predict_with_spec(frozen, train, prod, last_train_season),
                       0, None)

    # production predictor = blend 0.55*model + 0.45*ESPN (fallback to model
    # alone where ESPN has no 2027 projection)
    from . import espn as E
    w = frozen.get("blend", {}).get("weight_ours", 0.55)
    esp = E.projections(2027).set_index("athlete_id")["espn_proj_fppg"].to_dict()
    espn_vec = prod["athlete_id"].map(esp).to_numpy(float)
    has_espn = ~np.isnan(espn_vec)
    proj = xgb_proj.copy()
    proj[has_espn] = w * xgb_proj[has_espn] + (1 - w) * espn_vec[has_espn]

    p10, p90 = _residual_bands(proj, resid_df)
    p10 = np.clip(p10, 0, None)

    out = pd.DataFrame({
        "athlete_id": prod["athlete_id"].to_numpy(),
        "name": prod["name"].to_numpy(),
        "team": prod["team"].to_numpy(),
        "age": prod["age"].round(1).to_numpy(),
        "experience": prod["experience"].to_numpy(),
        "changed_team": prod["changed_team"].to_numpy(),
        "proj_fppg": np.round(proj, 2),
        "model_fppg": np.round(xgb_proj, 2),
        "espn_fppg": np.round(espn_vec, 2),
        "p10_fppg": np.round(p10, 2),
        "p90_fppg": np.round(p90, 2),
    }).sort_values("proj_fppg", ascending=False).reset_index(drop=True)

    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.OUTPUT / "projections_2027.csv", index=False)
    print(f"wrote {C.OUTPUT/'projections_2027.csv'}  ({len(out)} players)")
    print(out.head(25).to_string(index=False))
    return out


if __name__ == "__main__":
    project_2027()
