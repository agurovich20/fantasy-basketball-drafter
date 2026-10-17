"""Baselines, model families, target specs, and metrics.

Everything here is fold-local: estimators and the Marcel constants are fit on a
fold's training rows only.  Nothing reads the eval/holdout rows at fit time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import lightgbm as lgb
import xgboost as xgb

from . import config as C
from . import features as F

FAMILIES = ["ridge", "rf", "xgb", "lgbm"]
SPECS = ["a", "b", "c"]           # direct FPPG / residual-over-Marcel / mpg*fppm
FEATURE_SETS = ["hist", "hist_ctx"]


HALF_LIVES = [2.0, 4.0, 8.0, np.inf]


def sample_configs(family: str, n: int, rng: np.random.Generator) -> list[dict]:
    """n random hyperparameter configs for a family (seeded)."""
    out = []
    for _ in range(n):
        hl = float(rng.choice(HALF_LIVES))
        if family == "ridge":
            cfg = {"alpha": float(10 ** rng.uniform(-3, 3))}
        elif family == "rf":
            cfg = {
                "n_estimators": 500,
                "max_depth": rng.choice([6, 8, 12, None]),
                "min_samples_leaf": int(rng.integers(5, 51)),
                "max_features": float(rng.uniform(0.2, 0.7)),
            }
        elif family == "xgb":
            cfg = {
                "n_estimators": int(rng.integers(200, 1501)),
                "learning_rate": float(10 ** rng.uniform(-2, -1)),
                "max_depth": int(rng.integers(2, 7)),
                "min_child_weight": float(rng.uniform(1, 30)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.4, 1.0)),
                "reg_lambda": float(rng.uniform(0, 10)),
            }
        elif family == "lgbm":
            cfg = {
                "n_estimators": int(rng.integers(200, 1501)),
                "learning_rate": float(10 ** rng.uniform(-2, -1)),
                "num_leaves": int(rng.integers(7, 64)),
                "min_child_weight": float(rng.uniform(1, 30)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.4, 1.0)),
                "reg_lambda": float(rng.uniform(0, 10)),
            }
        else:
            raise ValueError(family)
        cfg["half_life"] = hl
        out.append(cfg)
    return out


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def last_season_pred(df: pd.DataFrame) -> np.ndarray:
    """Last season's FPPG, else the most recent season actually played."""
    out = df["L1_fppg"].copy()
    out = out.fillna(df["L2_fppg"]).fillna(df["L3_fppg"])
    return out.to_numpy(float)


@dataclass
class Marcel:
    """Games-weighted 3-season average, regressed to the mean, age-adjusted.

    Parameters (lag weights, regression constant K, age slope and peak) are fit
    on the training rows by minimising games-weighted MAE.
    """
    w: np.ndarray = field(default_factory=lambda: np.array([5.0, 4.0, 3.0]))
    K: float = 150.0          # regression strength, in weighted-games units
    gamma: float = 0.0        # fractional change per year away from peak
    peak_age: float = 27.0
    mu: float = 0.0           # population (games-weighted) mean FPPG

    def _raw(self, df: pd.DataFrame, w, K, gamma, peak):
        fppg = np.vstack([df[f"L{l}_fppg"].to_numpy(float) for l in (1, 2, 3)])
        games = np.vstack([df[f"L{l}_games"].to_numpy(float) for l in (1, 2, 3)])
        present = ~np.isnan(fppg)
        fppg = np.where(present, fppg, 0.0)
        g = np.where(present & ~np.isnan(games), np.nan_to_num(games), 0.0)
        wg = (w[:, None] * g)
        num = (wg * fppg).sum(axis=0)
        den = wg.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            base = np.where(den > 0, num / np.where(den > 0, den, 1.0), self.mu)
        proj0 = (den * base + K * self.mu) / (den + K)
        age = df["age"].to_numpy(float)
        age = np.where(np.isnan(age), peak, age)
        proj = proj0 * (1.0 + gamma * (peak - age))
        return proj

    def fit(self, df: pd.DataFrame) -> "Marcel":
        y = df["target_fppg"].to_numpy(float)
        wts = df["target_games"].to_numpy(float)
        self.mu = float(np.average(y, weights=wts))

        def clamp(theta):
            w2, w3, K, gamma, peak = theta
            return (np.clip(w2, 0, 1.5), np.clip(w3, 0, 1.5),
                    np.clip(K, 1, 2000), np.clip(gamma, -0.1, 0.1),
                    np.clip(peak, 22, 32))

        def obj(theta):
            w2, w3, K, gamma, peak = clamp(theta)
            pred = self._raw(df, np.array([1.0, w2, w3]), K, gamma, peak)
            return float(np.average(np.abs(pred - y), weights=wts))

        # a few restarts keep Nelder-Mead off degenerate corners
        starts = [np.array([0.8, 0.6, 150.0, 0.0, 27.0]),
                  np.array([0.7, 0.5, 400.0, 0.02, 27.0]),
                  np.array([1.0, 1.0, 50.0, 0.0, 26.0]),
                  np.array([0.6, 0.4, 800.0, 0.03, 28.0])]
        best, best_x = np.inf, starts[0]
        for x0 in starts:
            res = minimize(obj, x0, method="Nelder-Mead",
                           options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
            if res.fun < best:
                best, best_x = res.fun, res.x
        w2, w3, K, gamma, peak = clamp(best_x)
        self.w = np.array([1.0, w2, w3])
        self.K, self.gamma, self.peak_age = K, gamma, peak
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._raw(df, self.w, self.K, self.gamma, self.peak_age)


# --------------------------------------------------------------------------
# Feature matrices & recency weights
# --------------------------------------------------------------------------
def build_X(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    cols = F.feature_columns(df, feature_set)
    return df[cols].astype("float64")


def recency_weights(df: pd.DataFrame, last_train_season: int,
                    half_life: float) -> np.ndarray:
    """0.5^((last_train_season - season)/half_life); half_life=inf -> all 1."""
    g = df["target_games"].to_numpy(float)
    if not np.isfinite(half_life):
        return g
    age = last_train_season - df["season"].to_numpy(float)
    return g * np.power(0.5, age / half_life)


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def make_estimator(family: str, params: dict):
    p = dict(params)
    p.pop("half_life", None)
    if family == "ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
            ("est", Ridge(alpha=p.get("alpha", 1.0), random_state=C.SEED)),
        ])
    if family == "rf":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("est", RandomForestRegressor(
                n_estimators=p.get("n_estimators", 500),
                max_depth=p.get("max_depth", None),
                min_samples_leaf=p.get("min_samples_leaf", 10),
                max_features=p.get("max_features", 0.5),
                n_jobs=-1, random_state=C.SEED)),
        ])
    if family == "xgb":
        return xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 500),
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 4),
            min_child_weight=p.get("min_child_weight", 5),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_lambda=p.get("reg_lambda", 1.0),
            tree_method="hist", n_jobs=-1, random_state=C.SEED)
    if family == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 500),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 31),
            min_child_weight=p.get("min_child_weight", 5),
            subsample=p.get("subsample", 0.8),
            subsample_freq=1,
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_lambda=p.get("reg_lambda", 1.0),
            max_depth=-1, n_jobs=-1, random_state=C.SEED, verbose=-1)
    raise ValueError(family)


def _fit_one(family, params, X, y, w):
    est = make_estimator(family, params)
    est.fit(X, y, **{_sw_key(family, est): w})
    return est


def _sw_key(family, est):
    if family == "ridge":
        return "est__sample_weight"
    if family == "rf":
        return "est__sample_weight"
    return "sample_weight"


# --------------------------------------------------------------------------
# Spec-aware fit / predict
# --------------------------------------------------------------------------
def fit_predict(family: str, spec: str, feature_set: str, params: dict,
                train: pd.DataFrame, evl: pd.DataFrame,
                last_train_season: int, marcel: Marcel):
    """Return predicted FPPG on `evl` for one candidate, fit on `train`."""
    Xtr = build_X(train, feature_set)
    Xev = build_X(evl, feature_set)
    w = recency_weights(train, last_train_season, params.get("half_life", np.inf))

    if spec == "a":
        y = train["target_fppg"].to_numpy(float)
        est = _fit_one(family, params, Xtr, y, w)
        return est.predict(Xev)

    if spec == "b":
        resid = train["target_fppg"].to_numpy(float) - marcel.predict(train)
        est = _fit_one(family, params, Xtr, resid, w)
        return marcel.predict(evl) + est.predict(Xev)

    if spec == "c":
        ymin = train["target_mpg"].to_numpy(float)
        yrate = train["target_fppm"].to_numpy(float)
        est_min = _fit_one(family, params, Xtr, ymin, w)
        est_rate = _fit_one(family, params, Xtr, yrate, w)
        return est_min.predict(Xev) * est_rate.predict(Xev)

    raise ValueError(spec)


def predict_with_spec(spec: dict, train: pd.DataFrame, evl: pd.DataFrame,
                      last_train_season: int) -> np.ndarray:
    """Fit a saved spec (family/spec/feature_set/params) on `train`, predict evl.

    Marcel (needed by spec 'b') is fit on the same training rows.
    """
    marcel = Marcel().fit(train)
    return fit_predict(spec["family"], spec["spec"], spec["feature_set"],
                       spec["params"], train, evl, last_train_season, marcel)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def w_mae(y, p, w):
    return float(np.average(np.abs(p - y), weights=w))


def w_rmse(y, p, w):
    return float(np.sqrt(np.average((p - y) ** 2, weights=w)))


def w_bias(y, p, w):
    return float(np.average(p - y, weights=w))


def draft_pool_metrics(evl: pd.DataFrame, pred: np.ndarray,
                       pool_mask: np.ndarray) -> dict:
    """Spearman, MAE and decile calibration inside the Marcel top-200 pool."""
    y = evl["target_fppg"].to_numpy(float)[pool_mask]
    p = np.asarray(pred)[pool_mask]
    rho = float(spearmanr(p, y).statistic) if len(p) > 2 else np.nan
    mae = float(np.mean(np.abs(p - y)))
    order = np.argsort(-p)
    deciles = np.array_split(order, 10)
    calib = [(float(p[d].mean()), float(y[d].mean())) for d in deciles if len(d)]
    return {"spearman": rho, "mae": mae, "n": int(pool_mask.sum()),
            "calibration": calib}


def season_metrics(evl: pd.DataFrame, pred: np.ndarray,
                   pool_mask: np.ndarray) -> dict:
    y = evl["target_fppg"].to_numpy(float)
    w = evl["target_games"].to_numpy(float)
    m = {"mae": w_mae(y, pred, w), "rmse": w_rmse(y, pred, w),
         "bias": w_bias(y, pred, w)}
    m.update({f"pool_{k}": v for k, v in
              draft_pool_metrics(evl, pred, pool_mask).items()})
    return m
