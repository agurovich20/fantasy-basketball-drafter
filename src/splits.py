"""Walk-forward / holdout / production split helpers.

A split selects rows from the pooled modeling table by target season only.
Never random K-fold.  Training rows are the pooled rows whose target season
falls in the fold's training range; the eval rows are the single eval season.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


def rows_for_seasons(table: pd.DataFrame, seasons) -> pd.DataFrame:
    return table[table["season"].isin(list(seasons))]


def fold_frames(table: pd.DataFrame, bounds: tuple[int, int], eval_season: int):
    """(train_df, eval_df) for one walk-forward/holdout fold."""
    train = rows_for_seasons(table, C.train_seasons(bounds))
    ev = rows_for_seasons(table, [eval_season])
    return train, ev


def walk_forward(table: pd.DataFrame):
    """Yield (name, train_df, eval_df, eval_season) for each CV fold."""
    for bounds, ev in C.WALK_FORWARD_FOLDS:
        tr, evdf = fold_frames(table, bounds, ev)
        yield f"val_{ev}", tr, evdf, ev


def holdout(table: pd.DataFrame):
    """Yield (name, train_df, eval_df, eval_season) for each holdout season."""
    for bounds, ev in C.HOLDOUT_FOLDS:
        tr, evdf = fold_frames(table, bounds, ev)
        yield f"test_{ev}", tr, evdf, ev


def split_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Row counts for every train/eval set across both split tables."""
    rows = []
    for stage, folds in [("walk_forward", C.WALK_FORWARD_FOLDS),
                         ("holdout", C.HOLDOUT_FOLDS)]:
        for bounds, ev in folds:
            tr = rows_for_seasons(table, C.train_seasons(bounds))
            evdf = rows_for_seasons(table, [ev])
            rows.append({
                "stage": stage,
                "train_range": f"{bounds[0]}-{bounds[1]}",
                "eval_season": ev,
                "n_train": len(tr),
                "n_eval": len(evdf),
            })
    return pd.DataFrame(rows)
