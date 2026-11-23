"""監控：基金在不在區間內、回到政策需要多少換手。"""
import cvxpy as cp
import numpy as np
import pandas as pd


def one_way_turnover(w_from: pd.Series, w_to: pd.Series) -> float:
    """單邊換手 = ½ Σ|w_to − w_from|。"""
    idx = w_from.index.union(w_to.index)
    return float(0.5 * (w_to.reindex(idx).fillna(0) - w_from.reindex(idx).fillna(0)).abs().sum())


def nearest_in_band(w: pd.Series, bands: pd.DataFrame) -> pd.Series:
    """在區間內、加總為 1，且離現況最近（平方距離最小）的權重：最省換手的「回到區間」目標。"""
    idx = bands.index
    x = cp.Variable(len(idx))
    cur = w.reindex(idx).fillna(0).values
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - cur)),
                      [cp.sum(x) == 1, x >= bands.lower.values, x <= bands.upper.values])
    prob.solve()
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"區間加總不可行（{prob.status}）：下限加總 {bands.lower.sum():.3f}、上限加總 {bands.upper.sum():.3f}")
    return pd.Series(np.clip(x.value, 0, None), idx)


def band_report(w: pd.Series, policy: pd.Series, bands: pd.DataFrame) -> dict:
    """區間檢查摘要：各桶狀態、到政策目標與到區間邊界的換手。"""
    from .bands import band_status
    edge = nearest_in_band(w, bands)
    return dict(status=band_status(w, bands), n_out=int((band_status(w, bands) != "").sum()),
                turnover_to_policy=one_way_turnover(w, policy), turnover_to_band=one_way_turnover(w, edge),
                band_target=edge)
