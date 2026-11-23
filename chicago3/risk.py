"""風險：共變異估計、追蹤誤差、Beta。

所有函數吃 pandas 物件、回傳 pandas 物件或 float，不讀檔、不上網。
共變異一律年化（日報酬 × ann）。
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def cov_sample(R: pd.DataFrame, ann: int = 250) -> pd.DataFrame:
    return R.cov() * ann


def cov_ledoit_wolf(R: pd.DataFrame, ann: int = 250) -> pd.DataFrame:
    """Ledoit–Wolf 收縮（往縮放單位矩陣收縮），課程預設。"""
    return pd.DataFrame(LedoitWolf().fit(R.values).covariance_ * ann, R.columns, R.columns)


def cov_ewma(R: pd.DataFrame, halflife: float = 60, ann: int = 250) -> pd.DataFrame:
    """指數加權（RiskMetrics 型），近期資料權重較高；用來和 Ledoit–Wolf 對照風險估計是否過期。"""
    lam = 0.5 ** (1 / halflife)
    X = R.values - R.values.mean(axis=0)
    wts = lam ** np.arange(len(X))[::-1]
    wts /= wts.sum()
    return pd.DataFrame((X * wts[:, None]).T @ X * ann, R.columns, R.columns)


COV_METHODS = {"sample": cov_sample, "ledoit_wolf": cov_ledoit_wolf, "ewma": cov_ewma}


def estimate_cov(R: pd.DataFrame, method: str = "ledoit_wolf", ann: int = 250, **kw) -> pd.DataFrame:
    if method not in COV_METHODS:
        raise ValueError(f"不支援的共變異方法：{method}（可用 {list(COV_METHODS)}）")
    return COV_METHODS[method](R, ann=ann, **kw)


def te(active, S: pd.DataFrame) -> float:
    """追蹤誤差 = √(a'Σa)，a 是相對權重（對齊 S 的順序）。"""
    a = np.asarray(active, dtype=float)
    return float(np.sqrt(max(a @ S.values @ a, 0.0)))


def beta(w, wB, S: pd.DataFrame) -> float:
    """組合對指標的 Beta = w'Σw_B ÷ w_B'Σw_B。"""
    w, wB = np.asarray(w, float), np.asarray(wB, float)
    return float(w @ S.values @ wB / (wB @ S.values @ wB))


def vol(S: pd.DataFrame) -> pd.Series:
    return pd.Series(np.sqrt(np.diag(S.values)), S.index)


def corr(S: pd.DataFrame) -> pd.DataFrame:
    v = np.sqrt(np.diag(S.values))
    return pd.DataFrame(S.values / np.outer(v, v), S.index, S.columns)
