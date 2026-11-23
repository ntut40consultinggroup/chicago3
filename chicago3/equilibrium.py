"""均衡報酬：CAPM 反向最佳化。

Π = δ Σ w_B，δ = 股票風險溢酬 ÷ 指標變異數。
意思是：如果市場是對的（指標就是最佳組合），各桶「應該」有多少超額報酬。
"""
import numpy as np
import pandas as pd


def risk_aversion(S: pd.DataFrame, wB: pd.Series, erp: float = 0.06) -> float:
    wb = wB.reindex(S.index).values
    return float(erp / (wb @ S.values @ wb))


def implied_returns(S: pd.DataFrame, wB: pd.Series, erp: float = 0.06) -> tuple[pd.Series, float]:
    """回傳 (Π, δ)。性質：Π'w_B = erp。"""
    delta = risk_aversion(S, wB, erp)
    return pd.Series(delta * S.values @ wB.reindex(S.index).values, S.index), delta


def decompose_expected_active(Pi: pd.Series, alpha: pd.Series, w: pd.Series, wB: pd.Series) -> dict:
    """預期主動報酬 = Beta 溢酬 Π'(w − w_B) ＋ 觀點 α'(w − w_B)。

    Beta 溢酬恆等於 erp × (β − 1)：只是把風險加在高 Beta 的桶上，不是判斷力。
    """
    a = (w - wB).reindex(Pi.index).values
    bp, va = float(Pi.values @ a), float(alpha.reindex(Pi.index).values @ a)
    return dict(expected_active=bp + va, beta_premium=bp, view_alpha=va)
