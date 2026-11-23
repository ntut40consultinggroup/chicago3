"""Black–Litterman：把主觀觀點和均衡混合。

觀點的形式：「某桶未來一年對指標超額 q，信心 c」。
- 觀點組合 P_k = e_桶 − w_B（買該桶、賣指標）。
- 觀點不確定性 Ω_kk = (P_k τΣ P_k') × (1 − c) ÷ c：c → 1 時 Ω → 0（完全相信），c → 0 時 Ω → ∞（不理會）。
輸出 α = μ_BL − Π：只有「超出均衡」的部分才算觀點帶來的超額。

注意：因為 Ω 與 τΣ 成正比，τ 在混合後的預期報酬 μ_BL 裡會完全消掉（He–Litterman 的性質），
所以在這個設定下 τ 不影響政策權重；真正控制觀點強弱的是「信心」。τ 保留為參數只是為了和文獻一致。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .equilibrium import implied_returns


@dataclass(frozen=True)
class View:
    bucket: str
    excess: float        # 對指標的一年超額報酬，例如 0.05
    confidence: float    # 0 < c < 1

    def __post_init__(self):
        if not 0 < self.confidence < 1:
            raise ValueError(f"信心必須在 0 和 1 之間（不含端點）：{self}")


@dataclass
class BLResult:
    Pi: pd.Series       # 均衡超額報酬
    mu: pd.Series       # 混合後的預期超額報酬
    alpha: pd.Series    # μ_BL − Π
    delta: float


def black_litterman(S: pd.DataFrame, wB: pd.Series, views: list[View], erp: float = 0.06,
                    tau: float = 0.05) -> BLResult:
    Pi, delta = implied_returns(S, wB, erp)
    if not views:
        return BLResult(Pi, Pi.copy(), pd.Series(0.0, S.index), delta)
    wb = wB.reindex(S.index).values
    n, k = len(S), len(views)
    P, Q, conf = np.zeros((k, n)), np.zeros(k), np.zeros(k)
    for i, v in enumerate(views):
        if v.bucket not in S.index:
            raise ValueError(f"觀點的桶不存在：{v.bucket}")
        P[i] = -wb
        P[i, S.index.get_loc(v.bucket)] += 1
        Q[i], conf[i] = v.excess, v.confidence
    TS = tau * S.values
    Om = np.diag(np.diag(P @ TS @ P.T) * (1 - conf) / conf)
    TSi, Omi = np.linalg.inv(TS), np.linalg.inv(Om)
    M = np.linalg.inv(TSi + P.T @ Omi @ P)
    mu = M @ (TSi @ Pi.values + P.T @ Omi @ Q)
    mu = pd.Series(mu, S.index)
    return BLResult(Pi, mu, mu - Pi, delta)
