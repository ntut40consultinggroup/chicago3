"""再平衡區間：區間內不動，碰到邊界才再平衡。

規則（講義 3.5 節）：
1. 每桶往上、往下各自移動，直到策略 TE（對 N）增加 te_increment；移動的資金由其他非固定桶按比例供應。
2. 取「該距離」與「目標權重的一半」較小者（half_weight_cap），再夾在 clip 區間內。
3. no_up 裡的桶（已在法規上限，例如台積電 25%）不往上，上方寬度為 0。
"""
import numpy as np
import pandas as pd

from .risk import te


def rebalancing_bands(wP: pd.Series, wN: pd.Series, S: pd.DataFrame, te_increment: float = 0.01,
                      clip: tuple = (0.02, 0.10), half_weight_cap: bool = True, no_up: tuple = (),
                      fixed: tuple = (), search_max: float = 0.3, iters: int = 40) -> pd.DataFrame:
    """回傳 DataFrame：target、down（往下寬度）、up（往上寬度）、lower、upper。"""
    idx = wP.index
    Si = S.loc[idx, idx]
    base = te((wP - wN).values, Si)
    funders = [b for b in idx if b not in fixed]
    rows = {}
    for b in idx:
        widths = {}
        for sgn, side in ((1, "up"), (-1, "down")):
            if sgn == 1 and b in no_up:
                widths[side] = 0.0
                continue
            rest = [x for x in funders if x != b]
            lo, hi = 0.0, search_max
            for _ in range(iters):
                d = (lo + hi) / 2
                w = wP.copy()
                w[b] += sgn * d
                w[rest] -= sgn * d * wP[rest] / wP[rest].sum()
                if (w < 0).any() or te((w - wN).values, Si) - base > te_increment:
                    hi = d
                else:
                    lo = d
            dist = min(lo, max(clip[0], 0.5 * wP[b])) if half_weight_cap else lo
            widths[side] = float(np.clip(dist, *clip))
        rows[b] = dict(target=wP[b], down=widths["down"], up=widths["up"],
                       lower=wP[b] - widths["down"], upper=wP[b] + widths["up"])
    return pd.DataFrame(rows).T[["target", "down", "up", "lower", "upper"]]


def band_status(w: pd.Series, bands: pd.DataFrame) -> pd.Series:
    """每桶狀態：'' 在區間內、'above' 高於上限、'below' 低於下限。"""
    w = w.reindex(bands.index).fillna(0.0)
    return pd.Series(np.where(w > bands.upper, "above", np.where(w < bands.lower, "below", "")), bands.index)
