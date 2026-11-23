"""穩健性：觀點敏感度、區塊 bootstrap 重抽樣、歷史壓力重播。

敏感度與重抽樣都吃一個 solve(S, views) → 政策權重 的函數，其他設定（中性、預算、上限）由呼叫端綁好，
這樣這裡只負責「換輸入、重解、整理結果」。
"""
from typing import Callable

import numpy as np
import pandas as pd

from .views import View

Solve = Callable[[pd.DataFrame, list[View]], pd.Series]


def view_sensitivity(S: pd.DataFrame, solve: Solve, wN: pd.Series, bucket: str, excess_grid, conf_grid,
                     te_fn: Callable) -> pd.DataFrame:
    """觀點大小 × 信心 的政策權重表。超額 0 表示沒有觀點（信心不影響）。te_fn(a) 回傳策略 TE。"""
    rows = []
    for q in excess_grid:
        for c in conf_grid:
            vs = [View(bucket, q, c)] if q else []
            wP = solve(S, vs)
            rows.append(dict(excess=q, confidence=c, te_strategic=te_fn((wP - wN).values), **wP.to_dict()))
    return pd.DataFrame(rows)


def block_bootstrap(R: pd.DataFrame, cov_fn: Callable, solve: Solve, views: list[View], n: int = 300,
                    block: int = 20, seed: int = 20261122) -> pd.DataFrame:
    """重抽估計期間（區塊保留短期自我相關），每次重估共變異、重解政策。回傳每次的政策權重。"""
    if len(R) <= block:
        raise ValueError(f"估計期間只有 {len(R)} 天，不夠做 {block} 日區塊重抽樣")
    rng = np.random.default_rng(seed)
    nb = len(R) // block
    draws = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, len(R) - block, nb)])
        draws.append(solve(cov_fn(R.iloc[idx]), views))
    return pd.DataFrame(draws)


def bootstrap_summary(draws: pd.DataFrame, wP: pd.Series, bands: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(dict(policy=wP, p5=draws.quantile(0.05), median=draws.median(), p95=draws.quantile(0.95),
                             band_lower=bands.lower, band_upper=bands.upper))


def stress_replay(RB: pd.DataFrame, start, end, portfolios: dict, benchmark: str = "benchmark") -> tuple[pd.DataFrame, pd.Series]:
    """情境重播：把「現在的權重」套到歷史期間 (start, end] 的各桶累積報酬上（期間內不再平衡）。

    相對報酬一律和 portfolios[benchmark] 用同樣方式重播的結果比，兩邊期間與方法一致。
    （課程原始程式的起日 2026-10-03 是週六，桶報酬誤刪了 10/05、指標卻沒刪，造成相對數字差約 0.1 個百分點。）
    回傳（各組合期間報酬與相對指標、各桶期間報酬）。
    """
    x = RB.loc[pd.Timestamp(start):pd.Timestamp(end)]
    rS = (1 + x[x.index > pd.Timestamp(start)]).prod() - 1
    ret = {k: float(w.reindex(rS.index).fillna(0) @ rS) for k, w in portfolios.items()}
    base = ret.get(benchmark, 0.0)
    t = pd.DataFrame([dict(portfolio=k, return_=v, vs_benchmark=v - base) for k, v in ret.items()])
    return t, rS
