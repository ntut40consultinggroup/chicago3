"""容量：每檔股票最多能放多少（占淨資產），加總到桶。

每檔取三者最小：
1. 單一個股上限 × 淨資產（法規 10%；特定個股可另設，例如台積電 25%）
2. 已發行股份上限 × 該股市值（10%）
3. 出清天數 × 每日參與率 × 日均成交值
"""
import pandas as pd


def position_capacity(nav: float, mcap: pd.Series, adv: pd.Series, single_cap: float = 0.10,
                      single_cap_overrides: dict | None = None, shares_cap: float = 0.10,
                      liq_days: float = 10, participation: float = 0.20) -> pd.DataFrame:
    idx = mcap.index
    t = pd.DataFrame(dict(nav_limit=single_cap * nav, shares_limit=shares_cap * mcap,
                          liquidity_limit=liq_days * participation * adv.reindex(idx)), index=idx)
    for code, c in (single_cap_overrides or {}).items():
        if code in t.index:
            t.loc[code, "nav_limit"] = c * nav
    t["capacity"] = t[["nav_limit", "shares_limit", "liquidity_limit"]].min(axis=1)
    t["binding"] = t[["nav_limit", "shares_limit", "liquidity_limit"]].idxmin(axis=1)
    return t


def bucket_capacity(pos: pd.DataFrame, bucket: pd.Series, nav: float, order: list[str]) -> pd.Series:
    """各桶容量（占淨資產比例）。"""
    return (pos.capacity.groupby(bucket.reindex(pos.index)).sum() / nav).reindex(order)
