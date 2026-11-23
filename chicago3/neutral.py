"""可投資中性 N：指標權重，但每桶不得超過上限（法規、容量）；超出的部分按比例分給還有空間的桶。"""
import pandas as pd


def investable_neutral(wB: pd.Series, upper: pd.Series, max_iter: int = 50) -> pd.Series:
    """upper：每桶上限（缺的桶視為 1）。例如台積電桶上限 = min(容量, 25%)。

    性質：加總 = 1；沒有桶超過上限；沒有被壓的桶之間，彼此比例不變。
    """
    lim = upper.reindex(wB.index).fillna(1.0)
    if lim.sum() < 1 - 1e-12:
        raise ValueError(f"各桶上限加總 {lim.sum():.4f} < 1，不可能找到可投資中性")
    w = wB.astype(float).copy()
    for _ in range(max_iter):
        over = w > lim + 1e-12
        if not over.any():
            return w
        excess = (w[over] - lim[over]).sum()
        w[over] = lim[over]
        room = ~over & (w < lim)
        w[room] += excess * w[room] / w[room].sum()
    raise RuntimeError("可投資中性沒有收斂")
