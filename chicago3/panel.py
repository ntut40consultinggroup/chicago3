"""資料面板：個股日報酬 → 桶日報酬、桶權重、重建指標。

市值以「最新市值」為錨，用市值變化率往回推（避免分割、股票股利讓「現在股數 × 過去股價」失真）。
桶報酬 = 桶內前一日收盤市值加權的個股報酬。
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Panel:
    R: pd.DataFrame        # 個股日報酬（上市前為 0）
    MC: pd.DataFrame       # 個股收盤市值（上市前為空值）
    bucket: pd.Series      # 代號 → 桶
    order: list[str]
    RB: pd.DataFrame       # 桶日報酬
    WB: pd.DataFrame       # 桶權重（當日開始前）
    rB: pd.Series          # 重建指標日報酬

    def weights_at(self, date) -> pd.Series:
        """date 收盤時的桶權重（指標 w_B）。"""
        m = self.MC.loc[:pd.Timestamp(date)].iloc[-1]
        s = m.groupby(self.bucket).sum()
        return (s / s.sum()).reindex(self.order)

    def window(self, start, end) -> pd.DataFrame:
        """估計期間的桶日報酬：不含 start 當天、含 end（與課程相同）。"""
        x = self.RB.loc[pd.Timestamp(start):pd.Timestamp(end)]
        return x[x.index > pd.Timestamp(start)]


def build_panel(rets: dict, rcaps: dict, mcap_anchor: pd.Series, bucket: pd.Series, order: list[str]) -> Panel:
    """rets、rcaps：代號 → 日報酬／市值變化率序列；mcap_anchor：代號 → 最後一天的收盤市值。"""
    R = pd.DataFrame(rets).sort_index()
    codes = list(R.columns)
    RC = pd.DataFrame(rcaps).sort_index().reindex(R.index)[codes]
    G = (1 + RC).cumprod()
    MC = G / G.iloc[-1] * mcap_anchor.reindex(codes)
    prev = MC.shift(1)
    prev.iloc[0] = MC.iloc[0] / (1 + RC.iloc[0])   # 第一天也要有前一日市值，否則當天報酬會被當成 0
    R = R.fillna(0.0)
    bk = bucket.reindex(codes)
    rB = (prev.div(prev.sum(axis=1), axis=0).fillna(0) * R).sum(axis=1)
    RB = pd.DataFrame({b: (prev[bk.index[bk == b]] * R[bk.index[bk == b]]).sum(axis=1)
                       / prev[bk.index[bk == b]].sum(axis=1) for b in order})
    WB = pd.DataFrame({b: prev[bk.index[bk == b]].sum(axis=1) for b in order})
    WB = WB.div(WB.sum(axis=1), axis=0)
    return Panel(R=R, MC=MC, bucket=bk, order=order, RB=RB, WB=WB, rB=rB)


def period_return(r: pd.Series, start, end) -> float:
    """(start, end] 的累積報酬。"""
    x = r.loc[pd.Timestamp(start):pd.Timestamp(end)]
    x = x[x.index > pd.Timestamp(start)]
    return float((1 + x).prod() - 1)
