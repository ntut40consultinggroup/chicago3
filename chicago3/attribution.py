"""三層歸因：基金 − 指標 = ① 結構層（中性 − 指標）＋ ② 策略層（政策 − 中性）＋ ③ 執行層（其餘）。

回測方式（講義第 4 節）：每月第一個交易日把組合再平衡到目標；目標 = 當日指標桶權重 → 中性 → 加上政策傾斜。
結構層、策略層都用「重建組合之間的差」，兩邊帶著同樣的資料偏誤，大部分會互相抵銷。
"""
import pandas as pd

from .neutral import investable_neutral


def rebalanced_returns(R: pd.DataFrame, targets: pd.DataFrame, dates) -> pd.Series:
    """每月第一個交易日再平衡到 targets（索引：月份 Period），期間權重隨報酬漂移。回傳日報酬。"""
    out, w, month = [], None, None
    for t in dates:
        m = t.to_period("M")
        if m != month:
            w = targets.loc[m].reindex(R.columns).values.copy()
            month = m
        r = R.loc[t].values
        rp = float(w @ r)
        out.append(rp)
        w = w * (1 + r) / (1 + rp)
    return pd.Series(out, dates)


def monthly_targets(WB: pd.DataFrame, dates, upper: pd.Series, tilts: dict) -> dict:
    """回傳 {'benchmark','neutral', 各政策名稱: 目標表（月 × 桶）}。tilt 為政策 − 中性。"""
    months = sorted({d.to_period("M") for d in dates})
    first = {m: next(d for d in dates if d.to_period("M") == m) for m in months}
    tg = {"benchmark": {}, "neutral": {}, **{k: {} for k in tilts}}
    for m in months:
        wb = WB.loc[first[m]]
        wn = investable_neutral(wb, upper)
        tg["benchmark"][m], tg["neutral"][m] = wb, wn
        for k, a in tilts.items():
            wp = (wn + a.reindex(wn.index).fillna(0)).clip(lower=0)
            tg[k][m] = wp / wp.sum()
    return {k: pd.DataFrame(v).T for k, v in tg.items()}


def fund_market_returns(price: pd.DataFrame, div: pd.DataFrame) -> pd.Series:
    """基金市價含息日報酬：收盤 ÷（前收 − 當日配息）− 1。"""
    p = price.assign(date=pd.to_datetime(price.date)).set_index("date").close
    d = pd.Series(div.stock_and_cache_dividend.values, pd.to_datetime(div.date)) if len(div) else pd.Series(dtype=float)
    d = d.reindex(p.index).fillna(0.0)
    return (p / (p.shift(1) - d) - 1).iloc[1:]


def three_tier(r_fund: pd.Series, r_official: pd.Series, r_bench_rebuilt: pd.Series, r_neutral: pd.Series,
               r_policies: dict) -> dict:
    """期間總報酬的三層拆解。r_policies：{名稱: 日報酬}；每個政策各自算策略層與執行層。"""
    tot = lambda s: float((1 + s.fillna(0)).prod() - 1)
    F, T, Bb, N = tot(r_fund), tot(r_official), tot(r_bench_rebuilt), tot(r_neutral)
    active, structural = F - T, N - Bb
    out = dict(fund=F, benchmark=T, active=active, structural=structural, data_error=Bb - T, policies={})
    for k, r in r_policies.items():
        s = tot(r) - N
        out["policies"][k] = dict(strategic=s, execution=active - structural - s)
    return out


def monthly_three_tier(r_fund, r_official, r_bench_rebuilt, r_neutral, r_policy) -> pd.DataFrame:
    g = lambda s: (1 + s.fillna(0)).groupby(s.index.to_period("M")).prod() - 1
    m = pd.DataFrame(dict(benchmark=g(r_official), benchmark_rebuilt=g(r_bench_rebuilt), neutral=g(r_neutral),
                          policy=g(r_policy), fund=g(r_fund)))
    m["active"] = m.fund - m.benchmark
    m["structural"] = m.neutral - m.benchmark_rebuilt
    m["strategic"] = m.policy - m.neutral
    m["execution"] = m.active - m.structural - m.strategic
    m.index = m.index.astype(str)
    return m
