"""SAA 補充課：股票基金的策略性資產配置（政策組合）。

用法：python3 -I stage8_saa.py <saa資料夾> <每日資料夾> <加權報酬指數.json> <00981A價格.json> <00981A配息.json> <持股.csv> <輸出前綴>

三個組合：
  B 指標    ：上市市值加權（用 11/20 市值前 400 大重建，與官方加權報酬指數勾稽）
  N 可投資中性：指標，但台積電上限 25%（法規），超出部分按比例分給其他桶 → 「結構性」偏離，不是經理人的選擇
  P 政策組合：N ＋ 投資委員會每年核准的策略性傾斜（本課的 SAA）
實際基金 = P ＋ 戰術偏離 ＋ 選股 ＋ 費用。

方法：
- 個股日總報酬：一般日 = 收盤 ÷ 前收 − 1；除權息日 = 收盤 ÷ 除權息參考價 − 1（FinMind 除權息結果）；
  非除權息日漲跌逾 10.5%（可能是減資、分割）改用漲跌欄的參考價計算。
- 市值 = 11/20 已發行股數 × 當日收盤（期間股數變動忽略，誤差由勾稽殘差呈現）。
- 桶報酬：桶內市值加權；共變異：前一年日報酬 × 250，Ledoit–Wolf 收縮。
- 均衡報酬 Π = δ Σ w_B，δ = 假設股票風險溢酬 6% ÷ 指標變異數；觀點用 Black–Litterman（τ = 0.05）。
- 政策最佳化：max α'(w − w_N) − (λ/2) TE²，限制：權重 ≥ 0、台積電固定在 N、單桶傾斜 ≤ 15 個百分點、
  策略 TE（對 N）≤ 預算、各桶 ≤ 容量。
- 區間：每桶往上、往下各自移動到「策略 TE 增加 1 個百分點」為止（資金由其他非台積電桶按比例供應），
  再夾在 2–10 個百分點之間。
"""
import json
import sys
import warnings
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_health_check import holdings, load  # noqa: E402

warnings.filterwarnings("ignore")
ANN, TAU, ERP = 250, 0.05, 0.06
NAV = 2752e8                      # 00981A 規模約 2,752 億元（2026-11-13）
TE_TOTAL = 0.15                   # 階段 5 第 4 課：總追蹤誤差預算 B1
TILT_MAX = 0.15
IR = 0.5
BANDS = (0.02, 0.10)
LIQ_DAYS = 10
UNIVERSE = 400                   # 11/20 市值前 400 大上市股（占加權指數 97.4%）；取 400 而非 200 是為了降低存活者偏誤
BUCKETS = ["台積電", "其他半導體", "AI硬體零組件", "電腦與伺服器", "其他電子", "金融保險", "傳產與其他"]
AI_OVERRIDE = {"3017", "2059", "6805", "8210", "3324", "3653"}   # 奇鋐、川湖、富世達、勤誠、雙鴻、健策：散熱與機構件歸 AI 硬體
OTHER_ELEC = {"通信網路", "光電", "其他電子", "電子通路", "資訊服務", "數位雲端"}
P25 = ("2024-11-20", "2025-11-20")   # 2025 政策的估計期間
P26 = ("2025-11-20", "2026-11-20")   # 回溯期間、也是 2027 政策的估計期間
SHOCK = ("2026-10-03", "2026-11-20")  # AI 衝擊：10/5 高點前一個交易日收盤 → 11/20


def bucket_of(code, ind):
    if code == "2330":
        return "台積電"
    if code in AI_OVERRIDE or ind == "電子零組件":
        return "AI硬體零組件"
    if ind == "半導體":
        return "其他半導體"
    if ind == "電腦及週邊設備":
        return "電腦與伺服器"
    if ind in OTHER_ELEC:
        return "其他電子"
    if ind == "金融保險":
        return "金融保險"
    return "傳產與其他"


def stock_returns(sdir, code):
    p = pd.DataFrame(json.loads((sdir / f"TaiwanStockPrice_{code}.json").read_text())["data"])
    if p.empty:
        return None, None, None
    p["date"] = pd.to_datetime(p.date)
    p = p.set_index("date")
    p = p[p.close > 0]
    d = pd.DataFrame(json.loads((sdir / f"TaiwanStockDividendResult_{code}.json").read_text())["data"])
    r = p.close / p.close.shift(1) - 1
    rcap = r.copy()                       # 市值變化率：現金股利會讓市值下降，股票股利、分割不會
    flags = 0
    if not d.empty:
        d["date"] = pd.to_datetime(d.date)
        for _, x in d.iterrows():
            if x.date in r.index and x.after_price > 0:
                r.loc[x.date] = p.close.loc[x.date] / x.after_price - 1
                cash = x.stock_and_cache_dividend if x.stock_or_cache_dividend == "息" else 0.0
                rcap.loc[x.date] = p.close.loc[x.date] / x.after_price * (1 - cash / x.before_price) - 1
        exd = set(d.date)
    else:
        exd = set()
    jump = (r.abs() > 0.105) & ~r.index.isin(exd)
    if jump.any():                        # 減資、分割等：改用漲跌欄的參考價
        ref = p.close - p.spread
        r[jump] = (p.close / ref - 1)[jump]
        rcap[jump] = r[jump]
        flags = int(jump.sum())
    return r.iloc[1:], rcap.iloc[1:], (flags, p.Trading_money)


def lw_cov(R):
    return pd.DataFrame(LedoitWolf().fit(R.values).covariance_ * ANN, R.columns, R.columns)


def te(a, S):
    return float(np.sqrt(max(a @ S.values @ a, 0)))


def bl_alpha(S, wB, views, conf):
    """回傳 Π（均衡）與 α = μ_BL − Π。views：[(桶, 對指標超額 q)]。"""
    var_m = float(wB @ S.values @ wB)
    delta = ERP / var_m
    Pi = delta * S.values @ wB
    if not views:
        return pd.Series(Pi, S.index), pd.Series(0.0, S.index), delta
    n = len(S)
    P = np.zeros((len(views), n))
    Q = np.zeros(len(views))
    for k, (b, q) in enumerate(views):
        P[k] = -wB
        P[k, S.index.get_loc(b)] += 1
        Q[k] = q
    TS = TAU * S.values
    Om = np.diag(np.diag(P @ TS @ P.T)) * (1 - conf) / conf
    M = np.linalg.inv(np.linalg.inv(TS) + P.T @ np.linalg.inv(Om) @ P)
    mu = M @ (np.linalg.inv(TS) @ Pi + P.T @ np.linalg.inv(Om) @ Q)
    return pd.Series(Pi, S.index), pd.Series(mu - Pi, S.index), delta


def neutral(wB, cap=None):
    """可投資中性：台積電 ≤ 25%；若給容量，各桶也不得超過容量。超出部分按比例分給還有空間的桶。"""
    lim = pd.Series(1.0, wB.index) if cap is None else cap.reindex(wB.index).fillna(1.0).copy()
    lim["台積電"] = min(lim["台積電"], 0.25)
    w = wB.copy()
    for _ in range(20):
        over = w > lim + 1e-12
        if not over.any():
            break
        ex = (w[over] - lim[over]).sum()
        w[over] = lim[over]
        room = ~over & (w < lim)
        w[room] += ex * w[room] / w[room].sum()
    return w


def solve_policy(alpha, S, wN, budget, cap):
    n = len(wN)
    w = cp.Variable(n)
    a = w - wN.values
    L = np.linalg.cholesky(S.values + 1e-10 * np.eye(n))
    lam = IR / budget
    k = wN.index.get_loc("台積電")
    cons = [cp.sum(w) == 1, w >= 0, cp.abs(a) <= TILT_MAX, a[k] == 0,
            cp.norm(L.T @ a) <= budget, w <= cap.values]
    cp.Problem(cp.Maximize(alpha.values @ a - lam / 2 * cp.sum_squares(L.T @ a)), cons).solve()
    return pd.Series(np.clip(w.value, 0, None), wN.index)


def bands(wP, wN, S):
    """每桶往上、往下各自移動，直到策略 TE 增加 1 個百分點（資金由其他非台積電桶按比例供應）；
    再取 min(該距離, 目標權重的一半)，夾在 2–10 個百分點。台積電只往下算（法規上限 25%）。"""
    base = te((wP - wN).values, S)
    oth = [b for b in wP.index if b != "台積電"]
    out = {}
    for b in wP.index:
        res = []
        for sgn in (1, -1):
            if b == "台積電" and sgn == 1:
                res.append(0.0)
                continue
            lo, hi = 0.0, 0.3
            rest = [x for x in oth if x != b]
            for _ in range(40):
                d = (lo + hi) / 2
                w = wP.copy()
                w[b] += sgn * d
                w[rest] -= sgn * d * wP[rest] / wP[rest].sum()
                if (w < 0).any() or te((w - wN).values, S) - base > 0.01:
                    hi = d
                else:
                    lo = d
            res.append(float(np.clip(min(lo, max(BANDS[0], 0.5 * wP[b])), *BANDS)))
        out[b] = res
    return out


def port_path(R, targets, dates):
    """每月第一個交易日再平衡到 targets（DataFrame：月 → 桶權重），期間權重隨報酬漂移。回傳日報酬。"""
    out = []
    w = None
    month = None
    for t in dates:
        m = t.to_period("M")
        if m != month:
            w = targets.loc[m].values.copy()
            month = m
        r = R.loc[t].values
        rp = float(w @ r)
        out.append(rp)
        w = w * (1 + r) / (1 + rp)
    return pd.Series(out, dates)


def main(sdir, daily, tri_json, fpx, fdiv, hpath, prefix):
    sdir = Path(sdir)
    q, _ = load(daily)
    lst = q[q.市場 == "上市"].sort_values("市值", ascending=False)
    names = {r["Code"]: r["Name"] for r in json.loads((Path(daily) / "exchangeReport_STOCK_DAY_ALL.json").read_text())}
    codes = [c for c in lst.index[:UNIVERSE] if (sdir / f"TaiwanStockPrice_{c}.json").exists()]
    rets, rcaps, adv, flags = {}, {}, {}, {}
    for c in codes:
        r, rc, (fl, money) = stock_returns(sdir, c)
        if r is None:
            continue
        rets[c], rcaps[c], flags[c] = r, rc, fl
        adv[c] = float(money.iloc[-60:].mean())
    Rs = pd.DataFrame(rets).sort_index()
    codes = list(Rs.columns)
    bk = pd.Series({c: bucket_of(c, lst.loc[c, "產業"]) for c in codes})
    # 市值：以 11/20 市值為錨，用「市值變化率」往回推（避免分割、股票股利讓用現在股數算的過去市值失真）
    G = (1 + pd.DataFrame(rcaps).sort_index().reindex(Rs.index)).cumprod()
    MC = G / G.iloc[-1] * lst.loc[codes, "市值"]   # 收盤市值；上市前為空值
    MCprev = MC.shift(1)                          # 當日開始前（前一日收盤）的市值
    Rs = Rs.fillna(0.0)
    # 指標日報酬（重建）與桶報酬
    wprev = MCprev.div(MCprev.sum(axis=1), axis=0).fillna(0)
    rB = (wprev * Rs).sum(axis=1)
    RB = pd.DataFrame({b: (MCprev[bk.index[bk == b]] * Rs[bk.index[bk == b]]).sum(axis=1) / MCprev[bk.index[bk == b]].sum(axis=1)
                       for b in BUCKETS})
    WB = pd.DataFrame({b: MCprev[bk.index[bk == b]].sum(axis=1) for b in BUCKETS})
    WB = WB.div(WB.sum(axis=1), axis=0)
    tri = pd.DataFrame(json.loads(Path(tri_json).read_text())["data"])
    tri = pd.Series(tri.price.values, pd.to_datetime(tri.date))
    rtri = tri.pct_change().reindex(Rs.index)

    def per(s, a, b):
        x = s.loc[a:b]
        x = x[x.index > pd.Timestamp(a)]
        return float((1 + x).prod() - 1)

    recon = pd.DataFrame([dict(期間=f"{a} → {b}", 官方加權報酬指數=per(rtri, a, b), 重建=per(rB, a, b),
                               差距=per(rB, a, b) - per(rtri, a, b),
                               日報酬相關=float(np.corrcoef(rB.loc[a:b].iloc[1:], rtri.loc[a:b].iloc[1:])[0, 1]))
                          for a, b in (P25, P26)])
    print(recon.round(4).to_string())

    # 指標在 11/20 的權重（以 11/20 收盤市值）
    def wB_at(date):
        m = MC.loc[:date].iloc[-1]
        s = m.groupby(bk).sum()
        return (s / s.sum()).reindex(BUCKETS)

    # ---------------- 容量 ----------------
    m_now = MC.iloc[-1]
    # 流動性：10 天出清、每天參與率 20% → 10 × 0.2 × ADV（階段 5 第 6 課的可行性取捨：10 天＋20% TE）
    pos_cap = pd.DataFrame({"10%淨資產": 0.10 * NAV, "10%已發行股份": 0.10 * m_now, "10天×20%成交值": LIQ_DAYS * 0.2 * pd.Series(adv)})
    pos_cap.loc["2330", "10%淨資產"] = 0.25 * NAV
    pos_cap["可放金額"] = pos_cap.min(axis=1)
    cap = (pos_cap.可放金額.groupby(bk).sum() / NAV).reindex(BUCKETS)
    cap3 = (pd.concat([pos_cap[["10%淨資產", "10%已發行股份"]], 0.6 * pd.Series(adv)], axis=1).min(axis=1).groupby(bk).sum() / NAV).reindex(BUCKETS)
    capd = pd.DataFrame(dict(指標權重=wB_at(SHOCK[1]), 容量_占淨資產=cap, 對照_3天出清容量=cap3,
                             受限最多的條件=pos_cap.drop(columns="可放金額").idxmin(axis=1).groupby(bk).agg(lambda s: s.value_counts().idxmax()).reindex(BUCKETS)))

    # ---------------- 政策組合（兩個年度） ----------------
    def build(est, asof, views, conf, budget=None):
        S = lw_cov(RB.loc[est[0]:est[1]].iloc[1:])
        wB = wB_at(asof)
        wN = neutral(wB, cap)
        te_struct = te((wN - wB).values, S)
        if budget is None:
            budget = float(np.sqrt(max(TE_TOTAL ** 2 - te_struct ** 2, 0) / 2))
        Pi, alpha, delta = bl_alpha(S, wB.values, views, conf)
        wP = solve_policy(alpha, S, wN, budget, cap.clip(lower=wN))
        return dict(S=S, wB=wB, wN=wN, wP=wP, Pi=Pi, alpha=alpha, delta=delta, te_struct=te_struct, budget=budget,
                    te_strat=te((wP - wN).values, S), te_total=te((wP - wB).values, S))

    VIEW = [("AI硬體零組件", 0.05)]
    p25 = build(P25, P25[1], VIEW, 0.5)
    p25s = build(P25, P25[1], [("AI硬體零組件", 0.15)], 0.75)      # 對照：委員會核准強烈 AI 觀點
    p27 = build(P26, P26[1], VIEW, 0.5)
    for k, p in (("2025", p25), ("2027", p27)):
        print(k, "δ", round(p["delta"], 2), "結構TE", round(p["te_struct"], 4), "策略預算", round(p["budget"], 4),
              "策略TE", round(p["te_strat"], 4), "總TE", round(p["te_total"], 4))

    # 基金目前的桶權重（11/20 持股）
    st, fut = holdings(hpath)
    st["bucket"] = [bucket_of(c, q.loc[c, "產業"] if c in q.index else "") for c in st.code]
    wF = (st.groupby("bucket").weight_pct.sum() / 100).reindex(BUCKETS).fillna(0)
    wF_full = wF / wF.sum()                         # 只看股票部分的配置（期貨、現金另計）
    S27 = p27["S"]
    te_fund_now = te((wF_full - p27["wB"]).values, S27)
    # 若 2025 年就把「像基金現在這樣」的配置當政策：事前 TE
    te_fundlike_25 = te((wF_full - p25["wB"]).values, p25["S"])

    # 區間
    bd = bands(p27["wP"], p27["wN"], S27)
    pol = pd.DataFrame(dict(指標=p27["wB"], 可投資中性=p27["wN"], 政策目標=p27["wP"], 策略傾斜=p27["wP"] - p27["wN"],
                            區間下限=[p27["wP"][b] - bd[b][1] for b in BUCKETS],
                            區間上限=[p27["wP"][b] + bd[b][0] for b in BUCKETS],
                            容量=cap, 基金現況_股票部分=wF_full))
    pol["基金_vs_政策"] = pol.基金現況_股票部分 - pol.政策目標
    pol["超出區間"] = np.where(pol.基金現況_股票部分 > pol.區間上限, "高於上限",
                         np.where(pol.基金現況_股票部分 < pol.區間下限, "低於下限", ""))

    # 均衡與觀點
    bl = pd.DataFrame(dict(指標權重=p27["wB"], 年化波動=np.sqrt(np.diag(S27)), 均衡超額報酬Π=p27["Pi"],
                           觀點後α=p27["alpha"], 二零二五政策_均衡Π=p25["Pi"], 二零二五政策_α=p25["alpha"]))
    corr = S27.values / np.outer(np.sqrt(np.diag(S27)), np.sqrt(np.diag(S27)))
    corr = pd.DataFrame(corr, BUCKETS, BUCKETS)

    # 敏感度：觀點大小 × 信心
    sens = []
    for qv in (0.0, 0.025, 0.05, 0.075, 0.10):
        for cf in (0.25, 0.5, 0.75):
            p = build(P26, P26[1], [("AI硬體零組件", qv)] if qv else [], cf, budget=p27["budget"])
            row = dict(觀點_AI年超額=qv, 信心=cf, 策略TE=p["te_strat"])
            row.update({b: p["wP"][b] for b in BUCKETS})
            sens.append(row)
    sens = pd.DataFrame(sens)

    # 重抽樣：20 日區塊 bootstrap 估計期間，重估共變異 → 重解
    rng = np.random.default_rng(20261122)
    X = RB.loc[P26[0]:P26[1]].iloc[1:]
    nb = len(X) // 20
    draws = []
    for _ in range(300):
        idx = np.concatenate([np.arange(s, s + 20) for s in rng.integers(0, len(X) - 20, nb)])
        S_b = lw_cov(X.iloc[idx])
        Pi_b, a_b, _ = bl_alpha(S_b, p27["wB"].values, VIEW, 0.5)
        draws.append(solve_policy(a_b, S_b, p27["wN"], p27["budget"], cap.clip(lower=p27["wN"])))
    D = pd.DataFrame(draws)
    resamp = pd.DataFrame(dict(政策目標=p27["wP"], 重抽樣_P5=D.quantile(0.05), 重抽樣_中位=D.median(), 重抽樣_P95=D.quantile(0.95),
                               區間下限=pol.區間下限, 區間上限=pol.區間上限))

    # 壓力測試：AI 衝擊重播
    rS = (1 + RB.loc[SHOCK[0]:SHOCK[1]].iloc[1:]).prod() - 1
    rSB = per(rB, *SHOCK)
    stress = pd.DataFrame([dict(組合=k, 衝擊期報酬=float(w @ rS), 相對指標=float(w @ rS) - rSB)
                           for k, w in (("指標（重建）", p27["wB"]), ("可投資中性", p27["wN"]), ("2027 政策組合", p27["wP"]),
                                        ("基金現況配置（股票部分）", wF_full))])
    stress_b = pd.DataFrame(dict(桶衝擊期報酬=rS))

    # ---------------- 回溯：如果 2025-11 就有政策組合 ----------------
    dates = Rs.loc[P26[0]:P26[1]].index[1:]
    months = sorted(set(d.to_period("M") for d in dates))
    tgtB, tgtN, tgtP, tgtS = {}, {}, {}, {}
    a25, a25s = p25["wP"] - p25["wN"], p25s["wP"] - p25s["wN"]
    for m in months:
        first = [d for d in dates if d.to_period("M") == m][0]
        wb = WB.loc[first]
        wn = neutral(wb, cap)
        wp, ws = (wn + a25).clip(lower=0), (wn + a25s).clip(lower=0)
        tgtB[m], tgtN[m], tgtP[m], tgtS[m] = wb, wn, wp / wp.sum(), ws / ws.sum()
    R = RB.loc[dates]
    rN = port_path(R, pd.DataFrame(tgtN).T, dates)
    rP = port_path(R, pd.DataFrame(tgtP).T, dates)
    rPs = port_path(R, pd.DataFrame(tgtS).T, dates)
    rBb = port_path(R, pd.DataFrame(tgtB).T, dates)      # 桶層級重建的指標（應≈ rB）
    # 基金
    fp = pd.DataFrame(json.loads(Path(fpx).read_text())["data"])
    fp["date"] = pd.to_datetime(fp.date)
    fp = fp.set_index("date")
    fd = pd.DataFrame(json.loads(Path(fdiv).read_text())["data"])
    div = pd.Series(fd.stock_and_cache_dividend.values, pd.to_datetime(fd.date)).reindex(fp.index).fillna(0.0)
    rF = (fp.close / (fp.close.shift(1) - div) - 1).reindex(dates)
    tot = lambda s: float((1 + s.fillna(0)).prod() - 1)
    R_tri, R_B, R_Bb, R_N, R_P, R_F = tot(rtri.reindex(dates)), tot(rB.reindex(dates)), tot(rBb), tot(rN), tot(rP), tot(rF)
    R_Ps = tot(rPs)
    act = R_F - R_tri
    L1, L2, L2s = R_N - R_Bb, R_P - R_N, R_Ps - R_N
    tiers = pd.DataFrame([
        dict(層級="基金（市價含息）", 數值=R_F, 說明="2025-11-20 → 2026-11-20"),
        dict(層級="指標：官方加權報酬指數", 數值=R_tri, 說明=""),
        dict(層級="主動報酬 = 基金 − 指標", 數值=act, 說明="要拆解的總數"),
        dict(層級="① 結構層 = 可投資中性 − 指標", 數值=L1, 說明="台積電 25% 上限與容量造成，法規決定，不屬於經理人"),
        dict(層級="② 策略層 = 政策組合 − 可投資中性", 數值=L2, 說明="SAA：2025-11 依 AI 觀點（年超額 5%、信心 50%）核准的傾斜"),
        dict(層級="③ 執行層 = 主動 − ① − ②", 數值=act - L1 - L2, 說明="戰術偏離＋選股＋費用＋市價折溢價"),
        dict(層級="對照②′：強烈觀點政策 − 可投資中性", 數值=L2s, 說明="若委員會核准 AI 年超額 15%、信心 75%"),
        dict(層級="對照③′：主動 − ① − ②′", 數值=act - L1 - L2s, 說明="即使政策押 AI，執行層仍要解釋的部分"),
        dict(層級="參考：重建指標 − 官方指標", 數值=R_Bb - R_tri, 說明="存活者偏誤等資料誤差；①②用重建組合之間的差，抵銷大部分偏誤"),
    ])
    print(tiers.round(4).to_string())
    cum = pd.DataFrame(dict(指標_官方=(1 + rtri.reindex(dates)).cumprod() - 1, 指標_重建=(1 + rBb).cumprod() - 1,
                            可投資中性=(1 + rN).cumprod() - 1, 政策組合2025=(1 + rP).cumprod() - 1,
                            強烈觀點政策=(1 + rPs).cumprod() - 1, 基金=(1 + rF.fillna(0)).cumprod() - 1))
    monthly = pd.DataFrame({k: (1 + v).groupby(v.index.to_period("M")).prod() - 1
                            for k, v in dict(指標=rtri.reindex(dates), 指標重建=rBb, 可投資中性=rN, 政策組合=rP,
                                             基金=rF.fillna(0)).items()})
    monthly["主動"] = monthly.基金 - monthly.指標
    monthly["結構層"] = monthly.可投資中性 - monthly.指標重建
    monthly["策略層"] = monthly.政策組合 - monthly.可投資中性
    monthly["執行層"] = monthly.主動 - monthly.結構層 - monthly.策略層
    monthly.index = monthly.index.astype(str)
    p25t = pd.DataFrame(dict(指標=p25["wB"], 可投資中性=p25["wN"], 政策目標=p25["wP"], 策略傾斜=p25["wP"] - p25["wN"],
                             強烈觀點政策=p25s["wP"], 強烈觀點傾斜=p25s["wP"] - p25s["wN"]))
    te_tab = pd.DataFrame([
        dict(項目="總 TE 預算（階段 5）", 二零二五政策=TE_TOTAL, 二零二七政策=TE_TOTAL),
        dict(項目="結構 TE（N 對指標）", 二零二五政策=p25["te_struct"], 二零二七政策=p27["te_struct"]),
        dict(項目="策略 TE 預算 = √((15%² − 結構²)/2)", 二零二五政策=p25["budget"], 二零二七政策=p27["budget"]),
        dict(項目="策略 TE 實際（P 對 N）", 二零二五政策=p25["te_strat"], 二零二七政策=p27["te_strat"]),
        dict(項目="政策 TE（P 對指標）", 二零二五政策=p25["te_total"], 二零二七政策=p27["te_total"]),
        dict(項目="戰術 TE 剩餘空間 ≈ √(15%² − 政策TE²)", 二零二五政策=np.sqrt(max(TE_TOTAL ** 2 - p25["te_total"] ** 2, 0)),
             二零二七政策=np.sqrt(max(TE_TOTAL ** 2 - p27["te_total"] ** 2, 0))),
        dict(項目="對照：強烈觀點政策的策略 TE（P 對 N）", 二零二五政策=p25s["te_strat"], 二零二七政策=np.nan),
        dict(項目="基金現況配置的事前 TE（對指標）", 二零二五政策=te_fundlike_25, 二零二七政策=te_fund_now),
    ])
    print(te_tab.round(4).to_string())
    print(pol.round(4).to_string())
    print(stress.round(4).to_string())
    members = pd.DataFrame(dict(代號=codes, 名稱=[names.get(c, "") for c in codes], 產業=[lst.loc[c, "產業"] for c in codes],
                                桶=bk.values, 指標權重_1120=(m_now / m_now.sum()).values, 可放金額_億=(pos_cap.可放金額 / 1e8).values,
                                特殊跳空修正日數=[flags[c] for c in codes]))

    out = Path(prefix + ".xlsx")
    with pd.ExcelWriter(out) as xw:
        recon.to_excel(xw, "1_指標重建勾稽", index=False)
        te_tab.to_excel(xw, "2_TE預算分層", index=False)
        pol.rename_axis("桶").reset_index().to_excel(xw, "3_2027政策組合", index=False)
        capd.rename_axis("桶").reset_index().to_excel(xw, "4_容量", index=False)
        bl.rename_axis("桶").reset_index().to_excel(xw, "5_均衡與觀點", index=False)
        corr.round(3).rename_axis("桶").reset_index().to_excel(xw, "6_相關係數", index=False)
        sens.to_excel(xw, "7_觀點敏感度", index=False)
        resamp.rename_axis("桶").reset_index().to_excel(xw, "8_重抽樣", index=False)
        stress.to_excel(xw, "9_AI衝擊重播", index=False)
        stress_b.rename_axis("桶").reset_index().to_excel(xw, "9b_桶衝擊報酬", index=False)
        tiers.to_excel(xw, "10_回溯三層拆解", index=False)
        monthly.rename_axis("月").reset_index().to_excel(xw, "11_逐月三層", index=False)
        cum.reset_index().rename(columns={"index": "日期"}).to_excel(xw, "12_累積報酬", index=False)
        p25t.rename_axis("桶").reset_index().to_excel(xw, "13_2025政策組合", index=False)
        members.to_excel(xw, "14_成分與分桶", index=False)
    print("輸出", out, "成分", len(codes), "跳空修正", sum(flags.values()))
    return dict(p25=p25, p25s=p25s, p27=p27, pol=pol, tiers=tiers, monthly=monthly, stress=stress, sens=sens, resamp=resamp,
                te_tab=te_tab, recon=recon, capd=capd, bl=bl, wF=wF_full, members=members, corr=corr, rS=rS)


if __name__ == "__main__":
    main(*sys.argv[1:])
