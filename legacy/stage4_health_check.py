"""階段 4 第 1 課：每日基金健檢表 v1。

用法：
  python3 -I stage4_health_check.py fetch <新資料夾>                       # 下載證交所、櫃買中心公開資料
  python3 -I stage4_health_check.py run <資料夾> <持股.csv> <輸出檔名前綴>   # 產生健檢表（.xlsx + .md）

管線：資料檢查 → 持股估值 → 限制 → 集中度與曝險 → 流動性 → 觸發條件 → 當日貢獻 → 燈號摘要
每一項檢查都回傳（項目, 數值, 門檻, 燈號, 說明），燈號：紅 = 違反或需立即處理；黃 = 預警；綠 = 正常。
門檻中標「內規（教學）」者是教學設定，實務以公司內規為準。
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------- 設定 ----------------
FUND = "00981A 主動統一台股增長"
URLS = {
    "t187ap03_L.json": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
    "exchangeReport_STOCK_DAY_ALL.json": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
    "exchangeReport_MI_INDEX.json": "https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX",
    "exchangeReport_BWIBBU_ALL.json": "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL",
    "mopsfin_t187ap03_O.json": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
    "tpex_mainboard_daily_close_quotes.json": "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
}
INDUSTRY = {"01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織", "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷",
            "09": "造紙", "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造", "15": "航運", "16": "觀光餐旅",
            "17": "金融保險", "18": "貿易百貨", "20": "其他", "21": "化學", "22": "生技醫療", "23": "油電燃氣",
            "24": "半導體", "25": "電腦及週邊設備", "26": "光電", "27": "通信網路", "28": "電子零組件", "29": "電子通路",
            "30": "資訊服務", "31": "其他電子", "35": "綠能環保", "36": "數位雲端", "37": "運動休閒", "38": "居家生活"}
NON_AI = {"2891", "2002", "1802", "2637", "1590", "4551", "3008", "3376", "6278"}   # 主題標籤（教學判斷）
TOP_N = 300
LIMITS = dict(stock_min=70.0, top300_min=60.0, single_max=10.0, tsmc_max=25.0, own_shares_max=10.0)
INTERNAL = dict(use_warn=0.80, own_warn=5.0, theme_warn=75.0, ind_active_warn=20.0, illiquid_days=3.0,
                illiquid_warn=20.0, eff_n_warn=15.0, day_active_warn=-2.0)
DECISIONS = [dict(id="00981A-2026-1120-2308", code="2308", ref_px=1000, add_below=785, trim_above=1304,
                  review_move=0.10, valid_until="2026-12-10")]


# ---------------- 資料 ----------------
def fetch(outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=False)
    for fn, url in URLS.items():
        for _ in range(4):
            r = subprocess.run(["curl", "-sS", "-m", "60", url, "-o", str(out / fn)])
            if r.returncode == 0 and (out / fn).stat().st_size > 2:
                break
            time.sleep(3)
        print(fn, (out / fn).stat().st_size if (out / fn).exists() else "失敗")


def _f(x):
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None


def load(d):
    d = Path(d)
    j = lambda fn: json.loads((d / fn).read_text())
    tb, ob = j("t187ap03_L.json"), j("mopsfin_t187ap03_O.json")
    tq, oq = j("exchangeReport_STOCK_DAY_ALL.json"), j("tpex_mainboard_daily_close_quotes.json")
    idx = {r["指數"]: r for r in j("exchangeReport_MI_INDEX.json")}
    meta = {r["公司代號"]: ("上市", _f(r["已發行普通股數或TDR原股發行股數"]), r["產業別"]) for r in tb}
    meta.update({r["SecuritiesCompanyCode"]: ("上櫃", _f(r["IssueShares"]), r["SecuritiesIndustryCode"]) for r in ob})
    rows = []
    for r in tq:
        c, chg = _f(r["ClosingPrice"]), _f(r["Change"])
        rows.append((r["Code"], c, chg, _f(r["TradeValue"]), r["Date"]))
    for r in oq:
        c, chg = _f(r["Close"]), _f(r["Change"])
        rows.append((r["SecuritiesCompanyCode"], c, chg, _f(r["TransactionAmount"]), r["Date"]))
    q = pd.DataFrame(rows, columns=["代號", "收盤", "漲跌", "成交值", "資料日"])
    q = q[q.代號.isin(meta)].dropna(subset=["收盤"]).copy()
    q["市場"] = q.代號.map(lambda c: meta[c][0])
    q["股數"] = q.代號.map(lambda c: meta[c][1])
    q["產業"] = q.代號.map(lambda c: INDUSTRY.get(meta[c][2], meta[c][2]))
    q["報酬"] = q.漲跌 / (q.收盤 - q.漲跌)
    q["市值"] = q.收盤 * q.股數
    q = q.sort_values("市值", ascending=False)
    q["市值排名"] = range(1, len(q) + 1)
    twse = q[q.市場 == "上市"]
    q["指標權重"] = (q.市值 / twse.市值.sum()).where(q.市場 == "上市", 0.0)
    taiex = idx.get("發行量加權股價指數", {})
    pct = abs(_f(taiex.get("漲跌百分比", 0)) or 0) / 100      # 漲跌百分比可能已帶負號，方向以「漲跌」欄為準
    return q.set_index("代號"), -pct if taiex.get("漲跌") == "-" else pct


def holdings(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    st = pd.DataFrame([r for r in rows if r["type"] == "stock"])
    st["weight_pct"] = st.weight_pct.astype(float)
    st["shares"] = st.shares.astype(float)
    fut = sum(float(r["weight_pct"]) for r in rows if r["type"] == "future")
    return st, fut


# ---------------- 檢查 ----------------
def light(bad, warn):
    return "紅" if bad else ("黃" if warn else "綠")


def run(d, hpath, prefix):
    q, idx_ret = load(d)
    st, fut = holdings(hpath)
    checks, notes = [], []

    # 1. 資料檢查
    dates = sorted({f"{int(x[:3]) + 1911}-{x[3:5]}-{x[5:]}" for x in q.資料日})
    missing = [c for c in st.code if c not in q.index]
    checks.append(("資料日一致（上市／上櫃）", "、".join(dates), "只有一個日期", light(len(dates) > 1, False), ""))
    checks.append(("持股都有收盤價", f"缺 {len(missing)} 檔", "0", light(bool(missing), False), "、".join(missing)))

    h = st.merge(q, left_on="code", right_index=True, how="left")
    h["部位市值"] = h.shares * h.收盤
    nav = h.部位市值.sum() / (h.weight_pct.sum() / 100)
    stock_w = h.weight_pct.sum()

    # 2. 法規與契約限制
    checks.append(("股票權重 ≥ 70%", f"{stock_w:.2f}%", "≥ 70%", light(stock_w < LIMITS["stock_min"], stock_w < 75), "法規"))
    top300 = h.loc[h.市值排名 <= TOP_N, "weight_pct"].sum()
    checks.append(("市值前 300 大 ≥ 60%", f"{top300:.2f}%", "≥ 60%", light(top300 < LIMITS["top300_min"], top300 < 65),
                   "契約；以當日市值排名估算"))
    lim_rows = []
    for _, r in h.iterrows():
        cap = min(LIMITS["tsmc_max"], r.指標權重 * 100) if r.code == "2330" else LIMITS["single_max"]
        use = r.weight_pct / cap if cap else 0
        own = r.shares / r.股數 * 100 if r.股數 else 0
        lim_rows.append(dict(代號=r.code, 名稱=r["name"], 權重=r.weight_pct, 上限=round(cap, 2), 使用率=use,
                             持有已發行股份pct=own, 燈號=light(use > 1 or own > LIMITS["own_shares_max"],
                                                          use >= INTERNAL["use_warn"] or own >= INTERNAL["own_warn"])))
    lim = pd.DataFrame(lim_rows).sort_values("使用率", ascending=False)
    worst = lim.iloc[0]
    checks.append(("單一個股上限（台積電：≤ 25% 且 ≤ 指標權重）", f"最高 {worst.名稱} 使用率 {worst.使用率:.0%}",
                   "< 80% 綠；80–100% 黃", light(worst.使用率 > 1, worst.使用率 >= INTERNAL["use_warn"]), "法規"))
    top_own = lim.sort_values("持有已發行股份pct", ascending=False).iloc[0]
    checks.append(("持有已發行股份 ≤ 10%（本基金）", f"最高 {top_own.名稱} {top_own.持有已發行股份pct:.2f}%",
                   "< 5% 綠", light(top_own.持有已發行股份pct > 10, top_own.持有已發行股份pct >= 5),
                   "法規上限為全公司基金合計，需另行加總"))

    # 3. 集中度與曝險
    w = h.weight_pct / stock_w
    hhi = (w ** 2).sum()
    eff_n = 1 / hhi
    top10 = h.nlargest(10, "weight_pct").weight_pct.sum()
    checks.append(("有效持股數（1 ÷ HHI）", f"{eff_n:.1f} 檔（帳面 {len(h)} 檔）", "≥ 15（內規，教學）",
                   light(False, eff_n < INTERNAL["eff_n_warn"]), f"前十大 {top10:.1f}%"))
    ai = h.loc[~h.code.isin(NON_AI), "weight_pct"].sum() / stock_w * 100
    checks.append(("AI 供應鏈主題曝險（占股票部位）", f"{ai:.1f}%", "< 75%（內規，教學）",
                   light(False, ai >= INTERNAL["theme_warn"]), "第 6 課 FMEA 第一名風險"))
    ind = h.groupby("產業").weight_pct.sum().rename("基金%").to_frame()
    bench = q[q.市場 == "上市"].groupby("產業").指標權重.sum().mul(100).rename("指標%")
    ind = ind.join(bench, how="outer").fillna(0)
    ind["主動權重"] = ind["基金%"] - ind["指標%"]
    ind = ind.sort_values("主動權重", ascending=False)
    big = ind.iloc[0]
    checks.append(("最大產業主動權重", f"{big.name} {big.主動權重:+.1f} 個百分點", "< +20（內規，教學）",
                   light(False, big.主動權重 >= INTERNAL["ind_active_warn"]),
                   f"最大低配：{ind.index[-1]} {ind.iloc[-1].主動權重:+.1f}"))
    act = h[["code", "name", "weight_pct", "指標權重"]].copy()
    act["指標權重"] *= 100
    act["主動權重"] = act.weight_pct - act.指標權重

    # 4. 流動性（以當日成交值 20% 估計出清天數）
    h["出清天數"] = h.部位市值 / (0.2 * h.成交值)
    illiq = h.loc[h.出清天數 > INTERNAL["illiquid_days"], "weight_pct"].sum()
    checks.append(("出清 > 3 天的部位權重", f"{illiq:.1f}%", "< 20%（內規，教學）",
                   light(False, illiq >= INTERNAL["illiquid_warn"]), "以當日成交值估計；大跌日偏悲觀"))

    # 5. 觸發條件
    trig = []
    for dcs in DECISIONS:
        px = q.loc[dcs["code"], "收盤"]
        mv = px / dcs["ref_px"] - 1
        hit = []
        if px <= dcs["add_below"]:
            hit.append("可提案加碼")
        if px >= dcs["trim_above"]:
            hit.append("考慮調節")
        if round(abs(mv), 4) >= dcs["review_move"]:
            hit.append(f"變動 {mv:+.1%}，重新檢視")
        trig.append(dict(決定書=dcs["id"], 收盤=px, 參考價=dcs["ref_px"], 變動=mv, 觸發="；".join(hit) or "無"))
        checks.append((f"決定書 {dcs['id']}", f"收 {px:,.0f}（{mv:+.1%}）", "加碼 ≤ 785／調節 ≥ 1,304／±10%",
                       light(False, bool(hit)), "；".join(hit) or "無觸發"))

    # 6. 當日貢獻（用前一日權重：w0 ∝ w1 ÷ (1 + r)）
    h["前日權重"] = h.weight_pct / (1 + h.報酬)
    h["前日權重"] *= stock_w / h.前日權重.sum()
    h["貢獻"] = h.前日權重 * h.報酬
    fund_ret = h.貢獻.sum() / 100 + fut / 100 * idx_ret
    checks.append(("當日估計報酬 vs 加權指數", f"基金 {fund_ret:+.2%}／指數 {idx_ret:+.2%}",
                   "主動 > −2 個百分點（內規，教學）", light(False, (fund_ret - idx_ret) * 100 < INTERNAL["day_active_warn"]),
                   "以持股估算，非實際淨值"))

    summary = pd.DataFrame(checks, columns=["項目", "數值", "門檻", "燈號", "說明"])
    contrib = h[["code", "name", "前日權重", "報酬", "貢獻"]].sort_values("貢獻")
    liq = h[["code", "name", "weight_pct", "部位市值", "成交值", "出清天數"]].sort_values("出清天數", ascending=False)
    with pd.ExcelWriter(f"{prefix}.xlsx") as xw:
        summary.to_excel(xw, "0_燈號摘要", index=False)
        lim.to_excel(xw, "1_限制", index=False)
        ind.reset_index().to_excel(xw, "2_產業曝險", index=False)
        act.sort_values("主動權重").to_excel(xw, "3_個股主動權重", index=False)
        liq.to_excel(xw, "4_流動性", index=False)
        pd.DataFrame(trig).to_excel(xw, "5_觸發條件", index=False)
        contrib.to_excel(xw, "6_當日貢獻", index=False)

    n = summary.燈號.value_counts()
    md = [f"# 每日基金健檢表｜{FUND}", "", f"- 資料日：{'、'.join(dates)}　估計淨資產：約 {nav / 1e8:,.0f} 億元",
          f"- 燈號：紅 {n.get('紅', 0)}、黃 {n.get('黃', 0)}、綠 {n.get('綠', 0)}", "",
          "| 燈號 | 項目 | 數值 | 門檻 | 說明 |", "|---|---|---|---|---|"]
    icon = {"紅": "🔴", "黃": "🟡", "綠": "🟢"}
    for _, r in summary.sort_values("燈號", key=lambda s: s.map({"紅": 0, "黃": 1, "綠": 2})).iterrows():
        md.append(f"| {icon[r.燈號]} | {r.項目} | {r.數值} | {r.門檻} | {r.說明} |")
    md += ["", "## 產業曝險（前 5 大超配／低配）", "", "| 產業 | 基金% | 指標% | 主動 |", "|---|---|---|---|"]
    for name, r in pd.concat([ind.head(5), ind.tail(3)]).iterrows():
        md.append(f"| {name} | {r['基金%']:.1f} | {r['指標%']:.1f} | {r.主動權重:+.1f} |")
    md += ["", "## 當日拖累最大的 5 檔", "", "| 持股 | 前日權重% | 報酬 | 貢獻（百分點） |", "|---|---|---|---|"]
    for _, r in contrib.head(5).iterrows():
        md.append(f"| {r['name']} | {r.前日權重:.2f} | {r.報酬:+.2%} | {r.貢獻:+.2f} |")
    Path(f"{prefix}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2])
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4])
