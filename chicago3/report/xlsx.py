"""完整試算表：每個計算步驟一張工作表（模型輸出，數值由程式計算；重跑程式即可更新）。"""
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

FONT, BOLD = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
HEAD_FILL = PatternFill("solid", fgColor="F2F1EC")
PCT = "0.00%"


def _sheets(o: dict) -> list[tuple[str, pd.DataFrame, list | str | None]]:
    """回傳 [(工作表名, 表, 百分比欄位：list、'all' 或 None)]。"""
    cfg = o["config"]
    S = o["S"]
    from .. import risk
    pol = pd.DataFrame(dict(指標=o["wB"], 可投資中性=o["wN"], 政策目標=o["wP"], 策略傾斜=o["wP"] - o["wN"],
                            區間下限=o["bands"].lower, 區間上限=o["bands"].upper, 容量=o["capacity"]))
    if "wF" in o:
        pol["基金現況_股票部分"] = o["wF"]
        pol["基金_減_政策"] = o["wF"] - o["wP"]
        pol["狀態"] = o["fund_status"].map({"above": "高於上限", "below": "低於下限", "": ""})
    te = pd.DataFrame([("總 TE 預算", o["te"]["total_budget"]), ("結構 TE（中性對指標）", o["te"]["structural"]),
                       ("策略 TE 上限", o["te"]["strategic_cap"]), ("策略 TE 實際（政策對中性）", o["te"]["strategic"]),
                       ("政策 TE（政策對指標）", o["te"]["policy"]), ("戰術＋選股空間", o["te"]["tactical_room"])]
                      + ([("基金現況事前 TE（對指標）", o["te_fund"])] if "te_fund" in o else []), columns=["項目", "數值"])
    cap = pd.DataFrame(dict(指標權重=o["wB"], 容量=o["capacity"],
                            **{f"對照_{d:g}天出清": v for d, v in o["capacity_compare"].items()}))
    bl = pd.DataFrame(dict(指標權重=o["wB"], 年化波動=risk.vol(S), Beta=S.values @ o["wB"].values / (o["wB"].values @ S.values @ o["wB"].values),
                           均衡超額報酬Π=o["bl"].Pi, 混合後預期μ=o["bl"].mu, 觀點α=o["bl"].alpha), index=S.index)
    out = [("1_指標重建勾稽", o["recon"].rename(columns=dict(period="期間", official="官方加權報酬指數", rebuilt="重建", diff="差距", corr="日報酬相關")),
            ["官方加權報酬指數", "重建", "差距"]),
           ("2_TE預算分層", te, ["數值"]),
           ("3_政策組合", pol, [c for c in pol.columns if c != "狀態"]),
           ("4_容量", cap, "all"),
           ("5_均衡與觀點", bl, ["指標權重", "年化波動", "均衡超額報酬Π", "混合後預期μ", "觀點α"]),
           ("6_相關係數", risk.corr(S).round(3), None)]
    if "fund_expected" in o:
        fe = o["fund_expected"]
        out.append(("5b_預期主動拆解", pd.DataFrame([("Beta", o["beta_fund"]), ("預期主動報酬", fe["expected_active"]),
                                                    ("其中 Beta 溢酬 = ERP ×（β − 1）", fe["beta_premium"]),
                                                    ("其中觀點 α", fe["view_alpha"])], columns=["項目", "數值"]), None))
    rb = o.get("robustness", {})
    if "sensitivity" in rb:
        out.append(("7_觀點敏感度", rb["sensitivity"].rename(columns=dict(excess="觀點_年超額", confidence="信心", te_strategic="策略TE")),
                    ["觀點_年超額", "策略TE", *cfg.buckets.order]))
    if "bootstrap" in rb:
        out.append(("8_重抽樣", rb["bootstrap"].rename(columns=dict(policy="政策目標", p5="P5", median="中位數", p95="P95",
                                                                 band_lower="區間下限", band_upper="區間上限")), "all"))
    for name, (t, rS) in rb.get("stress", {}).items():
        t2 = t.assign(portfolio=t.portfolio.map(lambda k: {"benchmark": "指標", "neutral": "可投資中性", "policy": "政策組合",
                                                         "fund": "基金現況"}.get(k, k)))
        out.append((f"9_壓力_{name}"[:31], t2.rename(columns=dict(portfolio="組合", return_="期間報酬", vs_benchmark="相對指標")),
                    ["期間報酬", "相對指標"]))
        out.append((f"9b_桶報酬_{name}"[:31], rS.rename("期間報酬").to_frame(), "all"))
    at = o.get("attribution")
    if at:
        t = at["tiers"]
        rows = [("基金（市價含息）", t["fund"]), ("指標（官方加權報酬指數）", t["benchmark"]), ("主動報酬", t["active"]),
                ("① 結構層 = 中性 − 指標", t["structural"])]
        for k, v in t["policies"].items():
            rows += [(f"② 策略層［{k}］", v["strategic"]), (f"③ 執行層［{k}］", v["execution"])]
        rows.append(("參考：重建指標 − 官方指標（資料誤差）", t["data_error"]))
        out.append(("10_回溯三層拆解", pd.DataFrame(rows, columns=["層級", "數值"]), ["數值"]))
        out.append(("11_逐月三層", at["monthly"], "all"))
        out.append(("12_回溯政策傾斜", pd.DataFrame(at["tilts"]), "all"))
    if "frontier" in o:
        out.append(("13_效率前緣", o["frontier"].rename(columns=dict(risk="TE", ret="預期主動")), "all"))
    if "monitor" in o:
        m = o["monitor"]
        out.append(("14_區間監控", pd.DataFrame(dict(基金現況=o["wF"], 最近區間內目標=m["band_target"], 政策目標=o["wP"])), "all"))
        out.append(("14b_換手", pd.DataFrame([("區間外桶數", m["n_out"]), ("到政策目標的單邊換手", m["turnover_to_policy"]),
                                             ("到區間邊界的單邊換手（最小換手）", m["turnover_to_band"])], columns=["項目", "數值"]), ["數值"]))
    ct = o["capacity_table"].join(o["market"].panel.bucket.rename("bucket")).join(o["market"].snapshot[["名稱", "產業"]])
    ct["容量_億"] = ct.capacity / 1e8
    out.append(("15_成分與容量", ct[["名稱", "產業", "bucket", "容量_億", "binding"]].rename(columns=dict(bucket="桶", binding="最緊的限制")), None))
    return out


def write_workbook(o: dict, path):
    cfg = o["config"]
    sheets = _sheets(o)
    readme = pd.DataFrame([
        ("基金", f"{cfg.fund.code} {cfg.fund.name}"), ("指標權重日", str(cfg.dates.asof)),
        ("估計期間", f"{cfg.dates.estimation[0]} → {cfg.dates.estimation[1]}（不含起日）"),
        ("規模（容量計算用）", f"{cfg.fund.nav / 1e8:,.0f} 億元"),
        ("觀點", "；".join(f"{v.bucket} 年超額 {v.excess:.1%}、信心 {v.confidence:.0%}" for v in cfg.views) or "無"),
        ("數字來源", "chicago3 套件計算；所有數值為模型輸出，修改設定檔後重跑即可更新"),
        ("聲明", "教學與研究用途；模型輸出須經投資委員會核准，不構成投資建議"),
    ], columns=["項目", "內容"])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="0_說明", index=False)
        for name, df, _ in sheets:
            df.to_excel(xw, sheet_name=name, index=not isinstance(df.index, pd.RangeIndex))
    wb = load_workbook(path)
    pct = {name: p for name, _, p in sheets}
    for ws in wb.worksheets:
        heads = [c.value for c in ws[1]]
        p = pct.get(ws.title)
        for row in ws.iter_rows():
            for c in row:
                c.font = FONT
        for c in ws[1]:
            c.font, c.fill, c.alignment = BOLD, HEAD_FILL, Alignment(horizontal="center", vertical="center", wrap_text=True)
        for j, h in enumerate(heads, start=1):
            is_pct = p == "all" or (isinstance(p, list) and h in p)
            col = ws.cell(row=1, column=j).column_letter
            width = max(len(str(h or "")) * 1.8, 10)
            for i in range(2, ws.max_row + 1):
                cell = ws.cell(row=i, column=j)
                if isinstance(cell.value, bool) or j == 1:
                    pass
                elif isinstance(cell.value, int):
                    cell.number_format = "0"
                elif isinstance(cell.value, float):
                    cell.number_format = PCT if is_pct else "0.0000"
                width = max(width, min(len(str(cell.value or "")) * 1.3, 40))
            ws.column_dimensions[col].width = width
        ws.freeze_panes = "B2"
    wb.save(path)
