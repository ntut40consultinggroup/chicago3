"""政策組合書（Markdown）：依課程範本《00981A_2027年政策組合書_範本》的九節結構，由計算結果自動填數字。

研究依據與失效條件無法由程式產生，一律標「待補」並在第 3 節提醒：沒有這兩項的觀點不得列入政策。
"""
from datetime import date

STATUS = {"above": "高於上限", "below": "低於下限", "": "區間內"}


def _p(x, d=1):
    return f"{x * 100:.{d}f}%"


def policy_book(o: dict, today: date | None = None) -> str:
    cfg, te = o["config"], o["te"]
    yr = cfg.dates.asof.year + 1
    rows = []
    for b in cfg.buckets.order:
        rows.append(f"| {b} | {_p(o['wB'][b])} | {_p(o['wN'][b])} | **{_p(o['wP'][b])}** | "
                    f"{_p(o['bands'].lower[b])}–{_p(o['bands'].upper[b])} |")
    views = "\n".join(
        f"| V{i + 1} | {v.bucket} 對指標每年超額 {_p(v.excess, 1)} | {yr} 全年 | {v.confidence:.0%} | **待補** | **待補**（至少三個可觀察的條件） |"
        for i, v in enumerate(cfg.views)) or "| — | 無策略觀點（政策 = 可投資中性） | | | | |"
    c = cfg.constraints
    caps = ", ".join(f"{k} {v:.0%}" for k, v in c.single_cap_overrides.items())
    lines = [
        f"# {cfg.fund.code} {yr} 年政策組合書（草案）",
        "",
        f"> 由 Chicago 3.0 自動產生（{(today or date.today()).isoformat()}）；資料截至 {cfg.dates.asof}。"
        "觀點、研究依據與失效條件須經投資委員會核准；模型輸出不構成投資建議。",
        "",
        "## 1. 目的與範圍",
        "",
        f"本文件定義 {cfg.fund.code} 在 {yr} 年的政策組合（策略性資產配置），作為日常決策的錨，"
        "並把主動報酬拆成結構、策略、執行三層，讓責任歸屬可以事後檢驗。本文件不取代信託契約、公開說明書與法規。",
        "",
        "## 2. 指標與可投資中性",
        "",
        "| 項目 | 內容 |",
        "|---|---|",
        f"| 指標 | 重建自{cfg.data.market}市值前 {cfg.data.universe_top_n} 大，並與官方加權報酬指數勾稽（見附件工作表 1） |",
        f"| 可投資中性 | 指標按 {len(cfg.buckets.order)} 個桶加總；桶上限：{', '.join(f'{k} {v:.0%}' for k, v in c.bucket_upper.items()) or '無'}；"
        "各桶不超過容量；超出部分按比例分配 |",
        f"| 容量假設 | 規模 {cfg.fund.nav / 1e8:,.0f} 億元；每檔上限 = min({c.single_cap:.0%} 淨資產（{caps or '無例外'}）, "
        f"{c.shares_cap:.0%} 已發行股份, {c.liquidity.days:g} 天 × {c.liquidity.participation:.0%} 日均成交值) |",
        f"| 結構 TE（中性對指標） | {_p(te['structural'])}（不屬於經理人的判斷） |",
        "",
        f"**桶的定義：** {'｜'.join(cfg.buckets.order)}。分類清單見附件工作表 15。",
        "",
        "## 3. 策略觀點",
        "",
        "| 編號 | 觀點 | 期間 | 信心 | 研究依據 | 失效條件（任一出現即觸發重檢） |",
        "|---|---|---|---|---|---|",
        views,
        "",
        "沒有研究依據、沒有失效條件的觀點，不得列入政策。",
        "",
        "## 4. 政策目標與區間",
        "",
        "| 桶 | 指標 | 可投資中性 | **政策目標** | **區間** |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "- 權重以股票部位為分母，不含期貨與現金。",
        f"- 區間規則：每桶偏離到「策略 TE 增加 {cfg.bands.te_increment * 100:g} 個百分點」為止，"
        f"{'取該距離與目標權重一半的較小者，' if cfg.bands.half_weight_cap else ''}"
        f"限制在 {cfg.bands.clip[0] * 100:g}–{cfg.bands.clip[1] * 100:g} 個百分點"
        f"{'；' + '、'.join(cfg.bands.no_up) + ' 不往上（已在上限）' if cfg.bands.no_up else ''}。",
        "",
        "## 5. 風險預算",
        "",
        "| 項目 | 數值 | 核准層級 |",
        "|---|---|---|",
        f"| 總 TE 預算（事前，年化） | {_p(te['total_budget'])} | 投資委員會 |",
        f"| 結構 TE | {_p(te['structural'])} | 法規決定，僅監控 |",
        f"| 策略 TE 上限／本年實際 | {_p(te['strategic_cap'])}／{_p(te['strategic'])} | 投資委員會 |",
        f"| 政策 TE（政策對指標） | {_p(te['policy'])} | 投資委員會 |",
        f"| 戰術＋選股空間 | 約 {_p(te['tactical_room'])} | 經理人，在此範圍內自行決定 |",
        "",
        "## 6. 再平衡規則",
        "",
        "1. **區間內：** 經理人可自行調整，不需另行核准。",
        "2. **碰到或超出區間：** 5 個營業日內提出處理方案：(a) 回到區間的交易計畫（換手、成本、期限；優先用申購買回資金），"
        "或 (b) 修改政策的申請（附新的研究依據與風險預算）。",
        "3. **戰術偏離：** 每一筆寫訊號、規模（從戰術空間扣）、預期持有期間、出場條件，並記入決策日誌。",
        "",
        "## 7. 監控與觸發重檢",
        "",
        "| 觸發條件 | 理由 |",
        "|---|---|",
        "| 任一觀點的失效條件出現 | 政策的前提消失 |",
        "| 基金規模較年初變動 ±50% | 容量與可投資中性改變 |",
        "| 任一桶 60 日年化波動較政策估計值變動 ±50% | 風險估計過期 |",
        f"| 事前總 TE 連續 20 個營業日 > {_p(te['total_budget'], 0)} | 超出預算 |",
        "| 法規或契約修訂 | 限制改變 |",
        "",
    ]
    lines += _status_section(o)
    lines += [
        "## 9. 核准紀錄",
        "",
        "| 角色 | 姓名 | 日期 | 簽核 |",
        "|---|---|---|---|",
        "| 基金經理人（提案） | | | |",
        "| 風險管理 | | | |",
        "| 法令遵循 | | | |",
        "| 投資委員會主席 | | | |",
        "",
        "附件：同資料夾的試算表（每個計算步驟一張工作表）、效率前緣圖與區間圖；設定檔可完整重跑。",
        "",
    ]
    return "\n".join(lines)


def _status_section(o: dict) -> list[str]:
    cfg = o["config"]
    out = [f"## 8. 現況與待決事項（{cfg.dates.asof}）", ""]
    if "wF" not in o:
        return out + ["（未提供基金持股，無法檢查現況。）", ""]
    st = o["fund_status"]
    off = [f"{b} {_p(o['wF'][b])}（{'上限' if st[b] == 'above' else '下限'} "
           f"{_p(o['bands'].upper[b] if st[b] == 'above' else o['bands'].lower[b])}）" for b in st.index if st[b]]
    out.append(f"- 基金現況（股票部分）有 {len(off)} 桶在區間外" + (f"：{'、'.join(off)}。" if off else "。"))
    budget = _p(o["te"]["total_budget"], 0)
    verdict = f"**超過**總預算 {budget}" if o["te_fund"] > o["te"]["total_budget"] else f"在總預算 {budget} 內"
    out.append(f"- 現況配置的事前 TE 為 {_p(o['te_fund'])}，{verdict}；β {o['beta_fund']:.2f}。")
    fe = o["fund_expected"]
    out.append(f"- 現況配置的預期主動報酬 {_p(fe['expected_active'], 2)}，其中 Beta 溢酬 {_p(fe['beta_premium'], 2)}、"
               f"觀點 α {_p(fe['view_alpha'], 2)}：Beta 溢酬只是把風險加在高 Beta 的桶上，不是判斷力。")
    for name, (t, _) in o.get("robustness", {}).get("stress", {}).items():
        d = t.set_index("portfolio").vs_benchmark
        if "fund" in d and "policy" in d:
            out.append(f"- 壓力重播「{name}」：現況配置相對指標 {d['fund'] * 100:+.1f} 個百分點，政策組合 {d['policy'] * 100:+.1f}。")
    if "monitor" in o:
        m = o["monitor"]
        out.append(f"- 回到政策目標的單邊換手約 {_p(m['turnover_to_policy'], 0)}；只回到區間邊界（最小換手）約 "
                   f"{_p(m['turnover_to_band'], 0)}。")
    if off:
        out.append("- **待投資委員會決定：** 選項 A（修改政策，須補研究依據與失效條件）或選項 B（分階段回到區間）。")
    return out + [""]
