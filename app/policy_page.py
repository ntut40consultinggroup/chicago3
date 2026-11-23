"""Chicago 3.0 主流程（規劃書 §3.5）：① 選基金 → ② 資料 → ③ 現況 → ④ Optimize → ⑤ 參數 → ⑥ 執行 → ⑦ 結果。"""
import io
import json
import os
import tempfile
import time
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # 用 repo 裡的 chicago3（見 streamlit_app.py）
import streamlit as st

from chicago3 import webapp
from chicago3.holdings import SOURCES
from chicago3.pipeline import ReconcileError, fund_bucket_weights, load_market, reconcile, run_all, write_report
from chicago3.storage import open_storage
from chicago3.sync import latest_holdings, load_registry, materialize, read_manifest, sync_holdings

REPO = Path(__file__).resolve().parents[1]
UPLOAD = "__upload__"
COLORS = webapp.BUCKET_COLORS
INK2, GRID = "#52514e", "#e1e0d9"


# ---------- 環境 ----------
def _secret(key, default=None):
    try:
        return st.secrets.get(key, default)
    except Exception:          # 本機沒有 secrets.toml
        return default


def get_store():
    spec = os.environ.get("CHICAGO3_STORE")
    if spec:
        return spec, open_storage(spec)
    g = dict(_secret("google", {}) or {})
    if g.get("refresh_token"):
        return "gdrive", open_storage("gdrive", g)
    return None, None


def is_admin() -> bool:
    pw = os.environ.get("CHICAGO3_ADMIN_PASSWORD") or _secret("app_password", "")
    return bool(pw) and st.session_state.get("admin_pw") == pw


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_materialize(spec: str, stamp: str, fund: str | None, hdate: str | None):
    """stamp（manifest 更新時間）或持股日期變了才重新下載。"""
    _, store = get_store()
    return materialize(store, Path(tempfile.mkdtemp(prefix="c3_")), fund)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_market(key: str, _cfg):
    return load_market(_cfg)


def pct(df: pd.DataFrame, cols=None):
    cols = cols or [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    return df.style.format({c: "{:.2%}" for c in cols})


# ---------- 圖 ----------
def bar_compare(stats: pd.DataFrame):
    d = stats[["指標權重", "基金權重"]].reset_index(names="桶").melt("桶", var_name="組合", value_name="權重").dropna()
    return alt.Chart(d).mark_bar(cornerRadiusEnd=4, size=12).encode(
        y=alt.Y("桶:N", sort=list(stats.index), title=None), x=alt.X("權重:Q", axis=alt.Axis(format="%"), title="權重"),
        yOffset="組合:N", color=alt.Color("組合:N", scale=alt.Scale(range=COLORS[:2]), legend=alt.Legend(orient="top")),
        tooltip=["桶", "組合", alt.Tooltip("權重:Q", format=".2%")]).properties(height=320)


def lines(df: pd.DataFrame, y_title: str, fmt="%"):
    d = df.reset_index(names="日期").melt("日期", var_name="桶", value_name="值")
    return alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=alt.X("日期:T", title=None), y=alt.Y("值:Q", axis=alt.Axis(format=fmt), title=y_title),
        color=alt.Color("桶:N", scale=alt.Scale(domain=list(df.columns), range=COLORS[:len(df.columns)]),
                        legend=alt.Legend(orient="top")),
        tooltip=["日期:T", "桶", alt.Tooltip("值:Q", format=".2%")]).properties(height=320).interactive(bind_y=False)


def heatmap(corr: pd.DataFrame):
    d = corr.reset_index(names="列").melt("列", var_name="欄", value_name="相關")
    base = alt.Chart(d).encode(x=alt.X("欄:N", sort=list(corr.columns), title=None),
                               y=alt.Y("列:N", sort=list(corr.index), title=None))
    return (base.mark_rect().encode(color=alt.Color("相關:Q", scale=alt.Scale(domain=[0, 1], range=["#e6f0fc", "#184f95"]),
                                                    legend=alt.Legend(title="相關係數")), tooltip=["列", "欄", alt.Tooltip("相關:Q", format=".2f")])
            + base.mark_text(fontSize=11).encode(text=alt.Text("相關:Q", format=".2f"),
                                                 color=alt.condition("datum.相關 > 0.6", alt.value("white"), alt.value("#0b0b0b")))
            ).properties(height=320)


def pie(w: pd.Series):
    d = pd.DataFrame(dict(桶=w.index, 權重=w.values))
    d["標籤"] = [f"{b} {x:.1%}" for b, x in zip(d.桶, d.權重)]
    base = alt.Chart(d).encode(theta=alt.Theta("權重:Q", stack=True),
                               color=alt.Color("桶:N", sort=list(w.index), scale=alt.Scale(domain=list(w.index), range=COLORS[:len(w)]),
                                               legend=None), order=alt.Order("權重:Q", sort="descending"))
    return (base.mark_arc(outerRadius=120, stroke="#fcfcfb", strokeWidth=2, tooltip=True)
            + base.mark_text(radius=150, fontSize=12, color="#0b0b0b").encode(text="標籤:N")).properties(height=360)


def bands_chart(t: pd.DataFrame):
    d = t.reset_index(names="桶")
    y = alt.Y("桶:N", sort=list(t.index), title=None)
    band = alt.Chart(d).mark_bar(color="#b7d3f6", size=16).encode(y=y, x=alt.X("下限:Q", axis=alt.Axis(format="%"), title="權重（股票部位）"), x2="上限:Q",
                                                                  tooltip=["桶", alt.Tooltip("下限:Q", format=".1%"), alt.Tooltip("上限:Q", format=".1%")])
    pol = alt.Chart(d).mark_tick(color=COLORS[0], thickness=3, size=20).encode(y=y, x="政策目標:Q", tooltip=[alt.Tooltip("政策目標:Q", format=".1%")])
    layers = band + pol
    if "基金現況" in d:
        layers += alt.Chart(d).mark_circle(color=COLORS[1], size=90, opacity=1).encode(
            y=y, x="基金現況:Q", tooltip=["桶", alt.Tooltip("基金現況:Q", format=".1%"), "狀態"])
    return layers.properties(height=300)


def frontier_chart(f: pd.DataFrame, pts: dict, budget: float):
    names = {"benchmark": "指標", "neutral": "可投資中性", "policy": "政策組合", "fund": "基金現況"}
    line = alt.Chart(f).mark_line(color=COLORS[0], strokeWidth=2).encode(
        x=alt.X("risk:Q", axis=alt.Axis(format="%"), title="追蹤誤差（對指標）"), y=alt.Y("ret:Q", axis=alt.Axis(format="%"), title="預期主動報酬／年"))
    p = pd.DataFrame([dict(組合=names.get(k, k), TE=v["te"], 預期主動=v["expected_active"], Beta=v["beta"]) for k, v in pts.items()])
    dots = alt.Chart(p).mark_circle(size=90, color="#0b0b0b", opacity=1).encode(
        x="TE:Q", y="預期主動:Q", tooltip=["組合", alt.Tooltip("TE:Q", format=".2%"), alt.Tooltip("預期主動:Q", format=".2%"), alt.Tooltip("Beta:Q", format=".2f")])
    text = dots.mark_text(align="left", dx=8, dy=-8, fontSize=12, color="#0b0b0b").encode(text="組合:N")
    rule = alt.Chart(pd.DataFrame(dict(x=[budget]))).mark_rule(strokeDash=[4, 4], color=INK2).encode(x="x:Q")
    return (line + rule + dots + text).properties(height=360)


# ---------- 頁面 ----------
st.title("Chicago 3.0 — 股票基金政策組合")
st.caption("Black–Litterman ＋ 追蹤誤差最佳化，三層政策框架。教學與研究用途；模型輸出須經投資委員會核准，不構成投資建議。")

spec, store = get_store()
with st.sidebar:
    st.text_input("管理密碼（寫入雲端硬碟需要）", type="password", key="admin_pw")
    st.caption("已解鎖：可寫入雲端硬碟" if is_admin() else "未解鎖：可以分析，但不會寫入雲端硬碟")
    st.caption(f"資料來源：{spec or '未設定'}")

if store is None:
    st.error("找不到資料儲存位置：請在 Secrets 設定 [google] 憑證（含 refresh_token），或設定環境變數 CHICAGO3_STORE。")
    st.stop()
man = read_manifest(store)
if not man.get("snapshot") or not man.get("tri_through") or not man.get("prices_through") \
        or not store.exists("_market/prices.parquet"):
    st.warning("雲端硬碟 chicago3.0/_market 還沒有足夠的市場資料（需要快照、報酬指數與至少一部分個股股價）。"
               "同步開始後幾分鐘就會有報酬指數；股價每存一次檔就能先用部分資料測試。請稍後重新整理。")
    st.stop()

# ① 選基金
st.header("① 選擇基金")
funds = load_registry(REPO / "funds.yaml")
options = [f["code"] for f in funds] + [UPLOAD]
label = {f["code"]: f"{f['code']} {f['name']}" for f in funds} | {UPLOAD: "其他（上傳持股 CSV）"}
code = st.selectbox("股票基金", options, format_func=label.get, key="fund")
fund = next((f for f in funds if f["code"] == code), None)
c1, c2, c3 = st.columns(3)
c1.metric("市場資料日期", man["snapshot"])
c2.metric("個股檔數", len(man.get("prices_through", {})))
hdate = latest_holdings(store, code) if fund else None
c3.metric("已存持股日期", hdate or "—")
if st.session_state.get("loaded_for") != code:          # 換基金就清掉後面的步驟
    for k in ("data", "overview", "result", "stage"):
        st.session_state.pop(k, None)

# ② 資料
st.header("② 收集資料")
upload_df, upload_nav = None, None
if fund is None:
    up = st.file_uploader("持股 CSV（至少 code、weight_pct 兩欄；weight_pct 用百分比）", type="csv")
    upload_nav = st.number_input("基金淨資產（億元）", min_value=1.0, value=500.0, step=10.0)
    if up is not None:
        try:
            upload_df = webapp.parse_uploaded_holdings(pd.read_csv(up, dtype={"code": str}))
            st.caption(f"讀到 {len(upload_df)} 列、權重合計 {upload_df.weight_pct.sum():.1f}%")
        except Exception as e:
            st.error(str(e))
if st.button("載入資料（已下載的會跳過）", type="primary", disabled=fund is None and upload_df is None):
    with st.status("準備資料…", expanded=True) as s:
        meta = {}
        if fund is not None and (hdate is None or hdate < man["snapshot"]):
            s.write(f"下載 {code} 最新持股…")
            try:
                if is_admin():
                    hdate = sync_holdings(store, fund, on=man["snapshot"])
                    s.write(f"已存到雲端硬碟 chicago3.0/{code}/holdings/{hdate}.csv")
                else:
                    s.write("未解鎖管理密碼：持股只在這次使用，不寫入雲端硬碟")
            except Exception as e:
                s.write(f"持股下載失敗（{type(e).__name__}），改用已存的持股")
        s.write("從雲端硬碟讀取市場資料…")
        paths = dict(cached_materialize(spec, man.get("updated_at", ""), code if fund else None, hdate))
        if fund is not None and "holdings_file" not in paths:
            src = SOURCES[fund["source"]]
            h = src.fetch_holdings(code, fund.get("source_id")) if fund["source"] == "ezmoney" else \
                src.fetch_holdings(code, man["snapshot"], fund.get("source_id"))
            tmp = Path(tempfile.mkdtemp()) / f"{code}.csv"
            h.to_csv(tmp)
            paths["holdings_file"], paths["holdings_meta"] = str(tmp), dict(nav=h.nav, date=h.date)
        if upload_df is not None:
            tmp = Path(tempfile.mkdtemp()) / "upload.csv"
            upload_df.to_csv(tmp, index=False)
            paths["holdings_file"], paths["holdings_meta"] = str(tmp), dict(nav=upload_nav * 1e8, date="上傳")
        meta = paths.get("holdings_meta", {})
        nav = meta.get("nav") or (upload_nav * 1e8 if upload_nav else None)
        tpl = webapp.template_dict()
        asof = date.fromisoformat(man["snapshot"])
        prm = webapp.default_params(tpl)
        finfo = dict(code=code if fund else "UPLOAD", name=fund["name"] if fund else "上傳持股", nav=nav)
        cfg = webapp.build_config(tpl, finfo, paths, prm, asof)
        s.write("載入 400 檔個股、建立各桶報酬…")
        md = cached_market(json.dumps([paths["prices_file"], paths["snapshot_dir"], man.get("updated_at", "")]), cfg)
        peak, trough, dd = webapp.worst_drawdown(md.panel.rB.loc[pd.Timestamp(cfg.dates.estimation[0]):])
        st.session_state.update(data=dict(paths=paths, fund=finfo, holdings_date=meta.get("date") or hdate, md=md, tpl=tpl, prm=prm,
                                          asof=asof, stress=(peak.date(), trough.date(), dd),
                                          coverage=webapp.coverage(md, cfg, man["snapshot"])),
                                loaded_for=code)
        try:
            st.session_state["recon"] = reconcile(cfg, md)
        except ReconcileError as e:
            st.session_state["recon"] = None
            st.error(f"勾稽失敗，資料可能有問題：{e}")
        s.update(label="資料準備完成", state="complete")

data = st.session_state.get("data")
if not data:
    st.stop()
f = data["fund"]
st.success(f"{f['code']} {f['name']}｜持股日期 {data['holdings_date']}｜淨資產 {f['nav'] / 1e8:,.0f} 億元｜市場資料 {data['asof']}")
cv = data["coverage"]
if not cv["complete"]:
    st.warning(f"⚠️ 資料不完整，結果僅供測試：股價目前有 {cv['n']}／{cv['total']} 檔（占股票池市值 {cv['mcap_share']:.1%}），"
               f"最後日期 {cv['last']}。同步完成後重新整理、再按一次「載入資料」即可用完整資料。")
if st.session_state.get("recon") is not None:
    with st.expander("②′ 勾稽：重建指標 vs 官方加權報酬指數", expanded=False):
        st.dataframe(pct(st.session_state["recon"].rename(columns=dict(period="期間", official="官方", rebuilt="重建", diff="差距", corr="相關")),
                         ["官方", "重建", "差距"]), hide_index=True)

# ③ 現況
st.header("③ 各桶現況")
base_cfg = webapp.build_config(data["tpl"], f, data["paths"], data["prm"], data["asof"])
wF = fund_bucket_weights(base_cfg, data["md"].snapshot)
ov = webapp.overview(data["md"], base_cfg, wF)
t1, t2, t3, t4 = st.tabs(["基金配置", "歷史", "預期報酬與波動", "相關係數"])
with t1:
    st.altair_chart(bar_compare(ov["stats"]), width="stretch")
with t2:
    st.altair_chart(lines(ov["cum"], "累積報酬"), width="stretch")
    st.altair_chart(lines(ov["rolling_vol"], "滾動 60 日年化波動"), width="stretch")
with t3:
    st.dataframe(pct(ov["stats"], [c for c in ov["stats"].columns if c != "Beta"]).format({"Beta": "{:.2f}"}))
    st.caption("預期報酬用「均衡預期超額報酬 Π」：市場是對的時候，各桶承擔風險應得的補償。歷史平均只供參考，不能當成預期（階段 5 第 2 課）。")
with t4:
    st.altair_chart(heatmap(ov["corr"]), width="stretch")

# ④ Optimize → ⑤ 參數
st.header("④ 最佳化")
if st.button("Optimize", type="primary"):
    st.session_state["stage"] = "params"
if st.session_state.get("stage") not in ("params", "done"):
    st.stop()

st.header("⑤ 參數")
p = dict(data["prm"])
order = base_cfg.buckets.order
with st.form("params"):
    a, b, c = st.columns(3)
    p["total_te"] = a.number_input("總 TE 預算", 0.01, 0.50, p["total_te"], 0.01, format="%.2f")
    p["ir"] = b.number_input("資訊比率 IR", 0.05, 2.0, p["ir"], 0.05)
    p["erp"] = c.number_input("股票風險溢酬 ERP", 0.0, 0.20, p["erp"], 0.005, format="%.3f")
    a, b, c = st.columns(3)
    p["max_tilt"] = a.number_input("單桶最大傾斜", 0.0, 0.5, p["max_tilt"], 0.01, format="%.2f")
    p["liq_days"] = b.number_input("出清天數", 1.0, 60.0, float(p["liq_days"]), 1.0)
    p["participation"] = c.number_input("每日參與率", 0.01, 1.0, p["participation"], 0.05, format="%.2f")
    a, b, c = st.columns(3)
    p["single_cap"] = a.number_input("單一個股上限", 0.01, 1.0, p["single_cap"], 0.01, format="%.2f")
    p["tsmc_cap"] = b.number_input("台積電上限", 0.01, 1.0, p["tsmc_cap"], 0.01, format="%.2f")
    p["shares_cap"] = c.number_input("已發行股份上限", 0.01, 1.0, p["shares_cap"], 0.01, format="%.2f")
    f["nav"] = st.number_input("基金淨資產（億元，容量計算用）", 1.0, 100000.0, float(f["nav"] / 1e8), 10.0) * 1e8
    st.markdown("**策略觀點**（對指標的一年超額報酬；信心 0–1，越高越相信）")
    vdf = st.data_editor(pd.DataFrame(p["views"] or [dict(bucket=order[0], excess=0.0, confidence=0.5)]), num_rows="dynamic",
                         column_config=dict(bucket=st.column_config.SelectboxColumn("桶", options=order, required=True),
                                            excess=st.column_config.NumberColumn("年超額", format="%.3f", step=0.005),
                                            confidence=st.column_config.NumberColumn("信心", min_value=0.01, max_value=0.99, step=0.05)),
                         key="views_editor")
    with st.expander("進階"):
        a, b, c = st.columns(3)
        p["cov_method"] = a.selectbox("共變異估計", ["ledoit_wolf", "sample", "ewma"], index=["ledoit_wolf", "sample", "ewma"].index(p["cov_method"]))
        p["est_days"] = b.number_input("估計期間（日曆天）", 120, 720, int(p["est_days"]), 30)
        p["tau"] = c.number_input("τ（不影響結果，見說明）", 0.001, 1.0, p["tau"], 0.01,
                                  help="觀點不確定性設成和 τΣ 成正比時，τ 會在混合後完全消掉，真正控制觀點強弱的是信心（第 1 期發現）。")
        a, b, c = st.columns(3)
        p["te_increment"] = a.number_input("區間：TE 增量", 0.001, 0.05, p["te_increment"], 0.001, format="%.3f")
        p["band_min"] = b.number_input("區間最窄", 0.0, 0.2, p["band_min"], 0.005, format="%.3f")
        p["band_max"] = c.number_input("區間最寬", 0.01, 0.5, p["band_max"], 0.01, format="%.2f")
        a, b, c = st.columns(3)
        p["bootstrap_n"] = a.number_input("重抽樣次數", 20, 1000, int(p["bootstrap_n"]), 20)
        p["block_days"] = b.number_input("區塊天數", 5, 60, int(p["block_days"]), 5)
        s0, s1, dd = data["stress"]
        stress = c.date_input("壓力期間（預設：最大回檔）", (s0, s1))
        st.caption(f"預設壓力期間是估計期間內重建指標的最大回檔：{s0} → {s1}（{dd:.1%}）")
    run = st.form_submit_button("⑥ 執行分析", type="primary")

if run:
    p["views"] = [dict(bucket=r.bucket, excess=float(r.excess), confidence=float(r.confidence))
                  for r in vdf.itertuples() if pd.notna(r.bucket) and pd.notna(r.excess) and r.excess != 0 and pd.notna(r.confidence)]
    try:
        cfg = webapp.build_config(data["tpl"], f, data["paths"], p, data["asof"],
                                  stress=tuple(stress) if isinstance(stress, (list, tuple)) and len(stress) == 2 else None)
        with st.status("分析中…", expanded=True) as s:
            for step in ("容量與可投資中性", "均衡報酬與 Black–Litterman", "政策最佳化與區間", "重抽樣、敏感度、壓力測試", "效率前緣"):
                s.write(step)
            t0 = time.time()
            o = run_all(cfg, strict_reconcile=False)
            s.update(label=f"完成（{time.time() - t0:.0f} 秒）", state="complete")
        st.session_state.update(result=o, stage="done")
    except Exception as e:
        st.error(f"分析失敗：{type(e).__name__}：{e}")

o = st.session_state.get("result")
if not o:
    st.stop()

# ⑦ 結果
st.header("⑦ 結果")
te = o["te"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("結構 TE", f"{te['structural']:.2%}")
m2.metric("策略 TE（上限）", f"{te['strategic']:.2%}", f"上限 {te['strategic_cap']:.2%}", delta_color="off")
m3.metric("政策 TE", f"{te['policy']:.2%}")
m4.metric("戰術＋選股空間", f"{te['tactical_room']:.2%}")

left, right = st.columns([1, 1])
with left:
    st.subheader("政策目標權重")
    st.altair_chart(pie(o["wP"]), width="stretch")
with right:
    st.subheader("政策目標、區間與基金現況")
    t = pd.DataFrame({"指標": o["wB"], "可投資中性": o["wN"], "政策目標": o["wP"], "下限": o["bands"].lower, "上限": o["bands"].upper})
    if "wF" in o:
        t["基金現況"] = o["wF"]
        t["狀態"] = o["fund_status"].map({"above": "高於上限", "below": "低於下限", "": "區間內"})
    st.altair_chart(bands_chart(t), width="stretch")
st.dataframe(pct(t, [c for c in t.columns if c != "狀態"]))

if "wF" in o:
    fe, mon = o["fund_expected"], o["monitor"]
    st.info(f"基金現況：{mon['n_out']} 桶在區間外；事前 TE {o['te_fund']:.2%}、β {o['beta_fund']:.2f}；"
            f"預期主動 {fe['expected_active']:.2%} = Beta 溢酬 {fe['beta_premium']:.2%} ＋ 觀點 α {fe['view_alpha']:.2%}。"
            f"回到政策目標單邊換手約 {mon['turnover_to_policy']:.0%}，只回到區間邊界約 {mon['turnover_to_band']:.0%}。")

st.subheader("效率前緣")
st.altair_chart(frontier_chart(o["frontier"], o["frontier_points"], te["total_budget"]), width="stretch")
st.caption("判斷一個配置好不好，要看預期報酬裡有多少是觀點 α、有多少只是 Beta 溢酬（講義 3.8 節）。")

rb = o["robustness"]
x1, x2, x3 = st.tabs(["觀點敏感度", "重抽樣", "壓力測試"])
with x1:
    if "sensitivity" in rb:
        st.dataframe(pct(rb["sensitivity"].rename(columns=dict(excess="年超額", confidence="信心", te_strategic="策略TE"))), hide_index=True)
    else:
        st.caption("沒有觀點，所以沒有敏感度表。")
with x2:
    if "bootstrap" in rb:
        st.dataframe(pct(rb["bootstrap"].rename(columns=dict(policy="政策目標", p5="P5", median="中位數", p95="P95", band_lower="區間下限", band_upper="區間上限"))))
with x3:
    for name, (tab, rS) in rb.get("stress", {}).items():
        st.markdown(f"**{name}**")
        st.dataframe(pct(tab.assign(portfolio=tab.portfolio.map({"benchmark": "指標", "neutral": "可投資中性", "policy": "政策組合", "fund": "基金現況"}))
                         .rename(columns=dict(portfolio="組合", return_="期間報酬", vs_benchmark="相對指標"))), hide_index=True)

st.subheader("報告")
out_dir = Path(tempfile.mkdtemp(prefix="c3_out_"))
files = write_report(o, out_dir)
with st.expander("政策組合書（預覽）"):
    st.markdown(files["book"].read_text(encoding="utf-8"))
d1, d2, d3 = st.columns(3)
d1.download_button("下載試算表", files["xlsx"].read_bytes(), files["xlsx"].name)
d2.download_button("下載政策組合書", files["book"].read_bytes(), files["book"].name)
if d3.button("存到雲端硬碟", disabled=not is_admin(), help=None if is_admin() else "需要管理密碼"):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for k, pth in files.items():
        store.write_bytes(f"{f['code']}/runs/{stamp}/{pth.name}", pth.read_bytes())
    st.success(f"已存到 chicago3.0/{f['code']}/runs/{stamp}/")
