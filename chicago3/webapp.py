"""網頁版的邏輯（不依賴 Streamlit，可以單獨測試）：建立設定、現況總覽、預設參數。"""
import copy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import equilibrium, risk
from .config import Config

TEMPLATE = Path(__file__).resolve().parents[1] / "examples" / "00981A_2027.yaml"
BUCKET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]   # dataviz 參考色盤 1–7 槽


def template_dict(path=TEMPLATE) -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    d.pop("attribution", None)          # 網頁版沒有基金價格資料，不做回溯
    return d


def default_params(tpl: dict) -> dict:
    """參數表單的預設值（規劃書 §3.5 ⑤）。"""
    c, m = tpl["constraints"], tpl["model"]
    return dict(
        total_te=tpl["budget"]["total_te"], ir=m["ir"], erp=m["erp"], tau=m["tau"],
        cov_method=m["covariance"]["method"], est_days=365,
        single_cap=c["single_cap"], tsmc_cap=c["single_cap_overrides"].get("2330", 0.25), shares_cap=c["shares_cap"],
        liq_days=c["liquidity"]["days"], participation=c["liquidity"]["participation"], max_tilt=c["max_tilt"],
        te_increment=tpl["bands"]["te_increment"], band_min=tpl["bands"]["clip"][0], band_max=tpl["bands"]["clip"][1],
        bootstrap_n=tpl["robustness"]["bootstrap"]["n"], block_days=tpl["robustness"]["bootstrap"]["block_days"],
        views=[dict(bucket=v["bucket"], excess=v["excess"], confidence=v["confidence"]) for v in tpl["views"]],
    )


def build_config(tpl: dict, fund: dict, paths: dict, params: dict, asof: date, stress: tuple | None = None) -> Config:
    """fund：code、name、nav；paths：materialize 的回傳值（prices_file…、holdings_file 可無）。"""
    d = copy.deepcopy(tpl)
    d["fund"] = dict(code=fund["code"], name=fund.get("name", ""), nav=float(fund["nav"]))
    d["data"] = dict(root="/", prices_file=paths["prices_file"], dividends_file=paths["dividends_file"],
                     tri_file=paths["tri_file"], snapshot_dir=paths["snapshot_dir"],
                     holdings_file=paths.get("holdings_file"), market="上市",
                     universe_top_n=tpl["data"].get("universe_top_n", 400))
    est0 = asof - timedelta(days=int(params["est_days"]))
    d["dates"] = dict(asof=asof, estimation=[est0, asof], reconcile=[[est0, asof]])
    c = d["constraints"]
    c.update(single_cap=params["single_cap"], shares_cap=params["shares_cap"], max_tilt=params["max_tilt"],
             single_cap_overrides={"2330": params["tsmc_cap"]})
    c["liquidity"].update(days=params["liq_days"], participation=params["participation"])
    c["bucket_upper"] = {k: params["tsmc_cap"] for k in c.get("bucket_upper", {})}
    d["model"].update(ir=params["ir"], erp=params["erp"], tau=params["tau"])
    d["model"]["covariance"]["method"] = params["cov_method"]
    d["budget"]["total_te"] = params["total_te"]
    d["views"] = [v for v in params["views"] if v.get("bucket") and v.get("excess") not in (None, 0)]
    d["bands"].update(te_increment=params["te_increment"], clip=[params["band_min"], params["band_max"]])
    rb = d["robustness"]
    rb["bootstrap"].update(n=int(params["bootstrap_n"]), block_days=int(params["block_days"]))
    vb = d["views"][0]["bucket"] if d["views"] else None
    rb["view_grid"] = dict(bucket=vb, excess=[0.0, 0.025, 0.05, 0.075, 0.10], confidence=[0.25, 0.5, 0.75]) if vb else None
    rb["stress"] = [dict(name="最大回檔", start=stress[0], end=stress[1])] if stress else []
    return Config.model_validate(d)


def worst_drawdown(r: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """期間內最大回檔：(高點日, 低點日, 回檔幅度)。當成壓力測試的預設情境。"""
    lvl = (1 + r.fillna(0)).cumprod()
    dd = lvl / lvl.cummax() - 1
    trough = dd.idxmin()
    peak = lvl.loc[:trough].idxmax()
    return peak, trough, float(dd.min())


def overview(md, cfg: Config, wF: pd.Series | None) -> dict:
    """步驟 ③：各桶現況、歷史、預期報酬與波動、相關係數（還沒最佳化前就能看）。"""
    order = cfg.buckets.order
    p = md.panel
    R = p.window(*cfg.dates.estimation)[order]
    S = risk.estimate_cov(R, cfg.model.covariance.method, cfg.model.annualization)
    wB = p.weights_at(cfg.dates.asof)
    Pi, _ = equilibrium.implied_returns(S, wB, cfg.model.erp)
    betas = pd.Series(S.values @ wB.values / (wB.values @ S.values @ wB.values), order)
    stats = pd.DataFrame({"指標權重": wB, "基金權重": wF if wF is not None else np.nan, "均衡預期超額報酬Π": Pi,
                          "年化波動": risk.vol(S), "Beta": betas,
                          "歷史平均年化報酬（僅供參考）": R.mean() * cfg.model.annualization})
    cum = (1 + p.RB[order]).cumprod() - 1
    rolling = p.RB[order].rolling(60).std() * np.sqrt(cfg.model.annualization)
    return dict(stats=stats, cum=cum, rolling_vol=rolling.dropna(how="all"), corr=risk.corr(S), S=S, wB=wB)


def parse_uploaded_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """上傳的持股 CSV：至少要有 code 與 weight_pct（%）；type 缺的當成 stock。回傳標準欄位。"""
    cols = {c.lower().strip(): c for c in df.columns}
    if "code" not in cols or "weight_pct" not in cols:
        raise ValueError("持股 CSV 至少要有 code、weight_pct 兩欄（weight_pct 用百分比，例如 18.22）")
    out = pd.DataFrame(dict(type=df[cols["type"]] if "type" in cols else "stock",
                            code=df[cols["code"]].astype(str).str.strip(),
                            name=df[cols["name"]] if "name" in cols else "",
                            weight_pct=pd.to_numeric(df[cols["weight_pct"]], errors="raise"),
                            shares=df[cols["shares"]] if "shares" in cols else 0))
    if (out.weight_pct < 0).any() or out.weight_pct.sum() > 150:
        raise ValueError("weight_pct 看起來不對：要是百分比（0–100），且加總不該超過 150")
    return out


def coverage(md, cfg: Config, snapshot_day: str) -> dict:
    """資料完整度：股票池（上市市值前 N 大）裡有股價的檔數、占市值比例、股價最後日期。"""
    snap = md.snapshot
    top = snap[snap.市場 == cfg.data.market].head(cfg.data.universe_top_n)
    have = set(md.panel.R.columns)
    share = float(top.loc[top.index.isin(have), "市值"].sum() / top.市值.sum())
    last = md.panel.R.index.max().date().isoformat()
    return dict(n=len(have & set(top.index)), total=len(top), mcap_share=share, last=last,
                complete=share > 0.999 and last >= snapshot_day)
