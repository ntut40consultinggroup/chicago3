"""統一投信（ezmoney）ETF 持股。

基金頁 /ETF/Fund/Info?FundCode=<內部代碼> 的 HTML 內嵌兩段 JSON：
- div#DataAsset：最新一個交易日的完整投資組合（股票、期貨、現金…）
- div#DataFundList：所有 ETF 的證券代號與內部代碼對照
限制：只提供最新一天，沒有歷史；歷史要靠每天排程存檔累積。
"""
import html
import json
import re

import pandas as pd
import requests

from .base import COLUMNS, Holdings

BASE = "https://www.ezmoney.com.tw"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}
TYPE = {"ST": "stock", "GD": "future"}


def _div_json(page: str, div_id: str):
    m = re.search(r'<div id="%s" data-content="([^"]*)"' % div_id, page)
    if not m:
        raise ValueError(f"頁面裡找不到 {div_id}，網站可能改版")
    return json.loads(html.unescape(m.group(1)))


def parse_fund_list(page: str) -> dict:
    """回傳 {證券代號: 內部代碼}，例如 {'00981A': '49YTW'}。"""
    return {f["sStockNo"].strip(): f["sFundCode"] for f in _div_json(page, "DataFundList") if f.get("sStockNo")}


def parse_holdings(page: str, fund: str) -> Holdings:
    groups = _div_json(page, "DataAsset")
    rows, date = [], None
    for g in groups:
        kind = TYPE.get(g["AssetCode"])
        if not kind:
            continue
        for d in g.get("Details") or []:
            date = date or d["TranDate"][:10]
            rows.append(dict(type=kind, code=str(d["DetailCode"]).strip(), name=d["DetailName"].strip(),
                             shares=float(d["Share"]), amount=float(d["Amount"]), weight_pct=float(d["NavRate"])))
    nav = next((float(g["Value"]) for g in groups if g["AssetCode"] == "NAV"), None)
    other = {g["AssetName"]: float(g["Value"]) for g in groups if g["AssetCode"] not in ("ST", "GD")}
    return Holdings(fund=fund, date=date, table=pd.DataFrame(rows, columns=COLUMNS), nav=nav,
                    source="ezmoney", extra=dict(other_assets=other))


def fetch_holdings(fund: str, fund_code: str | None = None, session=None) -> Holdings:
    s = session or requests.Session()
    if fund_code is None:
        fund_code = fetch_fund_list(s)[fund]
    r = s.get(f"{BASE}/ETF/Fund/Info", params=dict(FundCode=fund_code), headers=UA, timeout=30)
    r.raise_for_status()
    return parse_holdings(r.text, fund)


def fetch_fund_list(session=None) -> dict:
    s = session or requests.Session()
    r = s.get(f"{BASE}/ETF/Fund/Info", params=dict(FundCode="49YTW"), headers=UA, timeout=30)
    r.raise_for_status()
    return parse_fund_list(r.text)
