"""復華投信 ETF 持股。

- ETF 清單頁 /ETF/etf_list：每列有證券代號與詳細頁連結 /ETF/etf_detail/ETFxx
- 持股 Excel /api/assetsExcel/ETFxx/YYYYMMDD：指定日期的股票持股；非交易日回傳空檔（約 12 bytes）
  前 10 列是基金名稱、日期、淨資產、流通單位數、每單位淨值，第 11 列起是持股表。
"""
import io
import re

import pandas as pd
import requests

from .base import COLUMNS, Holdings, to_number

BASE = "https://www.fhtrust.com.tw"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}


def parse_fund_list(page: str) -> dict:
    """回傳 {證券代號: 內部代碼}，例如 {'00991A': 'ETF23'}。"""
    out = {}
    for row in re.findall(r'<tr class="fundListTable-fundCard etfListTable-fundCard".*?</tr>', page, flags=re.S):
        code = re.search(r'etfListTable-code[^>]*>\s*([0-9A-Z]+)\s*<', row)
        link = re.search(r'/ETF/etf_detail/(ETF\d+)', row)
        if code and link:
            out[code.group(1)] = link.group(1)
    return out


def parse_holdings(content: bytes, fund: str) -> Holdings | None:
    """非交易日（空檔）回傳 None。"""
    if len(content) < 100:
        return None
    x = pd.read_excel(io.BytesIO(content), header=None, dtype=str)
    date = re.search(r"(\d{4})/(\d{2})/(\d{2})", str(x.iloc[2, 0]))
    head = x.index[x[0].astype(str).str.strip() == "證券代號"][0]
    t = x.iloc[head + 1:, :5].dropna(subset=[0])
    t.columns = ["code", "name", "shares", "amount", "weight_pct"]
    table = pd.DataFrame(dict(type="stock", code=t.code.str.strip(), name=t.name.str.strip(),
                              shares=t.shares.map(to_number), amount=t.amount.map(to_number),
                              weight_pct=t.weight_pct.map(to_number)))[COLUMNS].reset_index(drop=True)
    return Holdings(fund=fund, date="-".join(date.groups()), table=table, nav=to_number(x.iloc[4, 0]),
                    source="fhtrust", extra=dict(units=to_number(x.iloc[6, 0]), nav_per_unit=to_number(x.iloc[8, 0])))


def fetch_fund_list(session=None) -> dict:
    s = session or requests.Session()
    r = s.get(f"{BASE}/ETF/etf_list", headers=UA, timeout=30)
    r.raise_for_status()
    return parse_fund_list(r.text)


def fetch_holdings(fund: str, date: str, fund_code: str | None = None, session=None) -> Holdings | None:
    """date：YYYY-MM-DD 或 YYYYMMDD。"""
    s = session or requests.Session()
    if fund_code is None:
        fund_code = fetch_fund_list(s)[fund]
    r = s.get(f"{BASE}/api/assetsExcel/{fund_code}/{date.replace('-', '')}", headers=UA, timeout=60)
    r.raise_for_status()
    return parse_holdings(r.content, fund)
