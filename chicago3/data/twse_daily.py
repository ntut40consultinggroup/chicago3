"""證交所網站：每日收盤行情（MI_INDEX，一天一次涵蓋全部上市股票）、除權除息計算結果表（TWT49U，一次查一段期間）。

欄位轉成和 FinMind 相同的格式，下游完全不用改。2026-11 實測：
- MI_INDEX 的收盤價、漲跌價差、成交金額與 FinMind 完全一致（11/20 400 檔、2025-06-02 385 檔）
- TWT49U 與 FinMind 除權息結果 811 筆共同紀錄完全一致
注意：www.twse.com.tw 對頻繁請求很敏感，每次請求之間要間隔幾秒；開發沙盒連不到，要在雲端執行。
"""
import time

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}
MI_INDEX = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TWT49U = "https://www.twse.com.tw/exchangeReport/TWT49U"


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("X", "").str.strip(), errors="coerce")


def roc_date(s: str) -> str:
    """'113年11月01日' → '2024-11-01'。"""
    y, rest = s.split("年")
    m, d = rest.replace("日", "").split("月")
    return f"{int(y) + 1911}-{int(m):02d}-{int(d):02d}"


def parse_mi_index(js: dict, day: str) -> pd.DataFrame:
    """每日收盤行情 → FinMind TaiwanStockPrice 欄位（date, stock_id, close, spread, Trading_money）。
    非交易日或查無資料回傳空表。"""
    cols = ["date", "stock_id", "close", "spread", "Trading_money"]
    if js.get("stat") != "OK":
        return pd.DataFrame(columns=cols)
    tb = next((t for t in js.get("tables", []) if (t.get("title") or "").find("每日收盤行情") >= 0), None)
    if tb is None or not tb.get("data"):
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(tb["data"], columns=tb["fields"])
    sign = df["漲跌(+/-)"].astype(str).str.contains("-").map({True: -1.0, False: 1.0})
    out = pd.DataFrame(dict(date=day, stock_id=df["證券代號"].astype(str).str.strip(), close=_num(df["收盤價"]),
                            spread=_num(df["漲跌價差"]).fillna(0) * sign, Trading_money=_num(df["成交金額"])))
    return out.dropna(subset=["close"])[cols]


def parse_twt49u(js: dict) -> pd.DataFrame:
    """除權除息計算結果表 → FinMind TaiwanStockDividendResult 欄位。"""
    cols = ["date", "stock_id", "before_price", "after_price", "stock_and_cache_dividend", "stock_or_cache_dividend"]
    if js.get("stat") != "OK" or not js.get("data"):
        return pd.DataFrame(columns=cols)
    t = pd.DataFrame(js["data"], columns=js["fields"])
    return pd.DataFrame(dict(date=t["資料日期"].map(roc_date), stock_id=t["股票代號"].astype(str).str.strip(),
                             before_price=_num(t["除權息前收盤價"]), after_price=_num(t["除權息參考價"]),
                             stock_and_cache_dividend=_num(t["權值+息值"]), stock_or_cache_dividend=t["權/息"].astype(str).str.strip()))[cols]


def parse_stock_day_all(rows: list, day: str) -> pd.DataFrame:
    """OpenAPI 當日快照（STOCK_DAY_ALL）→ 同樣的價格欄位（11/20 實測與 FinMind 完全一致）。"""
    df = pd.DataFrame(rows)
    out = pd.DataFrame(dict(date=day, stock_id=df["Code"].astype(str).str.strip(), close=_num(df["ClosingPrice"]),
                            spread=_num(df["Change"]).fillna(0), Trading_money=_num(df["TradeValue"])))
    return out.dropna(subset=["close"])


def _get_json(s, url, params, tries=4, wait=5):
    for k in range(tries):
        try:
            r = s.get(url, params=params, headers=UA, timeout=60)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            if k == tries - 1:
                raise
        time.sleep(wait * (k + 1))
    raise RuntimeError(f"{url} {params} 重試 {tries} 次仍失敗")


def fetch_mi_index(day: str, session=None) -> pd.DataFrame:
    s = session or requests.Session()
    return parse_mi_index(_get_json(s, MI_INDEX, dict(date=day.replace("-", ""), type="ALLBUT0999", response="json")), day)


def fetch_twt49u(start: str, end: str, session=None) -> pd.DataFrame:
    s = session or requests.Session()
    return parse_twt49u(_get_json(s, TWT49U, dict(response="json", startDate=start.replace("-", ""), endDate=end.replace("-", ""))))
