"""FinMind：個股日價、除權息結果、加權報酬指數。

檔案格式沿用 FinMind API 原始回應（{"status": 200, "data": [...]}），檔名 <資料集>_<代號>.json。
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.finmindtrade.com/api/v4/data"
JUMP = 0.105   # 非除權息日漲跌超過 10.5%（台股漲跌幅 10%）視為減資、分割等，改用參考價


def read(path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(Path(path).read_text())["data"])


def stock_returns(price: pd.DataFrame, div: pd.DataFrame):
    """個股日總報酬與市值變化率。

    - 一般日：收盤 ÷ 前收 − 1
    - 除權息日：收盤 ÷ 除權息參考價 − 1（總報酬）；市值變化另扣現金股利（現金流出基金以外），股票股利、分割不影響市值
    - 非除權息日漲跌逾 10.5%：改用「收盤 − 漲跌」當參考價（減資、分割、新上市）
    回傳 (r, rcap, flags, trading_money)，r、rcap 不含第一天；空資料回傳 None。
    """
    if price.empty:
        return None
    p = price.assign(date=pd.to_datetime(price.date)).set_index("date")
    p = p[p.close > 0]
    r = p.close / p.close.shift(1) - 1
    rcap = r.copy()
    exd = set()
    if not div.empty:
        d = div.assign(date=pd.to_datetime(div.date))
        for _, x in d.iterrows():
            if x.date in r.index and x.after_price > 0:
                r.loc[x.date] = p.close.loc[x.date] / x.after_price - 1
                cash = x.stock_and_cache_dividend if x.stock_or_cache_dividend == "息" else 0.0
                rcap.loc[x.date] = p.close.loc[x.date] / x.after_price * (1 - cash / x.before_price) - 1
        exd = set(d.date)
    jump = (r.abs() > JUMP) & ~r.index.isin(exd)
    flags = int(jump.sum())
    if flags:
        ref = p.close - p.spread
        r[jump] = (p.close / ref - 1)[jump]
        rcap[jump] = r[jump]
    return r.iloc[1:], rcap.iloc[1:], flags, p.Trading_money


def load_stock(folder, code):
    folder = Path(folder)
    fp, fd = folder / f"TaiwanStockPrice_{code}.json", folder / f"TaiwanStockDividendResult_{code}.json"
    if not fp.exists():
        return None
    return stock_returns(read(fp), read(fd) if fd.exists() else pd.DataFrame())


def load_tri(path) -> pd.Series:
    t = pd.read_parquet(path) if str(path).endswith(".parquet") else read(path)
    return pd.Series(t.price.values, pd.to_datetime(t.date)).sort_index()


class FetchError(RuntimeError):
    pass


def fetch_frame(dataset: str, data_id: str, start: str, end: str, token: str | None = None,
                session=None, retries: int = 6, wait: float = 5) -> pd.DataFrame:
    """下載一個資料集，回傳 DataFrame（可能是空的）；402（超過頻率上限）時等 60 秒重試，最後失敗丟出 FetchError。"""
    s = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for _ in range(retries):
        try:
            r = s.get(API, params=dict(dataset=dataset, data_id=data_id, start_date=start, end_date=end),
                      headers=headers, timeout=60)
            js = r.json()
        except (requests.RequestException, ValueError):
            js = {}
        if js.get("status") == 200:
            return pd.DataFrame(js["data"])
        time.sleep(60 if js.get("status") == 402 else wait)
    raise FetchError(f"{dataset} {data_id} {start}–{end} 下載失敗")


def fetch(dataset: str, data_id: str, start: str, end: str, path, token: str | None = None,
          session=None, retries: int = 6) -> bool:
    """下載一個資料集到 path（FinMind 原始 JSON 格式）。成功回傳 True。"""
    try:
        df = fetch_frame(dataset, data_id, start, end, token, session, retries)
    except FetchError:
        return False
    Path(path).write_text(json.dumps(dict(status=200, data=df.to_dict("records")), ensure_ascii=False))
    return True


def adv(trading_money: pd.Series, window: int = 60, asof=None) -> float:
    """日均成交值（元）：截至 asof 的最近 window 個交易日平均。"""
    x = trading_money if asof is None else trading_money.loc[:pd.Timestamp(asof)]
    return float(x.iloc[-window:].mean()) if len(x) else np.nan
