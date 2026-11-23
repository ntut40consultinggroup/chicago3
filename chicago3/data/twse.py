"""證交所、櫃買中心 OpenAPI：當日全市場快照（股數、產業、收盤、成交值、市值）。

快照決定兩件事：指標成分（上市市值前 N 大）與分桶用的產業別（上市＋上櫃，基金持股可能有上櫃股）。
"""
import json
from pathlib import Path

import pandas as pd
import requests

URLS = {
    "t187ap03_L.json": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
    "exchangeReport_STOCK_DAY_ALL.json": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
    "mopsfin_t187ap03_O.json": "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
    "tpex_mainboard_daily_close_quotes.json": "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
}
INDUSTRY = {"01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織", "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷",
            "09": "造紙", "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造", "15": "航運", "16": "觀光餐旅",
            "17": "金融保險", "18": "貿易百貨", "20": "其他", "21": "化學", "22": "生技醫療", "23": "油電燃氣",
            "24": "半導體", "25": "電腦及週邊設備", "26": "光電", "27": "通信網路", "28": "電子零組件", "29": "電子通路",
            "30": "資訊服務", "31": "其他電子", "35": "綠能環保", "36": "數位雲端", "37": "運動休閒", "38": "居家生活"}


def _f(x):
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None


def parse_snapshot(files: dict) -> pd.DataFrame:
    """files：檔名 → 已解析的 JSON（見 URLS）。回傳以代號為索引的表：
    名稱、市場、股數、產業、收盤、成交值、市值、資料日；依市值由大到小排序。"""
    meta = {r["公司代號"]: ("上市", _f(r["已發行普通股數或TDR原股發行股數"]), r["產業別"]) for r in files["t187ap03_L.json"]}
    meta.update({r["SecuritiesCompanyCode"]: ("上櫃", _f(r["IssueShares"]), r["SecuritiesIndustryCode"])
                 for r in files["mopsfin_t187ap03_O.json"]})
    rows = [(r["Code"], r.get("Name", ""), _f(r["ClosingPrice"]), _f(r["TradeValue"]), r["Date"])
            for r in files["exchangeReport_STOCK_DAY_ALL.json"]]
    rows += [(r["SecuritiesCompanyCode"], r.get("CompanyName", ""), _f(r["Close"]), _f(r["TransactionAmount"]), r["Date"])
             for r in files["tpex_mainboard_daily_close_quotes.json"]]
    q = pd.DataFrame(rows, columns=["代號", "名稱", "收盤", "成交值", "資料日"])
    q = q[q.代號.isin(meta)].dropna(subset=["收盤"]).copy()
    q["市場"] = q.代號.map(lambda c: meta[c][0])
    q["股數"] = q.代號.map(lambda c: meta[c][1])
    q["產業"] = q.代號.map(lambda c: INDUSTRY.get(meta[c][2], meta[c][2]))
    q["市值"] = q.收盤 * q.股數
    return q.sort_values("市值", ascending=False).set_index("代號")


def load_snapshot(folder) -> pd.DataFrame:
    d = Path(folder)
    return parse_snapshot({fn: json.loads((d / fn).read_text()) for fn in URLS})


def fetch_snapshot(folder, session=None) -> Path:
    """下載當日快照到 folder（四個 JSON 檔）。"""
    s = session or requests.Session()
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    for fn, url in URLS.items():
        r = s.get(url, timeout=90)
        r.raise_for_status()
        (d / fn).write_bytes(r.content)
    return d
