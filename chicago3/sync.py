"""資料同步：上網下載 → 存到儲存層（本機或 Google Drive）；已經有的就跳過，只補缺的日期。

儲存結構（規劃書 §3.6，市場資料改成合併檔，網頁版啟動時只要下載幾個檔）：
  _market/manifest.json            最後更新時間、各資料集最後日期、失敗清單
  _market/prices.parquet           所有個股日價（FinMind TaiwanStockPrice 欄位）
  _market/dividends.parquet        所有個股除權息結果（FinMind TaiwanStockDividendResult 欄位）
  _market/tri.parquet              加權報酬指數
  _market/snapshot/<YYYYMMDD>/     證交所／櫃買當日快照（4 個 JSON）
  <代號>/holdings/<YYYY-MM-DD>.csv 基金持股（type,code,name,weight_pct,shares）
  <代號>/holdings/<YYYY-MM-DD>.json 持股中繼資料（淨資產、來源、其他資產）
"""
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml

from .data import finmind, twse
from .holdings import SOURCES
from .storage import Storage

M = "_market"
PRICE_COLS = ["date", "stock_id", "close", "spread", "Trading_money"]
DIV_COLS = ["date", "stock_id", "before_price", "after_price", "stock_and_cache_dividend", "stock_or_cache_dividend"]


# ---------- 小工具 ----------
def _read_parquet(store: Storage, path: str, cols: list[str]) -> pd.DataFrame:
    if not store.exists(path):
        return pd.DataFrame(columns=cols)
    return pd.read_parquet(io.BytesIO(store.read_bytes(path)))


def _write_parquet(store: Storage, path: str, df: pd.DataFrame):
    buf = io.BytesIO()
    df.reset_index(drop=True).to_parquet(buf, index=False)
    store.write_bytes(path, buf.getvalue())


def read_manifest(store: Storage) -> dict:
    p = f"{M}/manifest.json"
    return json.loads(store.read_text(p)) if store.exists(p) else {}


def _write_manifest(store: Storage, man: dict):
    man["updated_at"] = datetime.now().isoformat(timespec="seconds")
    store.write_text(f"{M}/manifest.json", json.dumps(man, ensure_ascii=False, indent=1, default=str))


def _roc_to_iso(s: str) -> str:
    """'1151120'（民國年）→ '2026-11-20'。"""
    s = str(s).strip()
    return f"{int(s[:-4]) + 1911}-{s[-4:-2]}-{s[-2:]}"


def _get_with_retry(s, url: str, tries: int = 4, wait: float = 5) -> bytes:
    """公開 API 偶爾會斷線（connection reset、5xx），重試幾次再放棄。"""
    import time
    for k in range(tries):
        try:
            r = s.get(url, timeout=90)
            if r.status_code < 500:
                r.raise_for_status()
                return r.content
        except requests.ConnectionError:
            if k == tries - 1:
                raise
        time.sleep(wait * (k + 1))
    raise RuntimeError(f"{url} 重試 {tries} 次仍失敗")


# ---------- 快照 ----------
def sync_snapshot(store: Storage, session=None) -> str:
    """下載今天的證交所／櫃買快照；同一天已經有就不重抓。回傳快照日期（YYYY-MM-DD）。"""
    s = session or requests.Session()
    files = {fn: _get_with_retry(s, url) for fn, url in twse.URLS.items()}
    d = _roc_to_iso(json.loads(files["exchangeReport_STOCK_DAY_ALL.json"])[0]["Date"])
    folder = f"{M}/snapshot/{d.replace('-', '')}"
    if not store.exists(f"{folder}/exchangeReport_STOCK_DAY_ALL.json"):
        for fn, b in files.items():
            store.write_bytes(f"{folder}/{fn}", b)
    man = read_manifest(store)
    man["snapshot"] = max(d, man.get("snapshot", ""))
    _write_manifest(store, man)
    return d


def latest_snapshot(store: Storage) -> str | None:
    days = [x for x in store.list(f"{M}/snapshot") if x.isdigit()]
    return max(days) if days else None


# ---------- 個股日價與除權息（增量） ----------
def sync_prices(store: Storage, codes: list[str], start: str, end: str, token: str | None = None,
                fetch=finmind.fetch_frame, log=print) -> dict:
    """每檔只下載「最後日期之後」到 end 的資料；新加入的代號從 start 開始抓。回傳摘要。"""
    prices = _read_parquet(store, f"{M}/prices.parquet", PRICE_COLS)
    divs = _read_parquet(store, f"{M}/dividends.parquet", DIV_COLS)
    man = read_manifest(store)
    through = man.get("prices_through", {})          # 代號 → 已抓到哪一天（價格與除權息一起）
    new_p, new_d, failed, done, skipped = [], [], [], 0, 0
    for i, c in enumerate(codes):
        last = through.get(c)
        if last and last >= end:
            skipped += 1
            continue
        s = (date.fromisoformat(last) + timedelta(days=1)).isoformat() if last else start
        try:
            p = fetch("TaiwanStockPrice", c, s, end, token)
            d = fetch("TaiwanStockDividendResult", c, s, end, token)
        except finmind.FetchError as e:
            failed.append(c)
            log(f"失敗：{e}")
            continue
        if len(p):
            new_p.append(p.reindex(columns=PRICE_COLS))
        if len(d):
            new_d.append(d.reindex(columns=DIV_COLS))
        through[c] = end
        done += 1
        if (i + 1) % 50 == 0:
            log(f"個股進度 {i + 1}/{len(codes)}")
    if new_p:
        prices = pd.concat([prices, *new_p]).drop_duplicates(["stock_id", "date"], keep="last").sort_values(["stock_id", "date"])
        _write_parquet(store, f"{M}/prices.parquet", prices)
    if new_d:
        divs = pd.concat([divs, *new_d]).drop_duplicates(["stock_id", "date"], keep="last").sort_values(["stock_id", "date"])
    if new_d or not store.exists(f"{M}/dividends.parquet"):     # 沒有任何除權息也要有檔（可以是空的）
        _write_parquet(store, f"{M}/dividends.parquet", divs)
    man["prices_through"] = through
    man["prices_failed"] = failed
    _write_manifest(store, man)
    return dict(updated=done, skipped=skipped, failed=failed, rows=len(prices))


def sync_tri(store: Storage, start: str, end: str, token: str | None = None, fetch=finmind.fetch_frame) -> int:
    tri = _read_parquet(store, f"{M}/tri.parquet", ["date", "stock_id", "price"])
    last = tri.date.max() if len(tri) else None
    if last and last >= end:
        return 0
    s = (date.fromisoformat(last) + timedelta(days=1)).isoformat() if last else start
    new = fetch("TaiwanStockTotalReturnIndex", "TAIEX", s, end, token)
    if len(new):
        tri = pd.concat([tri, new[["date", "stock_id", "price"]]]).drop_duplicates("date", keep="last").sort_values("date")
        _write_parquet(store, f"{M}/tri.parquet", tri)
    man = read_manifest(store)
    man["tri_through"] = end
    _write_manifest(store, man)
    return len(new)


# ---------- 基金持股 ----------
def load_registry(path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["funds"]


def sync_holdings(store: Storage, fund: dict, on: str | None = None, session=None) -> str | None:
    """下載一檔基金的持股存到 <代號>/holdings/。統一只有最新一天；復華可指定日期（預設往回找最近的交易日）。
    回傳持股日期；已經存過就不重寫。"""
    src = SOURCES[fund["source"]]
    if fund["source"] == "ezmoney":
        h = src.fetch_holdings(fund["code"], fund.get("source_id"), session=session)
    else:
        day = date.fromisoformat(on) if on else date.today()
        h = None
        for k in range(10):
            h = src.fetch_holdings(fund["code"], (day - timedelta(days=k)).isoformat(), fund.get("source_id"), session=session)
            if h is not None:
                break
    if h is None:
        return None
    base = f"{fund['code']}/holdings/{h.date}"
    if not store.exists(f"{base}.csv"):
        buf = io.StringIO()
        h.table[["type", "code", "name", "weight_pct", "shares"]].to_csv(buf, index=False)
        store.write_text(f"{base}.csv", buf.getvalue())
        store.write_text(f"{base}.json", json.dumps(dict(fund=h.fund, date=h.date, nav=h.nav, source=h.source, **h.extra),
                                                    ensure_ascii=False, indent=1, default=str))
    return h.date


def latest_holdings(store: Storage, code: str) -> str | None:
    days = [x[:-4] for x in store.list(f"{code}/holdings") if x.endswith(".csv")]
    return max(days) if days else None


# ---------- 給計算用：從儲存層拉到本機快取 ----------
def materialize(store: Storage, cache, fund: str | None = None) -> dict:
    """把計算需要的檔案下載到 cache 資料夾，回傳可直接填進設定檔 data 區的路徑。"""
    cache = Path(cache)
    out = {}
    for name in ("prices", "dividends", "tri"):
        out[f"{name}_file"] = str(store.download(f"{M}/{name}.parquet", cache / f"{name}.parquet"))
    snap = latest_snapshot(store)
    if snap is None:
        raise FileNotFoundError("儲存層裡還沒有證交所快照，請先執行同步")
    for fn in twse.URLS:
        store.download(f"{M}/snapshot/{snap}/{fn}", cache / "snapshot" / snap / fn)
    out["snapshot_dir"] = str(cache / "snapshot" / snap)
    if fund:
        d = latest_holdings(store, fund)
        if d:
            out["holdings_file"] = str(store.download(f"{fund}/holdings/{d}.csv", cache / fund / f"{d}.csv"))
            out["holdings_meta"] = json.loads(store.read_text(f"{fund}/holdings/{d}.json"))
    return out


def universe_from_snapshot(store: Storage, day: str, top_n: int, market: str = "上市") -> list[str]:
    folder = f"{M}/snapshot/{day.replace('-', '')}"
    snap = twse.parse_snapshot({fn: json.loads(store.read_bytes(f"{folder}/{fn}")) for fn in twse.URLS})
    return list(snap[snap.市場 == market].index[:top_n])


def run_sync(store: Storage, registry: list[dict], start: str = "2024-11-01", top_n: int = 400,
             token: str | None = None, steps=("snapshot", "prices", "tri", "holdings"), log=print) -> dict:
    """每日同步：快照 → 市值前 top_n 大個股（增量）→ 報酬指數 → 各基金持股。單一步驟失敗不影響其他步驟。"""
    out = {}
    day = read_manifest(store).get("snapshot")
    if "snapshot" in steps:
        day = sync_snapshot(store)
        log(f"快照 {day}")
    if day is None:
        raise RuntimeError("沒有快照，無法決定股票池與日期")
    out["date"] = day
    if "prices" in steps:
        out["prices"] = sync_prices(store, universe_from_snapshot(store, day, top_n), start, day, token, log=log)
        log(f"個股：{out['prices']}")
    if "tri" in steps:
        out["tri"] = sync_tri(store, start, day, token)
        log(f"報酬指數新增 {out['tri']} 天")
    if "holdings" in steps:
        out["holdings"] = {}
        for f in registry:
            try:
                out["holdings"][f["code"]] = sync_holdings(store, f, on=day)
            except Exception as e:     # 投信網站改版或暫時連不上：記錄下來，不讓整個同步失敗
                out["holdings"][f["code"]] = f"失敗：{type(e).__name__}: {e}"
            log(f"持股 {f['code']}：{out['holdings'][f['code']]}")
    return out
