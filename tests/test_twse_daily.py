"""證交所每日資料：解析（用雲端實際抓到的回應精簡版）與同步邏輯（模擬）。"""
import io
import json
from pathlib import Path

import pandas as pd

from chicago3 import sync
from chicago3.data import twse_daily
from chicago3.storage import LocalStorage

FX = Path(__file__).parent / "fixtures"


def test_parse_mi_index_matches_finmind_fields():
    df = twse_daily.parse_mi_index(json.loads((FX / "twse_mi_index_20261120_small.json").read_text()), "2026-11-20")
    ref = pd.read_csv(FX / "holdings_00981A_20261120.csv", dtype={"code": str})   # 只用來確認代號格式
    assert set(df.columns) == {"date", "stock_id", "close", "spread", "Trading_money"}
    assert {"2330", "2454"} <= set(df.stock_id) and df.date.unique().tolist() == ["2026-11-20"]
    assert (df.close > 0).all() and df.stock_id.str.fullmatch(r"\d+").all() and len(ref)


def test_parse_mi_index_non_trading_day():
    assert twse_daily.parse_mi_index({"stat": "很抱歉，沒有符合條件的資料!"}, "2026-11-22").empty


def test_parse_twt49u():
    df = twse_daily.parse_twt49u(json.loads((FX / "twse_twt49u_small.json").read_text()))
    assert len(df) > 0 and df.date.str.match(r"2026-\d\d-\d\d").all()
    assert (df.after_price < df.before_price).all() and df.stock_or_cache_dividend.isin(["息", "權", "權息"]).all()
    assert twse_daily.roc_date("113年11月01日") == "2024-11-01"


def test_parse_stock_day_all():
    rows = [dict(Code="2330", ClosingPrice="1,500.00", Change="-30.0000", TradeValue="1000"),
            dict(Code="9999", ClosingPrice="", Change="", TradeValue="0")]
    df = twse_daily.parse_stock_day_all(rows, "2026-11-20")
    assert df.stock_id.tolist() == ["2330"] and df.close.iloc[0] == 1500 and df.spread.iloc[0] == -30


def _store_with_calendar(tmp_path, days, snapshot=None):
    st = LocalStorage(tmp_path)
    sync._write_parquet(st, "_market/tri.parquet", pd.DataFrame(dict(date=days, stock_id="TAIEX", price=1.0)))
    if snapshot:
        d, rows = snapshot
        st.write_text(f"_market/snapshot/{d.replace('-', '')}/exchangeReport_STOCK_DAY_ALL.json", json.dumps(rows))
        sync._write_manifest(st, dict(snapshot=d))
    return st


def test_sync_prices_twse_incremental_and_snapshot(tmp_path):
    days = ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]
    snap = [dict(Code=c, ClosingPrice="10", Change="0", TradeValue="1") for c in ("2330", "2454", "1101")]
    st = _store_with_calendar(tmp_path, days, ("2026-11-20", snap))
    asked, divq = [], []

    def fday(d):
        asked.append(d)
        return pd.DataFrame(dict(date=d, stock_id=["2330", "2454", "1101", "0050"], close=1.0, spread=0.0, Trading_money=1.0))

    def fdiv(a, b):
        divq.append((a, b))
        return pd.DataFrame(dict(date=["2026-11-18"], stock_id=["2330"], before_price=10.0, after_price=9.0,
                                 stock_and_cache_dividend=1.0, stock_or_cache_dividend="息"))

    r = sync.sync_prices_twse(st, ["2330", "2454"], "2026-11-16", "2026-11-20", fetch_day=fday, fetch_divs=fdiv,
                              pause=0, log=lambda *_: None)
    assert asked == days[:-1]                         # 最新一天用快照，不再查
    assert r["updated_days"] == 5 and divq == [("2026-11-16", "2026-11-20")]
    p = pd.read_parquet(io.BytesIO(st.read_bytes("_market/prices.parquet")))
    assert set(p.stock_id) == {"2330", "2454"} and len(p) == 10       # 只留股票池裡的代號
    man = sync.read_manifest(st)
    assert man["prices_through"] == {"2330": "2026-11-20", "2454": "2026-11-20"} and man["dividends_through"] == "2026-11-20"
    # 第二次：新加入 1101 → 只為它補（全部交易日），其他不重抓；除權息已是最新也不再查
    asked.clear()
    divq.clear()
    sync.sync_prices_twse(st, ["2330", "2454", "1101"], "2026-11-16", "2026-11-20", fetch_day=fday, fetch_divs=fdiv,
                          pause=0, log=lambda *_: None)
    assert asked == days[:-1] and divq == []
    assert sync.read_manifest(st)["prices_through"]["1101"] == "2026-11-20"


def test_sync_prices_twse_checkpoint_resume(tmp_path):
    days = [f"2026-11-{d:02d}" for d in (2, 3, 4, 5, 6, 9)]
    st = _store_with_calendar(tmp_path, days)
    calls = []

    def boom(d):
        calls.append(d)
        if d == "2026-11-06":
            raise KeyboardInterrupt
        return pd.DataFrame(dict(date=d, stock_id=["2330"], close=1.0, spread=0.0, Trading_money=1.0))

    try:
        sync.sync_prices_twse(st, ["2330"], days[0], days[-1], fetch_day=boom, fetch_divs=lambda a, b: pd.DataFrame(),
                              pause=0, checkpoint=2, log=lambda *_: None)
    except KeyboardInterrupt:
        pass
    assert sync.read_manifest(st)["prices_through"]["2330"] == "2026-11-05"     # 每 2 天存一次
    calls.clear()

    def ok(d):
        calls.append(d)
        return pd.DataFrame(dict(date=d, stock_id=["2330"], close=1.0, spread=0.0, Trading_money=1.0))

    sync.sync_prices_twse(st, ["2330"], days[0], days[-1], fetch_day=ok, fetch_divs=lambda a, b: pd.DataFrame(),
                          pause=0, checkpoint=2, log=lambda *_: None)
    assert calls == ["2026-11-06", "2026-11-09"]                                 # 從斷點接著補


def test_run_sync_auto_falls_back_to_finmind(tmp_path, monkeypatch):
    st = LocalStorage(tmp_path)
    sync._write_manifest(st, dict(snapshot="2026-11-20"))
    monkeypatch.setattr(sync, "universe_from_snapshot", lambda store, day, n: ["2330"])
    monkeypatch.setattr(sync, "sync_prices_twse", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("連不上")))
    monkeypatch.setattr(sync, "sync_prices", lambda *a, **k: dict(source="finmind"))
    out = sync.run_sync(st, [], steps=("prices",), log=lambda *_: None)
    assert out["prices"]["source"] == "finmind"
