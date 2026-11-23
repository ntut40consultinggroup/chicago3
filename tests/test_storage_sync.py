"""第 5 期：儲存層與同步（離線：本機儲存、模擬的 Google Drive、模擬的 FinMind／投信）。"""
import io
import json
import re
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from chicago3 import sync
from chicago3.holdings.base import Holdings
from chicago3.storage import GDriveStorage, LocalStorage, open_storage


# ---------- 模擬的 Drive API v3（只實作程式用到的部分） ----------
class _Resp:
    def __init__(self, js=None, content=b"", status=200):
        self._js, self.content, self.status_code = js, content, status

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeDrive:
    def __init__(self):
        self.files, self.n = {}, 0      # id → dict(name, parents, mime, data)
        self.token_calls = 0

    def _new(self, name, parents, mime, data=b""):
        self.n += 1
        fid = f"id{self.n}"
        self.files[fid] = dict(name=name, parents=parents or ["root"], mime=mime, data=data)
        return fid

    def post(self, url, data=None, json=None, params=None, files=None, headers=None, timeout=None):
        if "oauth2" in url:
            self.token_calls += 1
            return _Resp(dict(access_token="t"))
        if "upload" in url:                                   # multipart 新檔
            meta = __import__("json").loads(files["metadata"][1])
            return _Resp(dict(id=self._new(meta["name"], meta["parents"], "file", files["file"][1])))
        return _Resp(dict(id=self._new(json["name"], json.get("parents"), json["mimeType"])))

    def patch(self, url, data=None, params=None, headers=None, timeout=None):
        self.files[url.rsplit("/", 1)[1]]["data"] = data
        return _Resp({})

    def get(self, url, params=None, headers=None, timeout=None):
        if params and params.get("alt") == "media":
            return _Resp(content=self.files[url.rsplit("/", 1)[1]]["data"])
        q = params["q"]
        name = re.search(r"name='((?:[^'\\]|\\.)*)'", q)
        par = re.search(r"'([^']+)' in parents", q)
        mime = re.search(r"mimeType(!?=)'([^']+)'", q)
        hits = []
        for fid, f in self.files.items():
            if name and f["name"] != name.group(1).replace("\\'", "'"):
                continue
            if par and par.group(1) not in f["parents"]:
                continue
            if mime and ((f["mime"] == mime.group(2)) != (mime.group(1) == "=")):
                continue
            hits.append(dict(id=fid, name=f["name"], mimeType=f["mime"]))
        return _Resp(dict(files=hits))


@pytest.fixture(params=["local", "gdrive"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalStorage(tmp_path)
    return GDriveStorage("c", "s", "r", session=FakeDrive())


def test_storage_roundtrip(store):
    assert not store.exists("a/b/c.txt") and store.list("a") == []
    store.write_text("a/b/c.txt", "一")
    store.write_text("a/b/c.txt", "二")                    # 覆寫
    store.write_bytes("a/d.bin", b"\x00\x01")
    assert store.read_text("a/b/c.txt") == "二" and store.read_bytes("a/d.bin") == b"\x00\x01"
    assert store.list("a") == ["b", "d.bin"] and store.list("a/b") == ["c.txt"]
    assert store.exists("a/b/c.txt") and not store.exists("a/x.txt")
    with pytest.raises((FileNotFoundError, OSError)):
        store.read_bytes("nope/x")


def test_gdrive_reuses_root_and_token():
    fd = FakeDrive()
    g = GDriveStorage("c", "s", "r", session=fd)
    g.write_text("x/1.txt", "1")
    g2 = GDriveStorage("c", "s", "r", session=fd)            # 另一次執行：要找到同一個 chicago3.0
    g2.write_text("x/2.txt", "2")
    roots = [f for f in fd.files.values() if f["name"] == "chicago3.0"]
    assert len(roots) == 1 and g2.list("x") == ["1.txt", "2.txt"]
    assert fd.token_calls == 2                                # 每個實例只換一次 token


def test_open_storage(tmp_path, monkeypatch):
    assert isinstance(open_storage(f"local:{tmp_path}"), LocalStorage)
    for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError):
        open_storage("gdrive")
    assert isinstance(open_storage("gdrive", dict(client_id="a", client_secret="b", refresh_token="c")), GDriveStorage)


# ---------- 增量同步 ----------
class FakeFinMind:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, dataset, code, start, end, token=None):
        self.calls.append((dataset, code, start, end))
        if code in self.fail:
            raise sync.finmind.FetchError(code)
        days = pd.bdate_range(start, end)
        if dataset == "TaiwanStockPrice":
            return pd.DataFrame(dict(date=days.strftime("%Y-%m-%d"), stock_id=code, close=100.0, spread=0.0, Trading_money=1e8))
        if dataset == "TaiwanStockTotalReturnIndex":
            return pd.DataFrame(dict(date=days.strftime("%Y-%m-%d"), stock_id="TAIEX", price=1000.0))
        return pd.DataFrame()


def test_sync_prices_incremental(tmp_path):
    st, f = LocalStorage(tmp_path), FakeFinMind()
    r = sync.sync_prices(st, ["1101", "2330"], "2026-01-01", "2026-01-31", fetch=f, log=lambda *_: None)
    assert r["updated"] == 2 and r["failed"] == []
    f.calls.clear()
    r = sync.sync_prices(st, ["1101", "2330", "2454"], "2026-01-01", "2026-02-06", fetch=f, log=lambda *_: None)
    starts = {c: s for d, c, s, e in f.calls if d == "TaiwanStockPrice"}
    assert starts == {"1101": "2026-02-01", "2330": "2026-02-01", "2454": "2026-01-01"}   # 舊的只補缺，新的從頭
    assert st.exists("_market/dividends.parquet")            # 沒有除權息也要有檔
    p = pd.read_parquet(io.BytesIO(st.read_bytes("_market/prices.parquet")))
    assert not p.duplicated(["stock_id", "date"]).any() and p.date.max() == "2026-02-06"
    f.calls.clear()
    r = sync.sync_prices(st, ["1101"], "2026-01-01", "2026-02-06", fetch=f, log=lambda *_: None)
    assert r["skipped"] == 1 and f.calls == []                                              # 已經是最新就不下載


def test_sync_prices_failure_not_marked(tmp_path):
    st = LocalStorage(tmp_path)
    r = sync.sync_prices(st, ["1101", "9999"], "2026-01-01", "2026-01-10", fetch=FakeFinMind(fail={"9999"}),
                         log=lambda *_: None)
    man = sync.read_manifest(st)
    assert r["failed"] == ["9999"] and "9999" not in man["prices_through"] and man["prices_failed"] == ["9999"]


def test_sync_tri_incremental(tmp_path):
    st, f = LocalStorage(tmp_path), FakeFinMind()
    assert sync.sync_tri(st, "2026-01-01", "2026-01-09", fetch=f) == 7
    assert sync.sync_tri(st, "2026-01-01", "2026-01-09", fetch=f) == 0
    assert sync.sync_tri(st, "2026-01-01", "2026-01-16", fetch=f) == 5
    assert f.calls[-1][2] == "2026-01-10"


# ---------- 持股 ----------
def _h(code, d):
    t = pd.DataFrame(dict(type=["stock"], code=["2330"], name=["台積電"], shares=[1.0], amount=[1.0], weight_pct=[10.0]))
    return Holdings(fund=code, date=d, table=t, nav=1e10, source="fake")


def test_sync_holdings_ezmoney_and_fallback(tmp_path, monkeypatch):
    st = LocalStorage(tmp_path)
    ez = type("M", (), {"fetch_holdings": staticmethod(lambda code, sid=None, session=None: _h(code, "2026-11-20"))})
    tried = []

    def fh_fetch(code, d, sid=None, session=None):
        tried.append(d)
        return _h(code, d) if d == "2026-11-20" else None   # 11/22、11/21 是週末

    fh = type("M", (), {"fetch_holdings": staticmethod(fh_fetch)})
    monkeypatch.setitem(sync.SOURCES, "ezmoney", ez)
    monkeypatch.setitem(sync.SOURCES, "fhtrust", fh)
    assert sync.sync_holdings(st, dict(code="00981A", source="ezmoney")) == "2026-11-20"
    assert sync.sync_holdings(st, dict(code="00991A", source="fhtrust"), on="2026-11-22") == "2026-11-20"
    assert tried == ["2026-11-22", "2026-11-21", "2026-11-20"]
    assert sync.latest_holdings(st, "00991A") == "2026-11-20"
    meta = json.loads(st.read_text("00991A/holdings/2026-11-20.json"))
    assert meta["nav"] == 1e10 and st.read_text("00981A/holdings/2026-11-20.csv").startswith("type,code,name,weight_pct,shares")


def test_materialize_requires_snapshot(tmp_path):
    st = LocalStorage(tmp_path)
    for n in ("prices", "dividends", "tri"):
        st.write_bytes(f"_market/{n}.parquet", b"x")
    with pytest.raises(FileNotFoundError):
        sync.materialize(st, tmp_path / "cache")


def test_registry_loads():
    from pathlib import Path
    funds = sync.load_registry(Path(__file__).parents[1] / "funds.yaml")
    assert {f["code"] for f in funds} >= {"00981A", "00991A"} and all(f["source"] in sync.SOURCES for f in funds)


def test_roc_date():
    assert sync._roc_to_iso("1151120") == "2026-11-20"
