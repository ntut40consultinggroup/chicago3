"""持股解析測試（離線，用存下來的投信網頁）。"""
from pathlib import Path

import pandas as pd

from chicago3.holdings import ezmoney, fhtrust

FX = Path(__file__).parent / "fixtures" / "holdings"
REF = Path(__file__).parent / "fixtures" / "holdings_00981A_20261120.csv"   # 課程手動整理的 11/20 持股


def test_ezmoney_fund_list():
    m = ezmoney.parse_fund_list((FX / "ezmoney_49YTW_20261120.html").read_text(encoding="utf-8"))
    assert m["00981A"] == "49YTW"


def test_ezmoney_matches_course_csv():
    h = ezmoney.parse_holdings((FX / "ezmoney_49YTW_20261120.html").read_text(encoding="utf-8"), "00981A")
    assert h.date == "2026-11-20"
    st = h.stocks()
    assert len(st) == 50
    ref = pd.read_csv(REF, dtype={"code": str})
    ref = ref[ref.type == "stock"]
    m = st.merge(ref, on="code", suffixes=("", "_ref"), validate="one_to_one")
    assert len(m) == 50
    assert (m.shares - m.shares_ref).abs().max() == 0
    assert (m.weight_pct - m.weight_pct_ref).abs().max() < 1e-9
    fut = h.table[h.table.type == "future"]
    assert list(fut.code) == ["TX"] and fut.shares.iloc[0] == 867
    assert h.nav > 1e11


def test_fhtrust_fund_list():
    m = fhtrust.parse_fund_list((FX / "fhtrust_etf_list.html").read_text(encoding="utf-8"))
    assert m["00991A"] == "ETF23"


def test_fhtrust_holdings():
    h = fhtrust.parse_holdings((FX / "fhtrust_ETF23_20261120.xlsx").read_bytes(), "00991A")
    assert h.date == "2026-11-20"
    st = h.stocks()
    assert len(st) == 50
    assert st.iloc[0].code == "2330" and st.iloc[0].shares == 4_230_000
    assert abs(st.iloc[0].weight_pct - 18.492) < 1e-9
    assert abs(h.nav - 45_750_099_134) < 1


def test_fhtrust_non_trading_day():
    assert fhtrust.parse_holdings(b"[]" * 6, "00991A") is None
