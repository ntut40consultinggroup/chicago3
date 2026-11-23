"""第 2 期：資料層與設定的單元測試（人造資料，不需要真實市場資料）。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from chicago3 import capacity
from chicago3.buckets import BucketRules
from chicago3.config import load_config
from chicago3.data.finmind import adv, stock_returns
from chicago3.panel import build_panel, period_return

EX = Path(__file__).parents[1] / "examples"


# ---------- 個股報酬 ----------
def _price(closes, spreads=None, money=None):
    n = len(closes)
    return pd.DataFrame(dict(date=pd.date_range("2026-01-01", periods=n, freq="B").strftime("%Y-%m-%d"), close=closes,
                             spread=spreads or [0.0] * n, Trading_money=money or [1e8] * n))


def test_plain_returns():
    r, rcap, flags, _ = stock_returns(_price([100, 110, 99]), pd.DataFrame())
    assert np.allclose(r.values, [0.10, -0.10]) and np.allclose(rcap, r) and flags == 0


def test_cash_dividend_day():
    p = _price([100, 98, 99])
    d = pd.DataFrame(dict(date=[p.date[1]], before_price=[100.0], after_price=[97.0], stock_and_cache_dividend=[3.0],
                          stock_or_cache_dividend=["息"]))
    r, rcap, _, _ = stock_returns(p, d)
    assert np.isclose(r.iloc[0], 98 / 97 - 1)                       # 總報酬用參考價
    assert np.isclose(rcap.iloc[0], 98 / 97 * (1 - 3 / 100) - 1)    # 市值扣掉現金股利


def test_split_like_jump_uses_reference_price():
    # 第二天收盤 40（像 1 拆 2.5），漲跌欄 +1 → 參考價 39
    r, rcap, flags, _ = stock_returns(_price([100, 40, 41], spreads=[0, 1.0, 1.0]), pd.DataFrame())
    assert flags == 1 and np.isclose(r.iloc[0], 40 / 39 - 1) and np.isclose(rcap.iloc[0], r.iloc[0])


def test_adv_window():
    s = pd.Series(np.arange(1, 101, dtype=float), pd.date_range("2026-01-01", periods=100))
    assert adv(s, 10) == np.mean(np.arange(91, 101))
    assert adv(s, 10, asof="2026-01-20") == np.mean(np.arange(11, 21))


# ---------- 分桶 ----------
def test_bucket_precedence():
    r = BucketRules(order=["T", "AI", "S", "O"], default="O", single_stock={"T": "2330"},
                    overrides={"AI": ["3017"]}, by_industry={"AI": ["電子零組件"], "S": ["半導體"]})
    assert r.bucket_of("2330", "半導體") == "T"
    assert r.bucket_of("3017", "電腦及週邊設備") == "AI"
    assert r.bucket_of("2454", "半導體") == "S"
    assert r.bucket_of("1101", "水泥") == "O"


def test_bucket_rules_validation():
    with pytest.raises(ValueError):
        BucketRules(order=["A"], default="B")
    with pytest.raises(ValueError):
        BucketRules(order=["A", "B"], default="A", by_industry={"A": ["半導體"], "B": ["半導體"]})


# ---------- 面板 ----------
def test_panel_bucket_returns_and_anchor():
    idx = pd.date_range("2026-01-01", periods=3, freq="B")
    rets = {"a": pd.Series([0.10, 0.0, 0.05], idx), "b": pd.Series([0.0, 0.20, 0.0], idx)}
    p = build_panel(rets, rets, pd.Series({"a": 300.0, "b": 100.0}), pd.Series({"a": "X", "b": "Y"}), ["X", "Y"])
    assert np.allclose(p.MC.iloc[-1], [300, 100])                    # 錨在最後一天
    assert np.isclose(p.RB.loc[idx[2], "X"], 0.05)
    w = p.MC.iloc[1] / p.MC.iloc[1].sum()                             # 第三天的指標報酬用前一天市值加權
    assert np.isclose(p.rB.iloc[2], w["a"] * 0.05)
    assert np.isclose(p.weights_at(idx[-1]).sum(), 1)
    assert np.isclose(period_return(p.rB, idx[0], idx[2]), (1 + p.rB.iloc[1]) * (1 + p.rB.iloc[2]) - 1)


# ---------- 容量 ----------
def test_capacity_min_of_three():
    t = capacity.position_capacity(nav=1000.0, mcap=pd.Series({"a": 5000.0, "b": 200.0, "c": 1e6}),
                                   adv=pd.Series({"a": 1e6, "b": 1e6, "c": 10.0}), single_cap_overrides={"a": 0.25})
    assert t.capacity.to_dict() == {"a": 250.0, "b": 20.0, "c": 20.0}
    assert t.binding.to_dict() == {"a": "nav_limit", "b": "shares_limit", "c": "liquidity_limit"}
    t3 = capacity.position_capacity(nav=1000.0, mcap=pd.Series({"c": 1e6}), adv=pd.Series({"c": 10.0}), liq_days=3)
    assert np.isclose(t3.capacity["c"], 6.0)


# ---------- 設定 ----------
def test_example_configs_load():
    for f in ("00981A_2027.yaml", "00981A_2025.yaml"):
        cfg = load_config(EX / f)
        assert cfg.fund.code == "00981A" and len(cfg.buckets.order) == 7
        assert cfg.buckets.rules().bucket_of("3017", "電腦及週邊設備") == "AI硬體零組件"


def test_config_rejects_unknown_bucket(tmp_path):
    txt = (EX / "00981A_2027.yaml").read_text(encoding="utf-8").replace("bucket: AI硬體零組件", "bucket: 不存在的桶")
    (tmp_path / "bad.yaml").write_text(txt, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(tmp_path / "bad.yaml")


def test_config_rejects_bad_confidence(tmp_path):
    txt = (EX / "00981A_2027.yaml").read_text(encoding="utf-8").replace("confidence: 0.50", "confidence: 1.5")
    (tmp_path / "bad.yaml").write_text(txt, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(tmp_path / "bad.yaml")
