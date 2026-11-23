"""第 6 期：網頁邏輯（不需要真實資料）。"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chicago3 import webapp

ROOT = Path(__file__).parents[1]
PATHS = dict(prices_file="/x/prices.parquet", dividends_file="/x/dividends.parquet", tri_file="/x/tri.parquet",
             snapshot_dir="/x/snap", holdings_file="/x/h.csv")


def test_defaults_and_build_config():
    tpl = webapp.template_dict()
    assert "attribution" not in tpl
    p = webapp.default_params(tpl)
    assert p["total_te"] == 0.15 and p["views"][0]["bucket"] == "AI硬體零組件"
    p.update(liq_days=3, tsmc_cap=0.2, views=[dict(bucket="金融保險", excess=-0.02, confidence=0.6), dict(bucket="其他電子", excess=0, confidence=0.5)])
    cfg = webapp.build_config(tpl, dict(code="X", name="測試", nav=1e10), PATHS, p, date(2026, 11, 20),
                              stress=(date(2026, 10, 6), date(2026, 11, 20)))
    assert cfg.constraints.liquidity.days == 3 and cfg.constraints.bucket_upper == {"台積電": 0.2}
    assert cfg.constraints.single_cap_overrides == {"2330": 0.2}
    assert [v.bucket for v in cfg.views] == ["金融保險"]                 # 超額 0 的觀點被拿掉
    assert cfg.robustness.view_grid.bucket == "金融保險" and cfg.robustness.stress[0].start == date(2026, 10, 6)
    assert cfg.dates.estimation[0] == date(2025, 11, 20) and cfg.data.prices_file == "/x/prices.parquet"


def test_build_config_without_views():
    tpl = webapp.template_dict()
    p = webapp.default_params(tpl)
    p["views"] = []
    cfg = webapp.build_config(tpl, dict(code="X", nav=1e10), PATHS, p, date(2026, 11, 20))
    assert cfg.views == [] and cfg.robustness.view_grid is None and cfg.robustness.stress == []


def test_worst_drawdown():
    idx = pd.date_range("2026-01-01", periods=6)
    r = pd.Series([0.1, 0.1, -0.2, -0.1, 0.05, 0.3], idx)
    peak, trough, dd = webapp.worst_drawdown(r)
    assert peak == idx[1] and trough == idx[3] and np.isclose(dd, 0.8 * 0.9 - 1)


def test_parse_uploaded_holdings():
    df = webapp.parse_uploaded_holdings(pd.DataFrame(dict(Code=["2330", "2454"], Weight_pct=[18.2, 9.3])))
    assert list(df.type) == ["stock", "stock"] and df.weight_pct.sum() == pytest.approx(27.5)
    with pytest.raises(ValueError):
        webapp.parse_uploaded_holdings(pd.DataFrame(dict(code=["2330"])))
    with pytest.raises(ValueError):                                     # 加總超過 150%：多半是單位填錯
        webapp.parse_uploaded_holdings(pd.DataFrame(dict(code=["2330", "1"], weight_pct=[120.0, 60.0])))


def test_entry_page_loads(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv("CHICAGO3_STORE", raising=False)
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("找不到資料儲存位置" in e.value for e in at.error)      # 沒有設定儲存位置時給清楚的訊息
