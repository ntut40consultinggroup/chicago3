"""第 6 期：用 Streamlit AppTest 無頭執行網頁主流程（需要真實資料；沒有就跳過）。"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DATA = Path(os.environ.get("CHICAGO3_DATA", "/home/claude/data"))


@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    if not (DATA / "saa_20261122").exists():
        pytest.skip("沒有原始資料")
    store = tmp_path_factory.mktemp("store")
    subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "seed_from_json.py"), f"local:{store}", str(DATA / "saa_20261122"),
                    str(DATA / "hist_taiex_20261122" / "tri.json"), str(DATA / "daily_20261120"), "2026-11-20"], check=True)
    (store / "00981A" / "holdings").mkdir(parents=True)
    (store / "00981A" / "holdings" / "2026-11-20.csv").write_bytes((ROOT / "tests" / "fixtures" / "holdings_00981A_20261120.csv").read_bytes())
    (store / "00981A" / "holdings" / "2026-11-20.json").write_text('{"nav": 211834195132.0, "date": "2026-11-20"}')
    old = {k: os.environ.get(k) for k in ("CHICAGO3_STORE", "CHICAGO3_ADMIN_PASSWORD")}
    os.environ.update(CHICAGO3_STORE=f"local:{store}", CHICAGO3_ADMIN_PASSWORD="pw")
    yield store
    for k, v in old.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _btn(at, label):
    return next(b for b in at.button if b.label.startswith(label))


def test_full_flow(app_env):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "policy_page.py"), default_timeout=300)
    at.run()
    assert not at.exception and [m.value for m in at.metric][:3] == ["2026-11-20", "400", "2026-11-20"]
    _btn(at, "載入資料").click().run()
    assert not at.exception and "持股日期 2026-11-20" in at.success[0].value and "淨資產 2,118 億元" in at.success[0].value
    assert "③ 各桶現況" in [h.value for h in at.header]
    _btn(at, "Optimize").click().run()
    assert "⑤ 參數" in [h.value for h in at.header]
    _btn(at, "⑥ 執行分析").click().run()
    assert not at.exception and not at.error
    m = {x.label: x.value for x in at.metric}
    assert m["結構 TE"] == "5.10%" and m["策略 TE（上限）"] == "0.51%"
    assert "5 桶在區間外" in at.info[0].value and "Beta 溢酬 2.17%" in at.info[0].value
    # 管理者存到雲端硬碟
    at.sidebar.text_input[0].set_value("pw").run()
    _btn(at, "存到雲端硬碟").click().run()
    assert any("已存到" in s.value for s in at.success)
    assert any((app_env / "00981A" / "runs").iterdir())


def test_requires_password_to_save(app_env):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "policy_page.py"), default_timeout=300)
    at.run()
    _btn(at, "載入資料").click().run()
    _btn(at, "Optimize").click().run()
    _btn(at, "⑥ 執行分析").click().run()
    assert _btn(at, "存到雲端硬碟").disabled


def test_partial_data_flow(app_env, tmp_path):
    """同步進行中：只有市值最大的 60 檔、股價只到 10/30，也能從頭跑到尾，並顯示「資料不完整」。"""
    import io
    import json
    import shutil

    import pandas as pd
    from streamlit.testing.v1 import AppTest

    from chicago3.data import twse
    part = tmp_path / "partial"
    shutil.copytree(app_env, part)
    snap = twse.load_snapshot(part / "_market" / "snapshot" / "20261120")
    top60 = list(snap[snap.市場 == "上市"].index[:60])
    p = pd.read_parquet(part / "_market" / "prices.parquet")
    p = p[p.stock_id.astype(str).isin(top60) & (p.date <= "2026-10-30")]
    p.to_parquet(part / "_market" / "prices.parquet", index=False)
    man = json.loads((part / "_market" / "manifest.json").read_text())
    man["prices_through"] = {c: "2026-10-30" for c in top60}
    (part / "_market" / "manifest.json").write_text(json.dumps(man))
    os.environ["CHICAGO3_STORE"] = f"local:{part}"
    try:
        at = AppTest.from_file(str(ROOT / "app" / "policy_page.py"), default_timeout=300)
        at.run()
        _btn(at, "載入資料").click().run()
        assert not at.exception
        warn = [w.value for w in at.warning]
        assert any("資料不完整" in w and "60／400" in w and "2026-10-30" in w for w in warn), warn
        _btn(at, "Optimize").click().run()
        _btn(at, "⑥ 執行分析").click().run()
        assert not at.exception and not at.error and "⑦ 結果" in [h.value for h in at.header]
    finally:
        os.environ["CHICAGO3_STORE"] = f"local:{app_env}"
