"""第 2 期驗收：只靠 YAML，從原始資料端到端重現課程的黃金數字。

需要原始資料（約 13 MB，不放在 repo）：設定環境變數 CHICAGO3_DATA 指向資料資料夾，
或資料放在 examples/*.yaml 裡 data.root 指定的位置；都沒有就跳過。
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chicago3.config import load_config
from chicago3.pipeline import ReconcileError, load_market, reconcile, run_policy

EX = Path(__file__).parents[1] / "examples"
G = json.loads((Path(__file__).parent / "golden" / "golden.json").read_text())
TOL = 0.001   # 0.1 個百分點


def _cfg(name):
    cfg = load_config(EX / name)
    root = Path(os.environ.get("CHICAGO3_DATA", cfg.data.root))
    if not (root / cfg.data.prices_dir).exists():
        pytest.skip(f"沒有原始資料（{root}）")
    return cfg


@pytest.fixture(scope="module")
def p27():
    return run_policy(_cfg("00981A_2027.yaml"))


@pytest.fixture(scope="module")
def p25():
    return run_policy(_cfg("00981A_2025.yaml"))


def test_reconcile(p27):
    r = p27["recon"]
    assert abs(r.official.iloc[1] - G["recon_2026"]["official"]) < 1e-9
    assert abs(r.rebuilt.iloc[1] - G["recon_2026"]["rebuilt"]) < 1e-9
    assert abs(r.rebuilt.iloc[0] - G["recon_2025"]["rebuilt"]) < 1e-9


def test_policy_2027(p27):
    for k, v in G["policy_2027"].items():
        assert abs(p27["wP"][k] - v) < TOL, k
    for k, v in G["neutral_2027"].items():
        assert abs(p27["wN"][k] - v) < 1e-9, k
    for k in ("struct", "budget", "strat", "policy"):
        name = dict(struct="structural", budget="strategic_cap", strat="strategic", policy="policy")[k]
        assert abs(p27["te"][name] - G["te_2027"][k]) < 1e-6, k


def test_capacity(p27):
    for k, v in G["capacity_10d"].items():
        assert abs(p27["capacity"][k] - v) < 1e-9, k
    for k, v in G["capacity_3d"].items():
        assert abs(p27["capacity_compare"][3][k] - v) < 1e-9, k


def test_bands_and_fund(p27):
    for b, (lo, hi) in G["bands_2027"].items():
        assert abs(p27["bands"].loc[b, "lower"] - lo) < 1e-5 and abs(p27["bands"].loc[b, "upper"] - hi) < 1e-5, b
    expect = {"高於上限": "above", "低於下限": "below", "": ""}
    assert p27["fund_status"].to_dict() == {b: expect[v] for b, v in G["out_of_band_2027"].items()}
    assert abs(p27["te_fund"] - G["te_fund_now_2027"]) < 1e-9
    assert abs(p27["fund_expected"]["beta_premium"] - G["fund_beta_premium"]) < 1e-9
    assert abs(p27["fund_expected"]["view_alpha"] - G["fund_view_alpha"]) < 1e-9


def test_policy_2025(p25):
    assert abs(p25["te"]["structural"] - G["te_2025"]["struct"]) < 1e-6
    assert abs(p25["te"]["strategic"] - G["te_2025"]["strat"]) < 1e-6
    ai = p25["wP"]["AI硬體零組件"] - p25["wN"]["AI硬體零組件"]
    assert abs(ai - G["policy_2025_ai_tilt"]) < TOL


def test_reconcile_failure_is_loud():
    cfg = _cfg("00981A_2027.yaml")
    cfg.reconcile.tolerance = 0.01      # 實際差距約 2–3%，應該失敗
    with pytest.raises(ReconcileError):
        reconcile(cfg, load_market(cfg))


# ---------- 第 3 期：穩健性、歸因、監控、報告 ----------
@pytest.fixture(scope="module")
def full():
    from chicago3.pipeline import run_all
    return run_all(_cfg("00981A_2027.yaml"))


def test_bootstrap_matches_lecture(full):
    b = full["robustness"]["bootstrap"]
    lo, hi = G["resample_ai_p5_p95"]
    assert abs(b.loc["AI硬體零組件", "p5"] - lo) < 1e-6 and abs(b.loc["AI硬體零組件", "p95"] - hi) < 1e-6


def test_stress_returns_match_lecture(full):
    t = full["robustness"]["stress"]["AI衝擊"][0].set_index("portfolio").return_
    for k in ("benchmark", "neutral", "policy", "fund"):
        assert abs(t[k] - G["stress"][k]) < 1e-9, k


def test_three_tier_matches_lecture(full):
    t = full["attribution"]["tiers"]
    g = G["tiers_2026"]
    assert abs(t["active"] - g["active"]) < 1e-9 and abs(t["structural"] - g["structural"]) < 1e-9
    mild, strong = t["policies"]["2025政策（溫和觀點）"], t["policies"]["強烈觀點"]
    assert abs(mild["strategic"] - g["strategic"]) < 1e-9 and abs(mild["execution"] - g["execution"]) < 1e-9
    assert abs(strong["strategic"] - g["strategic_strong"]) < 1e-9 and abs(strong["execution"] - g["execution_strong"]) < 1e-9


def test_sensitivity_and_turnover(full):
    s = full["robustness"]["sensitivity"]
    row = s[(s.excess == 0.10) & (s.confidence == 0.75)].iloc[0]
    assert abs(row["AI硬體零組件"] - 0.203) < 0.001                  # 講義 3.4：觀點 10%／信心 75% → 20.3%
    assert abs(full["monitor"]["turnover_to_policy"] - 0.39) < 0.005  # 講義 §5：單邊換手約 39%


def test_report_files(full, tmp_path):
    from openpyxl import load_workbook
    from chicago3.pipeline import write_report
    p = write_report(full, tmp_path)
    assert all(v.exists() and v.stat().st_size > 1000 for v in p.values())
    book = p["book"].read_text(encoding="utf-8")
    for h in [f"## {i}." for i in range(1, 10)]:
        assert h in book
    assert "**25.0%**" in book and "待補" in book and "5 桶在區間外" in book
    names = load_workbook(p["xlsx"], read_only=True).sheetnames
    assert {"0_說明", "3_政策組合", "8_重抽樣", "10_回溯三層拆解", "13_效率前緣"} <= set(names)
