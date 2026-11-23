"""第 4 期：人造 3 桶市場端到端，結果和 numpy 獨立手算比對（不需要外部資料，CI 會跑）。

手算刻意不呼叫 chicago3 的函數，並盡量用不同的算法：
- 桶報酬直接用「收盤 × 股數」當市值（套件是用市值變化率往回推）
- Black–Litterman 用 μ = Π + τΣP'(PτΣP' + Ω)⁻¹(Q − PΠ) 的形式（套件用反矩陣形式）
- 政策用拉格朗日解析解（加總為 0、固定桶不動兩條等式）
"""
import os

import numpy as np
import pandas as pd
import pytest

from synthetic import CODES, NAV, SHARES, TURNOVER, make_market, write_config

from chicago3.config import load_config
from chicago3.pipeline import run_all, write_report

ORDER = ["大型股", "電子", "其他"]
BUCKET = {"9001": "大型股", "9002": "電子", "9003": "電子", "9004": "電子", "9005": "其他", "9006": "其他"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    saved = os.environ.pop("CHICAGO3_DATA", None)    # 人造資料有自己的 root，不受環境變數影響
    try:
        root = tmp_path_factory.mktemp("synthetic")
        m = make_market(root)
        cfg = load_config(write_config(root, m))
        yield m, cfg, run_all(cfg)
    finally:
        if saved is not None:
            os.environ["CHICAGO3_DATA"] = saved


@pytest.fixture(scope="module")
def hand(world):
    m, cfg, _ = world
    close, R, mcap, dates = m["close"], m["R"], m["mcap"], m["dates"]
    G = np.array([[BUCKET[c] == b for c in CODES] for b in ORDER], float)          # 桶 × 股票
    bm = mcap @ G.T                                                                  # 每日各桶市值
    RB = ((mcap[:-1] * R[1:]) @ G.T) / bm[:-1]                                       # 第 1 天起的桶報酬
    RB = pd.DataFrame(RB, index=dates[1:], columns=ORDER)
    est = RB[RB.index > dates[49]]
    S = np.cov(est.values.T, ddof=1) * 250
    wB = bm[-1] / bm[-1].sum()
    # 可投資中性：大型股壓到 25%，其他桶等比例放大
    wN = wB.copy()
    wN[0] = 0.25
    wN[1:] = wB[1:] * 0.75 / wB[1:].sum()
    te_s = np.sqrt((wN - wB) @ S @ (wN - wB))
    cap_te = np.sqrt((0.15 ** 2 - te_s ** 2) / 2)
    delta = 0.06 / (wB @ S @ wB)
    Pi = delta * S @ wB
    P = -wB.copy()
    P[1] += 1                                                                        # 電子 − 指標
    tS = 0.05 * S
    om = (P @ tS @ P) * (1 - 0.5) / 0.5
    mu = Pi + tS @ P * (0.03 - P @ Pi) / (P @ tS @ P + om)
    alpha = mu - Pi
    lam = 0.5 / cap_te
    A = np.array([[1, 1, 1], [1, 0, 0]], float)                                     # 1'a = 0、a_大型股 = 0
    Si = np.linalg.inv(S)
    nu = np.linalg.solve(A @ Si @ A.T, A @ Si @ alpha)
    a = Si @ (alpha - A.T @ nu) / lam
    # 容量（10 天）：每檔 min(60% NAV（9001 25%）, 10% 市值, 10×20%×最近 60 日均成交值)
    money = close * np.array([SHARES[c] for c in CODES]) * TURNOVER
    adv = money[-60:].mean(axis=0)
    lim = np.minimum.reduce([np.where(np.array(CODES) == "9001", 0.25, 0.60) * NAV, 0.10 * mcap[-1], 10 * 0.2 * adv])
    cap = G @ lim / NAV
    wF = np.array([20.0, 75.0, 1.0]) / 96.0
    return dict(S=S, wB=wB, wN=wN, te_s=te_s, cap_te=cap_te, Pi=Pi, alpha=alpha, wP=wN + a, cap=cap, wF=wF,
                beta_prem=Pi @ (wF - wB), view_alpha=alpha @ (wF - wB))


def test_reconcile_exact(world):
    _, _, o = world
    assert abs(o["recon"]["diff"].iloc[0]) < 1e-12 and o["recon"]["corr"].iloc[0] > 0.999999


def test_benchmark_neutral_and_cov(world, hand):
    _, _, o = world
    assert np.allclose(o["wB"].values, hand["wB"], atol=1e-12)
    assert np.allclose(o["wN"].values, hand["wN"], atol=1e-12)
    assert np.allclose(o["S"].values, hand["S"], atol=1e-12)


def test_capacity(world, hand):
    _, _, o = world
    assert np.allclose(o["capacity"].values, hand["cap"], rtol=1e-12)


def test_equilibrium_and_views(world, hand):
    _, _, o = world
    assert np.allclose(o["bl"].Pi.values, hand["Pi"], atol=1e-12)
    assert np.allclose(o["bl"].alpha.values, hand["alpha"], atol=1e-12)


def test_policy_matches_lagrange_solution(world, hand):
    _, _, o = world
    a = hand["wP"] - hand["wN"]
    assert np.sqrt(a @ hand["S"] @ a) < hand["cap_te"] and np.abs(a).max() < 0.15 and (hand["wP"] > 0).all()  # 限制沒綁住
    assert np.allclose(o["wP"].values, hand["wP"], atol=1e-6)
    assert abs(o["te"]["structural"] - hand["te_s"]) < 1e-12
    assert abs(o["te"]["strategic_cap"] - hand["cap_te"]) < 1e-12


def test_fund_weights_and_decomposition(world, hand):
    _, _, o = world
    assert np.allclose(o["wF"].values, hand["wF"], atol=1e-12)
    assert abs(o["fund_expected"]["beta_premium"] - hand["beta_prem"]) < 1e-12
    assert abs(o["fund_expected"]["view_alpha"] - hand["view_alpha"]) < 1e-12


def test_full_pipeline_outputs(world, tmp_path):
    _, cfg, o = world
    assert len(o["robustness"]["bootstrap"]) == 3 and len(o["robustness"]["sensitivity"]) == 2
    assert o["attribution"] is None                      # 沒設定回溯就不做
    assert {"status", "turnover_to_band"} <= set(o["monitor"])
    p = write_report(o, tmp_path)
    assert all(v.exists() for v in p.values())
    book = p["book"].read_text(encoding="utf-8")
    assert "TEST3" in book and "大型股" in book and "## 8." in book
