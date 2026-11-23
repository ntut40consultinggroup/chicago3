"""用第 0 期凍結的 fixtures，確認新核心計算重現課程原始程式（legacy/stage8_saa.py）的結果。

不需要原始市場資料：輸入是凍結的桶日報酬與容量，期望值是凍結的輸出。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chicago3 import bands, budget, equilibrium, neutral, optimize, risk, views

FX = Path(__file__).parent / "fixtures"
G = json.loads((Path(__file__).parent / "golden" / "golden.json").read_text())
C = json.loads((FX / "constants.json").read_text())
TSMC = "台積電"
CASES = {  # fixture 前綴 → (估計期間桶報酬檔, 觀點)
    "p27": ("bucket_returns_est2027.csv", [views.View("AI硬體零組件", 0.05, 0.5)]),
    "p25": ("bucket_returns_est2025.csv", [views.View("AI硬體零組件", 0.05, 0.5)]),
    "p25s": ("bucket_returns_est2025.csv", [views.View("AI硬體零組件", 0.15, 0.75)]),
}


def load(k):
    W = pd.read_csv(FX / f"{k}_weights.csv", index_col=0)
    S = pd.read_csv(FX / f"{k}_cov.csv", index_col=0)
    sc = json.loads((FX / f"{k}_scalars.json").read_text())
    R = pd.read_csv(FX / CASES[k][0], index_col=0)
    return W, S, sc, R


def run(k):
    """新套件的完整政策流程（與 legacy build() 相同步驟）。"""
    W, _, _, R = load(k)
    S = risk.cov_ledoit_wolf(R, ann=C["ANN"])
    upper = W.cap.copy()
    upper[TSMC] = min(upper[TSMC], 0.25)
    wN = neutral.investable_neutral(W.wB, upper)
    te_struct = risk.te((wN - W.wB).values, S)
    cap = budget.strategic_cap(C["TE_TOTAL"], te_struct)
    bl = views.black_litterman(S, W.wB, CASES[k][1], erp=C["ERP"], tau=C["TAU"])
    wP = optimize.solve_policy(bl.alpha, S, wN, te_cap=cap, ir=C["IR"], upper=W.cap, max_tilt=C["TILT_MAX"],
                               fixed=(TSMC,))
    return dict(S=S, wN=wN, te_struct=te_struct, cap=cap, bl=bl, wP=wP)


@pytest.mark.parametrize("k", list(CASES))
def test_reproduces_legacy(k):
    W, S0, sc, _ = load(k)
    o = run(k)
    assert np.abs(o["S"].values - S0.values).max() < 1e-12
    assert np.abs(o["wN"] - W.wN).max() < 1e-12
    assert np.isclose(o["te_struct"], sc["te_struct"], atol=1e-12)
    assert np.isclose(o["cap"], sc["budget"], atol=1e-12)
    assert np.isclose(o["bl"].delta, sc["delta"], atol=1e-12)
    assert np.abs(o["bl"].Pi - W.Pi).max() < 1e-12
    assert np.abs(o["bl"].alpha - W.alpha).max() < 1e-12
    assert np.abs(o["wP"] - W.wP).max() < 1e-6          # 解題器數值誤差
    assert np.isclose(risk.te((o["wP"] - o["wN"]).values, o["S"]), sc["te_strat"], atol=1e-6)
    assert np.isclose(risk.te((o["wP"] - W.wB).values, o["S"]), sc["te_total"], atol=1e-6)


def test_bands_reproduce_legacy():
    W, _, _, _ = load("p27")
    o = run("p27")
    bd = bands.rebalancing_bands(o["wP"], o["wN"], o["S"], te_increment=0.01, clip=tuple(C["BANDS"]),
                                 no_up=(TSMC,), fixed=(TSMC,))
    for b, (lo, hi) in G["bands_2027"].items():
        assert abs(bd.loc[b, "lower"] - lo) < 1e-5 and abs(bd.loc[b, "upper"] - hi) < 1e-5, b
    wF = pd.read_csv(FX / "fund_bucket_weights_20261120.csv", index_col=0).iloc[:, 0]
    st = bands.band_status(wF, bd)
    expect = {"高於上限": "above", "低於下限": "below", "": ""}
    assert st.to_dict() == {b: expect[v] for b, v in G["out_of_band_2027"].items()}


def test_golden_te_and_decomposition():
    W, _, _, _ = load("p27")
    o = run("p27")
    wF = pd.read_csv(FX / "fund_bucket_weights_20261120.csv", index_col=0).iloc[:, 0].reindex(W.index).fillna(0)
    assert abs(risk.te((wF - W.wB).values, o["S"]) - G["te_fund_now_2027"]) < 1e-9
    assert abs(risk.beta(wF, W.wB, o["S"]) - G["fund_beta"]) < 1e-9
    d = equilibrium.decompose_expected_active(o["bl"].Pi, o["bl"].alpha, wF, W.wB)
    assert abs(d["beta_premium"] - G["fund_beta_premium"]) < 1e-9
    assert abs(d["view_alpha"] - G["fund_view_alpha"]) < 1e-9
    assert abs(d["expected_active"] - G["fund_expected_active"]) < 1e-9


def test_frontier_reproduces_lecture_point():
    """講義 3.8：TE 15% 附近的前緣組合 β 約 1.33，AI 硬體約 31%。"""
    W, _, _, _ = load("p27")
    o = run("p27")
    upper = W.cap.copy()
    upper[TSMC] = 0.25
    f = optimize.frontier(o["bl"].mu, o["S"], W.wB, upper, kind="active", n=40)
    row = f.iloc[(f.risk - 0.15).abs().argmin()]
    w = row[W.index]
    assert abs(risk.beta(w, W.wB, o["S"]) - 1.33) < 0.01
    assert abs(w["AI硬體零組件"] - 0.313) < 0.005 and w["金融保險"] < 1e-4
