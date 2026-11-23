"""核心計算的理論性質（用人造資料，不依賴真實市場資料）。"""
import numpy as np
import pandas as pd
import pytest

from chicago3 import bands, budget, equilibrium, neutral, optimize, risk, views

B = ["A", "B", "C", "D"]


@pytest.fixture
def market():
    rng = np.random.default_rng(0)
    F = rng.normal(0, 0.012, (500, 1))
    R = pd.DataFrame(F * [1.2, 0.9, 1.0, 0.6] + rng.normal(0, 0.008, (500, 4)), columns=B)
    S = risk.cov_ledoit_wolf(R)
    wB = pd.Series([0.4, 0.3, 0.2, 0.1], B)
    return R, S, wB


# ---------- risk ----------
def test_cov_methods_psd(market):
    R, _, _ = market
    for m in ("sample", "ledoit_wolf", "ewma"):
        S = risk.estimate_cov(R, m)
        assert np.linalg.eigvalsh(S.values).min() > 0
        assert np.allclose(S.values, S.values.T)


def test_te_scales_linearly(market):
    _, S, _ = market
    a = np.array([0.05, -0.02, -0.02, -0.01])
    assert risk.te(np.zeros(4), S) == 0
    assert np.isclose(risk.te(3 * a, S), 3 * risk.te(a, S))


def test_benchmark_beta_is_one(market):
    _, S, wB = market
    assert np.isclose(risk.beta(wB, wB, S), 1.0)


# ---------- equilibrium ----------
def test_benchmark_earns_erp(market):
    _, S, wB = market
    Pi, _ = equilibrium.implied_returns(S, wB, erp=0.06)
    assert np.isclose(Pi @ wB, 0.06)


def test_beta_premium_identity(market):
    _, S, wB = market
    Pi, _ = equilibrium.implied_returns(S, wB, erp=0.06)
    w = pd.Series([0.1, 0.5, 0.3, 0.1], B)
    d = equilibrium.decompose_expected_active(Pi, pd.Series(0.0, B), w, wB)
    assert np.isclose(d["beta_premium"], 0.06 * (risk.beta(w, wB, S) - 1))


# ---------- views ----------
def test_no_views_mu_equals_pi(market):
    _, S, wB = market
    r = views.black_litterman(S, wB, [])
    assert np.allclose(r.mu, r.Pi) and np.allclose(r.alpha, 0)


def test_low_confidence_approaches_equilibrium(market):
    _, S, wB = market
    r = views.black_litterman(S, wB, [views.View("D", 0.10, 1e-6)])
    assert np.abs(r.alpha).max() < 1e-5


def test_high_confidence_approaches_view(market):
    _, S, wB = market
    r = views.black_litterman(S, wB, [views.View("D", 0.10, 1 - 1e-9)])
    view_port = pd.Series(-wB.values, B)
    view_port["D"] += 1
    assert np.isclose(view_port @ r.mu, 0.10, atol=1e-6)


def test_positive_view_raises_bucket_alpha(market):
    _, S, wB = market
    r = views.black_litterman(S, wB, [views.View("D", 0.05, 0.5)])
    assert r.alpha.idxmax() == "D"


def test_tau_cancels_when_omega_proportional(market):
    """Ω ∝ τΣ 時 τ 會消掉：網頁不該讓使用者以為調 τ 會改變結果。"""
    _, S, wB = market
    v = [views.View("D", 0.05, 0.5)]
    a1 = views.black_litterman(S, wB, v, tau=0.05).alpha
    a2 = views.black_litterman(S, wB, v, tau=0.5).alpha
    assert np.allclose(a1, a2)


def test_confidence_must_be_open_interval():
    with pytest.raises(ValueError):
        views.View("A", 0.05, 1.0)


# ---------- neutral ----------
def test_neutral_properties():
    wB = pd.Series([0.45, 0.25, 0.20, 0.10], B)
    up = pd.Series({"A": 0.25})
    wN = neutral.investable_neutral(wB, up)
    assert np.isclose(wN.sum(), 1) and wN["A"] == 0.25
    others = wN.drop("A") / wB.drop("A")
    assert np.allclose(others, others.iloc[0])     # 沒被壓的桶等比例放大


def test_neutral_cascade():
    wB = pd.Series([0.45, 0.25, 0.20, 0.10], B)
    wN = neutral.investable_neutral(wB, pd.Series({"A": 0.25, "B": 0.30}))
    assert (wN <= pd.Series({"A": 0.25, "B": 0.30, "C": 1, "D": 1}) + 1e-12).all() and np.isclose(wN.sum(), 1)


def test_neutral_impossible():
    with pytest.raises(ValueError):
        neutral.investable_neutral(pd.Series(0.25, B), pd.Series(0.2, B))


# ---------- optimize ----------
def test_no_alpha_policy_equals_neutral(market):
    _, S, wB = market
    wP = optimize.solve_policy(pd.Series(0.0, B), S, wB, te_cap=0.05)
    assert np.allclose(wP, wB, atol=1e-6)


def test_matches_closed_form_when_unconstrained(market):
    _, S, wB = market
    alpha = pd.Series([0.002, -0.001, 0.0015, -0.0005], B)
    te_cap, ir = 0.10, 0.5
    wP = optimize.solve_policy(alpha, S, wB, te_cap=te_cap, ir=ir, max_tilt=1.0)
    a = optimize.closed_form_active(alpha, S, lam=ir / te_cap)
    assert risk.te(a.values, S) < te_cap and (wB + a > 0).all()   # 確認限制沒有綁住
    assert np.allclose(wP - wB, a, atol=1e-5)


def test_fixed_bucket_and_tilt_cap(market):
    _, S, wB = market
    alpha = pd.Series([0.05, -0.05, 0.05, -0.05], B)
    wP = optimize.solve_policy(alpha, S, wB, te_cap=0.10, max_tilt=0.03, fixed=("A",))
    assert np.isclose(wP["A"], wB["A"], atol=1e-7)
    assert (wP - wB).abs().max() <= 0.03 + 1e-6
    assert risk.te((wP - wB).values, S) <= 0.10 + 1e-6


def test_infeasible_raises(market):
    _, S, _ = market
    wN = pd.Series(0.2, B)   # 加總 0.8，又固定所有桶 → 不可能加總為 1
    with pytest.raises(optimize.InfeasibleError):
        optimize.solve_policy(pd.Series(0.01, B), S, wN, te_cap=0.05, fixed=tuple(B))


def test_frontier_monotone(market):
    _, S, wB = market
    mu = pd.Series([0.08, 0.05, 0.06, 0.03], B)
    f = optimize.frontier(mu, S, wB, pd.Series(1.0, B), kind="active", n=8)
    assert (np.diff(f.ret.values) >= -1e-7).all() and (np.diff(f.risk.values) > 0).all()


# ---------- budget ----------
def test_budget_layers():
    cap = budget.strategic_cap(0.15, 0.051)
    assert np.isclose(cap, np.sqrt((0.15 ** 2 - 0.051 ** 2) / 2))
    assert np.isclose(budget.tactical_room(0.15, 0.0), 0.15)
    assert budget.strategic_cap(0.05, 0.10) == 0


# ---------- bands ----------
def test_band_properties(market):
    _, S, wB = market
    wP = pd.Series([0.38, 0.32, 0.20, 0.10], B)
    bd = bands.rebalancing_bands(wP, wB, S, no_up=("A",), fixed=("A",))
    assert (bd.lower <= bd.target + 1e-12).all() and (bd.target <= bd.upper + 1e-12).all()
    assert bd.loc["A", "up"] == 0
    rest = bd.drop("A")
    assert ((rest[["down", "up"]] >= 0.02 - 1e-12) & (rest[["down", "up"]] <= 0.10 + 1e-12)).all().all()


def test_band_status():
    bd = pd.DataFrame(dict(lower=[0.1, 0.1], upper=[0.2, 0.2]), index=["A", "B"])
    s = bands.band_status(pd.Series({"A": 0.25, "B": 0.05}), bd)
    assert list(s) == ["above", "below"]
