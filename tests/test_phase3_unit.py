"""第 3 期：穩健性、歸因、監控的單元測試（人造資料）。"""
import numpy as np
import pandas as pd
import pytest

from chicago3 import attribution, monitor, robustness, risk
from chicago3.views import View

B = ["A", "B", "C"]


def test_stress_replay_relative_to_replayed_benchmark():
    idx = pd.date_range("2026-01-01", periods=4, freq="B")
    RB = pd.DataFrame({"A": [0.5, -0.1, -0.1, 0.0], "B": [0.5, 0.0, 0.0, 0.0], "C": [0.5, 0.1, 0.1, 0.0]}, idx)
    ports = {"benchmark": pd.Series([1 / 3] * 3, B), "tilted": pd.Series([1.0, 0, 0], B)}
    t, rS = robustness.stress_replay(RB, idx[0], idx[-1], ports)     # 不含起日那天的 +50%
    assert np.isclose(rS["A"], 0.9 * 0.9 - 1)
    tt = t.set_index("portfolio")
    assert np.isclose(tt.loc["benchmark", "vs_benchmark"], 0)
    assert np.isclose(tt.loc["tilted", "vs_benchmark"], tt.loc["tilted", "return_"] - tt.loc["benchmark", "return_"])


def test_block_bootstrap_deterministic_and_shape():
    rng = np.random.default_rng(1)
    R = pd.DataFrame(rng.normal(0, 0.01, (120, 3)), columns=B)
    solve = lambda S, vs: pd.Series(np.diag(S.values) / np.diag(S.values).sum(), B)
    d1 = robustness.block_bootstrap(R, risk.cov_sample, solve, [], n=5, block=10, seed=7)
    d2 = robustness.block_bootstrap(R, risk.cov_sample, solve, [], n=5, block=10, seed=7)
    assert d1.shape == (5, 3) and np.allclose(d1, d2)
    with pytest.raises(ValueError):
        robustness.block_bootstrap(R.iloc[:10], risk.cov_sample, solve, [], block=20)


def test_view_sensitivity_zero_excess_means_no_view():
    S = pd.DataFrame(np.eye(3) * 0.04, B, B)
    seen = []
    solve = lambda S_, vs: (seen.append(vs), pd.Series(1 / 3, B))[1]
    t = robustness.view_sensitivity(S, solve, pd.Series(1 / 3, B), "A", [0.0, 0.05], [0.5], lambda a: 0.0)
    assert seen[0] == [] and seen[1] == [View("A", 0.05, 0.5)] and len(t) == 2


def test_rebalanced_returns_drift_and_reset():
    idx = pd.to_datetime(["2026-01-29", "2026-01-30", "2026-02-02"])
    R = pd.DataFrame({"A": [0.10, 0.10, 0.0], "B": [0.0, 0.0, 0.10]}, idx)
    tg = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=pd.PeriodIndex(["2026-01", "2026-02"], freq="M"))
    r = attribution.rebalanced_returns(R, tg, idx)
    assert np.isclose(r.iloc[0], 0.05)
    assert np.isclose(r.iloc[1], 0.55 / 1.05 * 0.10)      # 漂移後 A 的權重變大
    assert np.isclose(r.iloc[2], 0.05)                     # 2 月初重設回 50/50


def test_three_tier_adds_up():
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    f, t, b, n, p = (pd.Series(x, idx) for x in ([0.01] * 5, [0.02] * 5, [0.021] * 5, [0.019] * 5, [0.018] * 5))
    out = attribution.three_tier(f, t, b, n, {"P": p})
    s = out["structural"] + out["policies"]["P"]["strategic"] + out["policies"]["P"]["execution"]
    assert np.isclose(s, out["active"])
    m = attribution.monthly_three_tier(f, t, b, n, p)
    assert np.allclose(m.structural + m.strategic + m.execution, m.active)


def test_fund_market_returns_add_back_dividend():
    price = pd.DataFrame(dict(date=["2026-01-01", "2026-01-02"], close=[10.0, 9.8]))
    div = pd.DataFrame(dict(date=["2026-01-02"], stock_and_cache_dividend=[0.3]))
    assert np.isclose(attribution.fund_market_returns(price, div).iloc[0], 9.8 / 9.7 - 1)


def test_turnover_and_nearest_band():
    bands = pd.DataFrame(dict(lower=[0.2, 0.2, 0.2], upper=[0.4, 0.4, 0.4]), index=B)
    w = pd.Series([0.6, 0.3, 0.1], B)
    edge = monitor.nearest_in_band(w, bands)
    assert np.isclose(edge.sum(), 1) and (edge >= 0.2 - 1e-8).all() and (edge <= 0.4 + 1e-8).all()
    assert np.allclose(edge, [0.4, 0.4, 0.2], atol=1e-6)
    assert np.isclose(monitor.one_way_turnover(w, edge), 0.2)
    r = monitor.band_report(w, pd.Series(1 / 3, B), bands)
    assert r["n_out"] == 2 and r["turnover_to_band"] <= r["turnover_to_policy"] + 1e-9
