"""政策最佳化與效率前緣。

政策：max α'(w − w_N) − (λ/2)·TE²，λ = IR ÷ 策略 TE 上限（Grinold–Kahn 主動風險觀念）。
只用 α = μ_BL − Π，不用 μ：Beta 溢酬不算判斷力（見講義 3.8 節）。
"""
import cvxpy as cp
import numpy as np
import pandas as pd


class InfeasibleError(RuntimeError):
    """限制互相衝突，找不到可行解。"""


def _chol(S: pd.DataFrame) -> np.ndarray:
    return np.linalg.cholesky(S.values + 1e-10 * np.eye(len(S)))


def _check(prob: cp.Problem, what: str):
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise InfeasibleError(f"{what}：解題狀態 {prob.status}。請檢查上限、傾斜上限、固定桶與預算是否互相衝突。")


def solve_policy(alpha: pd.Series, S: pd.DataFrame, wN: pd.Series, te_cap: float, ir: float = 0.5,
                 upper: pd.Series | None = None, max_tilt: float = 0.15, fixed: tuple = ()) -> pd.Series:
    """回傳政策權重 P。

    限制：權重加總 1、≥ 0；|w − w_N| ≤ max_tilt；fixed 裡的桶固定在 w_N；策略 TE ≤ te_cap；w ≤ upper。
    upper 低於 w_N 時以 w_N 為準（中性本身一定可行）。
    """
    idx = wN.index
    n = len(idx)
    w = cp.Variable(n)
    a = w - wN.values
    L = _chol(S.loc[idx, idx])
    lam = ir / te_cap
    up = (pd.Series(1.0, idx) if upper is None else upper.reindex(idx).fillna(1.0)).clip(lower=wN)
    cons = [cp.sum(w) == 1, w >= 0, cp.abs(a) <= max_tilt, cp.norm(L.T @ a) <= te_cap, w <= up.values]
    cons += [a[idx.get_loc(b)] == 0 for b in fixed]
    prob = cp.Problem(cp.Maximize(alpha.reindex(idx).values @ a - lam / 2 * cp.sum_squares(L.T @ a)), cons)
    prob.solve()
    _check(prob, "政策最佳化")
    return pd.Series(np.clip(w.value, 0, None), idx)


def closed_form_active(alpha: pd.Series, S: pd.DataFrame, lam: float) -> pd.Series:
    """只有「加總為 0」限制時的解析解：a = Σ⁻¹(α − c·1)/λ，c 使 1'a = 0。用來驗證最佳化器。"""
    Si = np.linalg.inv(S.values)
    one = np.ones(len(S))
    c = (one @ Si @ alpha.values) / (one @ Si @ one)
    return pd.Series(Si @ (alpha.values - c * one) / lam, S.index)


def frontier(mu: pd.Series, S: pd.DataFrame, wB: pd.Series, upper: pd.Series, kind: str = "active",
             n: int = 40) -> pd.DataFrame:
    """效率前緣。只用硬限制（≥ 0、≤ upper、加總 1），不加政策用的傾斜上限。

    kind='total'：給定總波動求最大 μ'w；kind='active'：給定 TE（對指標）求最大 μ'(w − w_B)。
    回傳每一列：risk、ret，以及各桶權重。
    """
    if kind not in ("total", "active"):
        raise ValueError("kind 必須是 'total' 或 'active'")
    idx = mu.index
    L = _chol(S.loc[idx, idx])
    w = cp.Variable(len(idx))
    base = [cp.sum(w) == 1, w >= 0, w <= upper.reindex(idx).fillna(1.0).values]
    wb = wB.reindex(idx).values

    def risk(x):
        return cp.norm(L.T @ x) if kind == "total" else cp.norm(L.T @ (x - wb))

    p = cp.Problem(cp.Minimize(risk(w)), base)
    p.solve()
    _check(p, "前緣（最小風險端）")
    r0 = float(risk(w).value)
    p = cp.Problem(cp.Maximize(mu.values @ w), base)
    p.solve()
    _check(p, "前緣（最大報酬端）")
    r1 = float(risk(w).value)
    rows = []
    for r in np.linspace(r0, r1, n):
        p = cp.Problem(cp.Maximize(mu.values @ w), base + [risk(w) <= r + 1e-9])
        p.solve()
        _check(p, f"前緣（風險 {r:.4f}）")
        x = np.clip(w.value, 0, None)
        ret = float(mu.values @ x) - (0.0 if kind == "total" else float(mu.values @ wb))
        rows.append(dict(risk=r, ret=ret, **dict(zip(idx, x))))
    return pd.DataFrame(rows)


def portfolio_stats(mu: pd.Series, S: pd.DataFrame, w: pd.Series, wB: pd.Series) -> dict:
    """單一組合在前緣圖上的位置。"""
    idx = mu.index
    x, wb, Sv = w.reindex(idx).values, wB.reindex(idx).values, S.loc[idx, idx].values
    return dict(beta=float(x @ Sv @ wb / (wb @ Sv @ wb)), vol=float(np.sqrt(x @ Sv @ x)),
                expected_excess=float(mu.values @ x), te=float(np.sqrt((x - wb) @ Sv @ (x - wb))),
                expected_active=float(mu.values @ (x - wb)))
