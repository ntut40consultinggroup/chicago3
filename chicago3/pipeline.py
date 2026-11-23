"""從設定檔到政策組合：載入資料 → 勾稽 → 容量 → 中性 → 均衡與觀點 → 政策 → 區間 → 基金現況。"""
from dataclasses import dataclass, field

import pandas as pd

from . import attribution, bands, budget, capacity, equilibrium, monitor, neutral, optimize, risk, robustness, views
from .config import Config
from .data import finmind, twse
from .panel import Panel, build_panel, period_return


class ReconcileError(RuntimeError):
    """重建指標和官方指數差太多：資料有問題，後面的數字都不能信。"""


@dataclass
class MarketData:
    snapshot: pd.DataFrame      # 全市場快照（上市＋上櫃）
    panel: Panel
    adv: pd.Series              # 代號 → 交易值序列（算日均成交值用）
    tri: pd.Series              # 官方加權報酬指數
    flags: dict = field(default_factory=dict)   # 代號 → 跳空修正天數


def load_market(cfg: Config) -> MarketData:
    snap = twse.load_snapshot(cfg.data.path(cfg.data.snapshot_dir))
    rules = cfg.buckets.rules()
    listed = snap[snap.市場 == cfg.data.market]
    pdir = cfg.data.path(cfg.data.prices_dir)
    codes = [c for c in listed.index[:cfg.data.universe_top_n] if (pdir / f"TaiwanStockPrice_{c}.json").exists()]
    rets, rcaps, money, flags = {}, {}, {}, {}
    for c in codes:
        x = finmind.load_stock(pdir, c)
        if x is None:
            continue
        rets[c], rcaps[c], flags[c], money[c] = x
    bucket = rules.assign(listed.loc[list(rets), "產業"])
    panel = build_panel(rets, rcaps, listed.loc[list(rets), "市值"], bucket, cfg.buckets.order)
    return MarketData(snapshot=snap, panel=panel, adv=pd.Series(money), tri=finmind.load_tri(cfg.data.path(cfg.data.tri_file)),
                      flags=flags)


def reconcile(cfg: Config, md: MarketData, strict: bool = True) -> pd.DataFrame:
    """重建指標 vs 官方加權報酬指數。strict 時差距超過 tolerance 就丟出 ReconcileError。"""
    rtri = md.tri.pct_change().reindex(md.panel.rB.index)
    rows = []
    for a, b in cfg.dates.reconcile:
        off, reb = period_return(rtri, a, b), period_return(md.panel.rB, a, b)
        x, y = md.panel.rB.loc[pd.Timestamp(a):pd.Timestamp(b)].iloc[1:], rtri.loc[pd.Timestamp(a):pd.Timestamp(b)].iloc[1:]
        rows.append(dict(period=f"{a} → {b}", official=off, rebuilt=reb, diff=reb - off, corr=float(x.corr(y))))
    t = pd.DataFrame(rows)
    if strict and len(t) and (t["diff"].abs() > cfg.reconcile.tolerance).any():
        raise ReconcileError(f"重建指標與官方指數差距超過 {cfg.reconcile.tolerance:.1%}：\n{t.to_string()}")
    return t


def capacities(cfg: Config, md: MarketData) -> tuple[pd.DataFrame, pd.Series, dict]:
    """回傳（個股容量表、各桶容量、{對照天數: 各桶容量}）。"""
    c, p = cfg.constraints, md.panel
    asof = pd.Timestamp(cfg.dates.capacity_asof or cfg.dates.asof)
    mcap = p.MC.loc[:asof].iloc[-1]
    adv = pd.Series({k: finmind.adv(v, c.liquidity.adv_window, asof) for k, v in md.adv.items()}).reindex(mcap.index)
    kw = dict(nav=cfg.fund.nav, mcap=mcap, adv=adv, single_cap=c.single_cap, single_cap_overrides=c.single_cap_overrides,
              shares_cap=c.shares_cap, participation=c.liquidity.participation)
    pos = capacity.position_capacity(liq_days=c.liquidity.days, **kw)
    cap = capacity.bucket_capacity(pos, p.bucket, cfg.fund.nav, cfg.buckets.order)
    alt = {d: capacity.bucket_capacity(capacity.position_capacity(liq_days=d, **kw), p.bucket, cfg.fund.nav, cfg.buckets.order)
           for d in c.liquidity.compare_days}
    return pos, cap, alt


def fund_bucket_weights(cfg: Config, snap: pd.DataFrame) -> pd.Series | None:
    """基金股票部位的桶權重（只看股票，加總為 1）。"""
    if not cfg.data.holdings_file:
        return None
    h = pd.read_csv(cfg.data.path(cfg.data.holdings_file), dtype={"code": str})
    st = h[h.type == "stock"].copy()
    ind = snap.產業.reindex(st.code).fillna("").values
    rules = cfg.buckets.rules()
    st["bucket"] = [rules.bucket_of(c, i) for c, i in zip(st.code, ind)]
    w = (st.groupby("bucket").weight_pct.sum() / 100).reindex(cfg.buckets.order).fillna(0)
    return w / w.sum()


def run_policy(cfg: Config, md: MarketData | None = None, strict_reconcile: bool = True) -> dict:
    md = md or load_market(cfg)
    recon = reconcile(cfg, md, strict=strict_reconcile)
    order, c, m = cfg.buckets.order, cfg.constraints, cfg.model
    R = md.panel.window(*cfg.dates.estimation)
    kw = dict(halflife=m.covariance.ewma_halflife) if m.covariance.method == "ewma" else {}
    S = risk.estimate_cov(R[order], m.covariance.method, m.annualization, **kw)
    wB = md.panel.weights_at(cfg.dates.asof)
    pos, cap, cap_alt = capacities(cfg, md)
    upper = cap.copy()
    for b, u in c.bucket_upper.items():
        upper[b] = min(upper[b], u)
    wN = neutral.investable_neutral(wB, upper)
    te_struct = risk.te((wN - wB).values, S)
    te_cap = budget.strategic_cap(cfg.budget.total_te, te_struct, cfg.budget.strategic_share)
    bl = views.black_litterman(S, wB, cfg.view_objects(), erp=m.erp, tau=m.tau)
    wP = optimize.solve_policy(bl.alpha, S, wN, te_cap=te_cap, ir=m.ir, upper=cap, max_tilt=c.max_tilt,
                               fixed=tuple(c.fixed_buckets))
    te_strat, te_policy = risk.te((wP - wN).values, S), risk.te((wP - wB).values, S)
    bd = bands.rebalancing_bands(wP, wN, S, cfg.bands.te_increment, tuple(cfg.bands.clip), cfg.bands.half_weight_cap,
                                 no_up=tuple(cfg.bands.no_up), fixed=tuple(c.fixed_buckets))
    out = dict(config=cfg, market=md, recon=recon, R=R, S=S, wB=wB, capacity_table=pos, capacity=cap,
               capacity_compare=cap_alt, upper=upper, wN=wN, bl=bl, wP=wP, bands=bd,
               te=dict(total_budget=cfg.budget.total_te, structural=te_struct, strategic_cap=te_cap, strategic=te_strat,
                       policy=te_policy, tactical_room=budget.tactical_room(cfg.budget.total_te, te_policy)))
    wF = fund_bucket_weights(cfg, md.snapshot)
    if wF is not None:
        out.update(wF=wF, fund_status=bands.band_status(wF, bd), te_fund=risk.te((wF - wB).values, S),
                   beta_fund=risk.beta(wF, wB, S), fund_expected=equilibrium.decompose_expected_active(bl.Pi, bl.alpha, wF, wB))
    return out


def _solver(cfg: Config, wB: pd.Series, wN: pd.Series, te_cap: float, cap: pd.Series):
    """綁好中性、預算、上限的 solve(S, views) → 政策權重。"""
    c, m = cfg.constraints, cfg.model

    def solve(S, vs):
        bl = views.black_litterman(S, wB, vs, erp=m.erp, tau=m.tau)
        return optimize.solve_policy(bl.alpha, S, wN, te_cap=te_cap, ir=m.ir, upper=cap, max_tilt=c.max_tilt,
                                     fixed=tuple(c.fixed_buckets))
    return solve


def _cov_fn(cfg: Config):
    m = cfg.model
    kw = dict(halflife=m.covariance.ewma_halflife) if m.covariance.method == "ewma" else {}
    return lambda R: risk.estimate_cov(R, m.covariance.method, m.annualization, **kw)


def run_robustness(cfg: Config, o: dict) -> dict:
    rb, out = cfg.robustness, {}
    solve = _solver(cfg, o["wB"], o["wN"], o["te"]["strategic_cap"], o["capacity"])
    if rb.view_grid:
        g = rb.view_grid
        out["sensitivity"] = robustness.view_sensitivity(o["S"], solve, o["wN"], g.bucket, g.excess, g.confidence,
                                                         lambda a: risk.te(a, o["S"]))
    if rb.bootstrap:
        b = rb.bootstrap
        draws = robustness.block_bootstrap(o["R"][cfg.buckets.order], _cov_fn(cfg), solve, cfg.view_objects(),
                                           n=b.n, block=b.block_days, seed=b.seed)
        out["bootstrap"] = robustness.bootstrap_summary(draws, o["wP"], o["bands"])
    ports = {"benchmark": o["wB"], "neutral": o["wN"], "policy": o["wP"]}
    if "wF" in o:
        ports["fund"] = o["wF"]
    out["stress"] = {p.name: robustness.stress_replay(o["market"].panel.RB, p.start, p.end, ports)
                     for p in rb.stress}
    return out


def run_attribution(cfg: Config, md: MarketData) -> dict | None:
    at = cfg.attribution
    if at is None:
        return None
    tilts, variants = {}, {}
    for p in at.policies:
        v = cfg.model_copy(deep=True)
        v.dates.asof, v.dates.estimation, v.dates.capacity_asof = p.asof, p.estimation, p.capacity_asof
        v.views, v.data.holdings_file = p.views, None
        r = run_policy(v, md, strict_reconcile=False)
        tilts[p.name] = r["wP"] - r["wN"]
        variants[p.name] = r
    panel = md.panel
    dates = panel.R.index[(panel.R.index > pd.Timestamp(at.start)) & (panel.R.index <= pd.Timestamp(at.end))]
    upper = variants[at.policies[0].name]["upper"]
    tg = attribution.monthly_targets(panel.WB, dates, upper, tilts)
    R = panel.RB.loc[dates]
    rets = {k: attribution.rebalanced_returns(R, t, dates) for k, t in tg.items()}
    r_off = md.tri.pct_change().reindex(dates)
    rF = attribution.fund_market_returns(finmind.read(cfg.data.path(at.fund_price_file)),
                                         finmind.read(cfg.data.path(at.fund_dividend_file))).reindex(dates)
    pols = {k: rets[k] for k in tilts}
    main = at.policies[0].name
    return dict(variants=variants, tilts=tilts, returns=rets, fund=rF, official=r_off,
                tiers=attribution.three_tier(rF, r_off, rets["benchmark"], rets["neutral"], pols),
                monthly=attribution.monthly_three_tier(rF, r_off, rets["benchmark"], rets["neutral"], rets[main]))


def run_all(cfg: Config, strict_reconcile: bool = True) -> dict:
    """年度 SAA 全套：政策＋穩健性＋監控＋回溯歸因。"""
    md = load_market(cfg)
    o = run_policy(cfg, md, strict_reconcile)
    o["robustness"] = run_robustness(cfg, o)
    if "wF" in o:
        o["monitor"] = monitor.band_report(o["wF"], o["wP"], o["bands"])
    o["attribution"] = run_attribution(cfg, md)
    o["frontier"] = optimize.frontier(o["bl"].mu, o["S"], o["wB"], o["upper"], kind="active")
    pts = {"benchmark": o["wB"], "neutral": o["wN"], "policy": o["wP"]}
    if "wF" in o:
        pts["fund"] = o["wF"]
    o["frontier_points"] = {k: optimize.portfolio_stats(o["bl"].mu, o["S"], w, o["wB"]) for k, w in pts.items()}
    return o


def write_report(o: dict, out_dir) -> dict:
    """輸出試算表、政策組合書、效率前緣圖、區間圖。回傳 {種類: 路徑}。"""
    from pathlib import Path
    from .report import charts, policy_book, xlsx
    cfg = o["config"]
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.fund.code}_{cfg.dates.asof.year + 1}"
    paths = dict(xlsx=d / f"{stem}_政策組合.xlsx", book=d / f"{stem}_政策組合書.md",
                 frontier=d / f"{stem}_效率前緣.png", bands=d / f"{stem}_區間.png")
    xlsx.write_workbook(o, paths["xlsx"])
    paths["book"].write_text(policy_book.policy_book(o), encoding="utf-8")
    charts.frontier_chart(o["frontier"], o["frontier_points"], cfg.budget.total_te, paths["frontier"],
                          title=f"{cfg.fund.code} 追蹤誤差前緣（{cfg.dates.asof}）")
    t = pd.DataFrame(dict(policy=o["wP"], band_lower=o["bands"].lower, band_upper=o["bands"].upper))
    if "wF" in o:
        t["fund"] = o["wF"]
    charts.bands_chart(t, paths["bands"], title=f"{cfg.fund.code} 政策目標、區間與基金現況")
    return paths
