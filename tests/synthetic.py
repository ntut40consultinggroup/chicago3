"""人造市場：6 檔股票、3 個桶，輸出成和真實資料相同格式的原始檔（證交所快照、FinMind 日價與除權息、報酬指數）。

用途：端到端測試不需要外部資料，可以在 CI 跑；而且答案可以用 numpy 獨立手算。
- 9001：半導體，最大權值股，單獨成「大型股」桶（上限 25%）
- 9002、9003：電子零組件；9004：半導體 → 「電子」桶
- 9005：金融保險（第 150 天配現金股利 2 元）；9006：水泥 → 「其他」桶（預設）
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

CODES = ["9001", "9002", "9003", "9004", "9005", "9006"]
IND = {"9001": "24", "9002": "28", "9003": "28", "9004": "24", "9005": "17", "9006": "01"}
SHARES = {"9001": 1.5e9, "9002": 1.0e9, "9003": 2.0e9, "9004": 1.5e9, "9005": 3.0e9, "9006": 2.0e9}
P0 = {"9001": 1000.0, "9002": 300.0, "9003": 150.0, "9004": 200.0, "9005": 60.0, "9006": 40.0}
BETA = np.array([1.1, 1.4, 1.3, 1.2, 0.5, 0.7])
N, DIV_DAY, DIV = 300, 150, 2.0
TURNOVER = 0.02
NAV = 2.0e10


def make_market(root: Path, seed: int = 3):
    """寫出原始檔，回傳手算需要的真實值（收盤、總報酬、股數、日期）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=N)
    f = rng.normal(0.0004, 0.011, N)
    eps = rng.normal(0, 0.008, (N, len(CODES)))
    R = f[:, None] * BETA + eps                     # 每日總報酬（第 0 天不用）
    R[0] = 0.0
    close = np.zeros((N, len(CODES)))
    close[0] = [P0[c] for c in CODES]
    j5 = CODES.index("9005")
    for t in range(1, N):
        close[t] = close[t - 1] * (1 + R[t])
        if t == DIV_DAY:                            # 除息日：參考價 = 前收 − 股利，收盤 = 參考價 × (1 + 總報酬)
            close[t, j5] = (close[t - 1, j5] - DIV) * (1 + R[t, j5])
    money = close * np.array([SHARES[c] for c in CODES]) * TURNOVER    # 日成交值 = 市值 × 週轉率

    pdir, sdir = root / "prices", root / "snapshot"
    pdir.mkdir(parents=True)
    sdir.mkdir(parents=True)
    ds = [d.strftime("%Y-%m-%d") for d in dates]
    for j, c in enumerate(CODES):
        rows = [dict(date=ds[t], stock_id=c, close=float(close[t, j]),
                     spread=float(close[t, j] - (close[t - 1, j] if t else close[t, j])), Trading_money=float(money[t, j]))
                for t in range(N)]
        (pdir / f"TaiwanStockPrice_{c}.json").write_text(json.dumps(dict(status=200, data=rows)))
        div = []
        if c == "9005":
            before = float(close[DIV_DAY - 1, j])
            div = [dict(date=ds[DIV_DAY], before_price=before, after_price=before - DIV, stock_and_cache_dividend=DIV,
                        stock_or_cache_dividend="息")]
        (pdir / f"TaiwanStockDividendResult_{c}.json").write_text(json.dumps(dict(status=200, data=div)))
    # 證交所快照（最後一天）
    (sdir / "t187ap03_L.json").write_text(json.dumps(
        [{"公司代號": c, "已發行普通股數或TDR原股發行股數": f"{SHARES[c]:.0f}", "產業別": IND[c]} for c in CODES], ensure_ascii=False))
    (sdir / "exchangeReport_STOCK_DAY_ALL.json").write_text(json.dumps(
        [dict(Code=c, Name=f"股{c}", ClosingPrice=f"{close[-1, j]:.10f}", TradeValue=f"{money[-1, j]:.0f}", Date="1151120")
         for j, c in enumerate(CODES)], ensure_ascii=False))
    (sdir / "mopsfin_t187ap03_O.json").write_text("[]")
    (sdir / "tpex_mainboard_daily_close_quotes.json").write_text("[]")
    # 官方報酬指數 = 真實市值加權總報酬（前一日市值加權）
    mcap = close * np.array([SHARES[c] for c in CODES])
    w = mcap[:-1] / mcap[:-1].sum(axis=1, keepdims=True)
    rb = np.r_[0.0, (w * R[1:]).sum(axis=1)]
    tri = 1000 * np.cumprod(1 + rb)
    (root / "tri.json").write_text(json.dumps(dict(status=200, data=[dict(date=ds[t], stock_id="TAIEX", price=float(tri[t]))
                                                                     for t in range(N)])))
    # 基金持股：重押電子
    pd.DataFrame(dict(type="stock", code=["9001", "9002", "9003", "9004", "9005"], name="",
                      weight_pct=[20.0, 30.0, 25.0, 20.0, 1.0], shares=1000)).to_csv(root / "holdings.csv", index=False)
    return dict(dates=dates, close=close, R=R, mcap=mcap)


CONFIG = """
fund: {{code: TEST3, name: 人造三桶, nav: {nav}}}
data:
  root: {root}
  prices_dir: prices
  snapshot_dir: snapshot
  tri_file: tri.json
  holdings_file: holdings.csv
  universe_top_n: 10
dates:
  asof: {asof}
  estimation: [{est0}, {asof}]
  reconcile: [[{d0}, {asof}]]
buckets:
  order: [大型股, 電子, 其他]
  default: 其他
  single_stock: {{大型股: "9001"}}
  by_industry: {{電子: [半導體, 電子零組件]}}
constraints:
  single_cap: 0.60                     # 每桶只有 2–3 檔，放寬單一個股上限，讓容量不綁住中性
  single_cap_overrides: {{"9001": 0.25}}
  liquidity: {{days: 10, participation: 0.20, adv_window: 60, compare_days: [3]}}
  bucket_upper: {{大型股: 0.25}}
  fixed_buckets: [大型股]
  max_tilt: 0.15
model: {{covariance: {{method: sample}}, erp: 0.06, tau: 0.05, ir: 0.5}}
budget: {{total_te: 0.15}}
views: [{{bucket: 電子, excess: 0.03, confidence: 0.5}}]
bands: {{no_up: [大型股]}}
robustness:
  bootstrap: {{n: 20, block_days: 20, seed: 1}}
  view_grid: {{bucket: 電子, excess: [0.0, 0.03], confidence: [0.5]}}
  stress: [{{name: 後段, start: {s0}, end: {asof}}}]
"""


def write_config(root: Path, m: dict) -> Path:
    d = m["dates"]
    p = root / "config.yaml"
    p.write_text(CONFIG.format(nav=NAV, root=root, asof=d[-1].date(), est0=d[49].date(), d0=d[0].date(), s0=d[249].date()),
                 encoding="utf-8")
    return p
