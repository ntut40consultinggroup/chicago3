"""設定檔（YAML）：所有「某檔基金專屬」的東西都在這裡，程式裡不寫死。

資料路徑可以是相對路徑，相對於 data.root；data.root 可用環境變數 CHICAGO3_DATA 或 CLI 的 --data-root 覆蓋。
"""
import os
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .buckets import BucketRules
from .views import View


class Fund(BaseModel):
    code: str
    name: str = ""
    nav: float = Field(gt=0, description="基金淨資產（元），容量計算用")


class Data(BaseModel):
    root: str = "."
    prices_dir: str | None = None         # FinMind 個股檔：TaiwanStockPrice_<代號>.json、TaiwanStockDividendResult_<代號>.json
    prices_file: str | None = None        # 或：合併檔（sync 產生的 prices.parquet）
    dividends_file: str | None = None     # 搭配 prices_file 的 dividends.parquet
    snapshot_dir: str                     # 證交所／櫃買當日快照（data/twse.py 的四個檔）
    tri_file: str                         # 加權報酬指數（FinMind TaiwanStockTotalReturnIndex）
    holdings_file: str | None = None      # 基金持股 CSV（type,code,name,weight_pct,shares）
    market: str = "上市"
    universe_top_n: int = Field(400, gt=0)

    @model_validator(mode="after")
    def _one_source(self):
        if bool(self.prices_dir) == bool(self.prices_file):
            raise ValueError("data 必須設定 prices_dir 或 prices_file（擇一）")
        if self.prices_file and not self.dividends_file:
            raise ValueError("使用 prices_file 時也要設定 dividends_file")
        return self

    def path(self, p: str) -> Path:
        root = Path(os.environ.get("CHICAGO3_DATA", self.root))
        q = Path(p)
        return q if q.is_absolute() else root / q


class Dates(BaseModel):
    asof: date                            # 指標權重的日期
    estimation: tuple[date, date]         # 共變異估計期間（不含起日）
    capacity_asof: date | None = None     # 容量用的市值與成交值日期，預設 = asof
    reconcile: list[tuple[date, date]] = []

    @model_validator(mode="after")
    def _check(self):
        if self.estimation[0] >= self.estimation[1]:
            raise ValueError("estimation 起日必須早於迄日")
        return self


class Buckets(BaseModel):
    order: list[str]
    default: str
    single_stock: dict[str, str] = {}
    overrides: dict[str, list[str]] = {}
    by_industry: dict[str, list[str]] = {}

    def rules(self) -> BucketRules:
        return BucketRules(**self.model_dump())


class Liquidity(BaseModel):
    days: float = Field(10, gt=0)
    participation: float = Field(0.20, gt=0, le=1)
    adv_window: int = Field(60, gt=0)
    compare_days: list[float] = [3]       # 另外列出對照用的出清天數


class Constraints(BaseModel):
    single_cap: float = Field(0.10, gt=0, le=1)
    single_cap_overrides: dict[str, float] = {}     # 代號 → 上限
    shares_cap: float = Field(0.10, gt=0, le=1)
    liquidity: Liquidity = Liquidity()
    bucket_upper: dict[str, float] = {}             # 桶 → 上限（例如台積電桶 25%），與容量取小
    fixed_buckets: list[str] = []                   # 政策固定在中性的桶
    max_tilt: float = Field(0.15, gt=0, le=1)


class Covariance(BaseModel):
    method: str = "ledoit_wolf"
    ewma_halflife: float = 60

    @field_validator("method")
    @classmethod
    def _m(cls, v):
        if v not in ("sample", "ledoit_wolf", "ewma"):
            raise ValueError("covariance.method 只能是 sample、ledoit_wolf、ewma")
        return v


class Model(BaseModel):
    annualization: int = 250
    covariance: Covariance = Covariance()
    erp: float = Field(0.06, description="股票風險溢酬")
    tau: float = Field(0.05, gt=0, description="Ω ∝ τΣ 時不影響結果，保留與文獻一致")
    ir: float = Field(0.5, gt=0)


class Budget(BaseModel):
    total_te: float = Field(0.15, gt=0)
    strategic_share: float = Field(0.5, gt=0, le=1)


class ViewCfg(BaseModel):
    bucket: str
    excess: float
    confidence: float = Field(gt=0, lt=1)


class Bands(BaseModel):
    te_increment: float = Field(0.01, gt=0)
    clip: tuple[float, float] = (0.02, 0.10)
    half_weight_cap: bool = True
    no_up: list[str] = []


class Reconcile(BaseModel):
    tolerance: float = Field(0.04, gt=0, description="重建指標與官方指數單期報酬差距上限")


class Bootstrap(BaseModel):
    n: int = Field(300, gt=0)
    block_days: int = Field(20, gt=1)
    seed: int = 20261122


class ViewGrid(BaseModel):
    bucket: str
    excess: list[float] = [0.0, 0.025, 0.05, 0.075, 0.10]
    confidence: list[float] = [0.25, 0.5, 0.75]


class Period(BaseModel):
    name: str
    start: date
    end: date


class Robustness(BaseModel):
    bootstrap: Bootstrap | None = Bootstrap()
    view_grid: ViewGrid | None = None
    stress: list[Period] = []


class PolicyVariant(BaseModel):
    """回溯用的政策：在主設定上換掉日期與觀點。"""
    name: str
    asof: date
    estimation: tuple[date, date]
    capacity_asof: date | None = None
    views: list[ViewCfg] = []


class Attribution(BaseModel):
    start: date
    end: date
    fund_price_file: str           # FinMind TaiwanStockPrice 格式
    fund_dividend_file: str        # FinMind TaiwanStockDividendResult 格式
    policies: list[PolicyVariant]  # 第一個是主要政策（逐月表用它）


class Config(BaseModel):
    fund: Fund
    data: Data
    dates: Dates
    buckets: Buckets
    constraints: Constraints = Constraints()
    model: Model = Model()
    budget: Budget = Budget()
    views: list[ViewCfg] = []
    bands: Bands = Bands()
    reconcile: Reconcile = Reconcile()
    robustness: Robustness = Robustness()
    attribution: Attribution | None = None

    @model_validator(mode="after")
    def _buckets_exist(self):
        known = set(self.buckets.order)
        used = {v.bucket for v in self.views} | set(self.constraints.bucket_upper) | set(self.constraints.fixed_buckets) \
            | set(self.bands.no_up)
        if self.robustness.view_grid:
            used.add(self.robustness.view_grid.bucket)
        if self.attribution:
            used |= {v.bucket for p in self.attribution.policies for v in p.views}
        if bad := used - known:
            raise ValueError(f"設定裡用到不存在的桶：{bad}")
        return self

    def view_objects(self) -> list[View]:
        return [View(v.bucket, v.excess, v.confidence) for v in self.views]


def load_config(path, data_root: str | None = None) -> Config:
    cfg = Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if data_root:
        cfg.data.root = data_root
    return cfg
