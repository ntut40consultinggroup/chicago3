"""持股來源的共同格式。

每個來源提供兩種函數：
- parse_*：只吃已下載的內容（bytes / str），回傳 Holdings。純函數，可離線測試。
- fetch_*：負責下載，再呼叫 parse_*。需要網路（開發沙盒連不到的來源，要在雲端測試）。
"""
from dataclasses import dataclass, field

import pandas as pd

COLUMNS = ["type", "code", "name", "shares", "amount", "weight_pct"]   # type：stock／future


@dataclass
class Holdings:
    fund: str                 # ETF 證券代號，例如 00981A
    date: str                 # 持股日期 YYYY-MM-DD
    table: pd.DataFrame       # 欄位見 COLUMNS
    nav: float | None = None  # 基金淨資產（元）
    source: str = ""
    extra: dict = field(default_factory=dict)

    def stocks(self) -> pd.DataFrame:
        return self.table[self.table.type == "stock"].reset_index(drop=True)

    def to_csv(self, path):
        """存成和課程 inputs/holdings_*.csv 相同的欄位：type,code,name,weight_pct,shares。"""
        self.table[["type", "code", "name", "weight_pct", "shares"]].to_csv(path, index=False)


def to_number(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    return float(str(x).replace(",", "").replace("%", "").strip())
