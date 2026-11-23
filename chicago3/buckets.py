"""分桶規則：個股 → 桶。

優先順序：single_stock（單一個股成桶，例如台積電）→ overrides（指定個股，例如散熱股歸 AI 硬體）
→ by_industry（產業對應）→ default。
"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BucketRules:
    order: list[str]                                          # 桶的顯示順序（也是所有計算的索引順序）
    default: str
    single_stock: dict[str, str] = field(default_factory=dict)       # 桶 → 代號
    overrides: dict[str, list[str]] = field(default_factory=dict)    # 桶 → 代號清單
    by_industry: dict[str, list[str]] = field(default_factory=dict)  # 桶 → 產業清單

    def __post_init__(self):
        named = {self.default, *self.single_stock, *self.overrides, *self.by_industry}
        if missing := named - set(self.order):
            raise ValueError(f"這些桶沒有出現在 order 裡：{missing}")
        seen = {}
        for b, inds in self.by_industry.items():
            for i in inds:
                if i in seen:
                    raise ValueError(f"產業「{i}」同時對應到 {seen[i]} 與 {b}")
                seen[i] = b
        self._ind = seen
        self._code = {c: b for b, cs in self.overrides.items() for c in cs}
        self._code.update({c: b for b, c in self.single_stock.items()})   # single_stock 優先

    def bucket_of(self, code: str, industry: str) -> str:
        return self._code.get(code) or self._ind.get(industry) or self.default

    def assign(self, industries: pd.Series) -> pd.Series:
        """industries：代號 → 產業。回傳代號 → 桶。"""
        return pd.Series({c: self.bucket_of(c, i) for c, i in industries.items()})
