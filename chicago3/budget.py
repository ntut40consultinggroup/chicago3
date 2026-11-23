"""追蹤誤差預算分層：總預算 → 結構層（法規造成）→ 策略層（投資委員會）→ 戰術層（經理人）。

假設各層互不相關，所以用平方相加：總² = 結構² ＋ 策略² ＋ 戰術²。
"""
import numpy as np


def strategic_cap(total: float, te_structural: float, strategic_share: float = 0.5) -> float:
    """策略 TE 上限 = √((總² − 結構²) × 策略份額)。課程用平分（份額 0.5）。"""
    return float(np.sqrt(max(total ** 2 - te_structural ** 2, 0.0) * strategic_share))


def tactical_room(total: float, te_policy: float) -> float:
    """政策確定後，留給戰術與選股的空間 ≈ √(總² − 政策 TE²)。"""
    return float(np.sqrt(max(total ** 2 - te_policy ** 2, 0.0)))
