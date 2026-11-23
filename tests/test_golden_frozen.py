"""確認凍結的黃金數字和講義（階段8_補充課_策略性資產配置SAA）一致。

講義數字四捨五入到 0.1 個百分點，所以容許誤差取 0.06 個百分點。
第 1 期起，新套件的輸出要和 golden.json 比對（容許 0.1 個百分點）。
"""
import json
from pathlib import Path

import pytest

G = json.loads((Path(__file__).parent / "golden" / "golden.json").read_text())
TOL = 0.0006

LECTURE = [
    ("recon_2026.official", 0.3372), ("recon_2026.rebuilt", 0.3585),
    ("te_2027.struct", 0.051), ("te_2027.strat", 0.0051), ("te_2027.policy", 0.052),
    ("te_fund_now_2027", 0.173), ("te_fundlike_2025", 0.088),
    ("policy_2027.台積電", 0.250), ("policy_2027.其他半導體", 0.168), ("policy_2027.AI硬體零組件", 0.130),
    ("policy_2027.電腦與伺服器", 0.057), ("policy_2027.其他電子", 0.112), ("policy_2027.金融保險", 0.136),
    ("policy_2027.傳產與其他", 0.147),
    ("policy_2025_ai_tilt", 0.094),
    ("fund_expected_active", 0.0261), ("fund_beta_premium", 0.0217), ("fund_view_alpha", 0.0044),
    ("stress.policy", -0.272), ("stress.fund", -0.395),
    ("tiers_2026.active", -0.0707), ("tiers_2026.structural", -0.0037), ("tiers_2026.strategic", 0.0001),
    ("tiers_2026.execution", -0.0671), ("tiers_2026.strategic_strong", -0.0502),
]


def get(path):
    v = G
    for k in path.split("."):
        v = v[k]
    return v


@pytest.mark.parametrize("path,expected", LECTURE)
def test_matches_lecture(path, expected):
    assert abs(get(path) - expected) <= TOL, f"{path}: {get(path):.4f} vs 講義 {expected}"


def test_resample_range():
    lo, hi = G["resample_ai_p5_p95"]
    assert abs(lo - 0.125) <= TOL and abs(hi - 0.144) <= TOL


def test_five_buckets_out_of_band():
    assert sum(1 for v in G["out_of_band_2027"].values() if v) == 5


def test_policy_sums_to_one():
    assert abs(sum(G["policy_2027"].values()) - 1) < 1e-9
