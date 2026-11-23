"""第 0 期：跑課程原始程式（legacy/stage8_saa.py），凍結黃金數字與核心函數的輸入輸出。

用法：python3 -I tools/freeze_golden.py <legacy 資料夾> <輸出資料夾> <stage8_saa.py 的 7 個參數…>
輸出：
  golden/golden.json   講義上的關鍵數字（黃金測試用）
  fixtures/*.csv|json  核心函數的輸入與輸出（第 1 期單元測試用，不需要原始資料）
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(legacy, out, *saa_args):
    sys.path.insert(0, str(Path(legacy).resolve()))
    import stage8_saa as saa

    # 記錄 lw_cov 的輸入（桶日報酬），前三次呼叫對應 p25、p25s、p27 的估計期間
    cov_inputs = []
    orig_lw = saa.lw_cov

    def lw_rec(R):
        if len(cov_inputs) < 3:
            cov_inputs.append(R.copy())
        return orig_lw(R)

    saa.lw_cov = lw_rec
    with contextlib.redirect_stdout(io.StringIO()):
        o = saa.main(*saa_args)
    saa.lw_cov = orig_lw

    out = Path(out)
    fx = out / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)

    # ---------- fixtures ----------
    cov_inputs[0].to_csv(fx / "bucket_returns_est2025.csv")
    cov_inputs[2].to_csv(fx / "bucket_returns_est2027.csv")
    cap = o["capd"]["容量_占淨資產"]
    for k in ("p25", "p25s", "p27"):
        p = o[k]
        p["S"].to_csv(fx / f"{k}_cov.csv")
        pd.DataFrame(dict(wB=p["wB"], wN=p["wN"], wP=p["wP"], Pi=p["Pi"], alpha=p["alpha"], cap=cap)).to_csv(fx / f"{k}_weights.csv")
        (fx / f"{k}_scalars.json").write_text(json.dumps(
            dict(delta=p["delta"], budget=p["budget"], te_struct=p["te_struct"], te_strat=p["te_strat"], te_total=p["te_total"]),
            indent=2))
    pol = o["pol"]
    pol.to_csv(fx / "p27_policy_table.csv")
    o["wF"].to_csv(fx / "fund_bucket_weights_20261120.csv")
    (fx / "constants.json").write_text(json.dumps(dict(
        ANN=saa.ANN, TAU=saa.TAU, ERP=saa.ERP, TE_TOTAL=saa.TE_TOTAL, TILT_MAX=saa.TILT_MAX, IR=saa.IR,
        BANDS=list(saa.BANDS), LIQ_DAYS=saa.LIQ_DAYS, NAV=saa.NAV, UNIVERSE=saa.UNIVERSE,
        views_base=[["AI硬體零組件", 0.05, 0.5]], views_strong_2025=[["AI硬體零組件", 0.15, 0.75]]), ensure_ascii=False, indent=2))

    # ---------- golden ----------
    p27, p25 = o["p27"], o["p25"]
    S, wB, wF = p27["S"].values, p27["wB"].values, o["wF"].reindex(p27["wB"].index).fillna(0).values
    beta_F = float((S @ wB) @ wF / (wB @ S @ wB))
    beta_prem = float(p27["Pi"].values @ (wF - wB))
    view_alpha = float(p27["alpha"].values @ (wF - wB))
    rs = o["resamp"]
    tiers = o["tiers"].set_index("層級")["數值"]
    stress = o["stress"].set_index("組合")["衝擊期報酬"]
    recon = o["recon"]
    g = {
        "recon_2026": dict(official=float(recon.官方加權報酬指數.iloc[1]), rebuilt=float(recon.重建.iloc[1]),
                           corr=float(recon.日報酬相關.iloc[1])),
        "recon_2025": dict(official=float(recon.官方加權報酬指數.iloc[0]), rebuilt=float(recon.重建.iloc[0])),
        "te_2027": dict(struct=p27["te_struct"], budget=p27["budget"], strat=p27["te_strat"], policy=p27["te_total"]),
        "te_2025": dict(struct=p25["te_struct"], budget=p25["budget"], strat=p25["te_strat"], policy=p25["te_total"]),
        "te_fund_now_2027": float(o["te_tab"].set_index("項目").iloc[7, 1]),
        "te_fundlike_2025": float(o["te_tab"].set_index("項目").iloc[7, 0]),
        "policy_2027": p27["wP"].to_dict(),
        "neutral_2027": p27["wN"].to_dict(),
        "bands_2027": {b: [float(pol.loc[b, "區間下限"]), float(pol.loc[b, "區間上限"])] for b in pol.index},
        "out_of_band_2027": pol.超出區間.fillna("").to_dict(),
        "policy_2025_ai_tilt": float(p25["wP"]["AI硬體零組件"] - p25["wN"]["AI硬體零組件"]),
        "capacity_10d": cap.to_dict(),
        "capacity_3d": o["capd"]["對照_3天出清容量"].to_dict(),
        "fund_beta": beta_F,
        "fund_expected_active": beta_prem + view_alpha,
        "fund_beta_premium": beta_prem,
        "fund_view_alpha": view_alpha,
        "resample_ai_p5_p95": [float(rs.loc["AI硬體零組件", "重抽樣_P5"]), float(rs.loc["AI硬體零組件", "重抽樣_P95"])],
        "stress": dict(benchmark=float(stress.iloc[0]), neutral=float(stress.iloc[1]), policy=float(stress.iloc[2]),
                       fund=float(stress.iloc[3])),
        "tiers_2026": dict(active=float(tiers.iloc[2]), structural=float(tiers.iloc[3]), strategic=float(tiers.iloc[4]),
                           execution=float(tiers.iloc[5]), strategic_strong=float(tiers.iloc[6]),
                           execution_strong=float(tiers.iloc[7])),
        "tolerance_pp": 0.1,
        "source": "legacy/stage8_saa.py 於 2026-11-23 以 data/saa_20261122 重跑",
    }
    (out / "golden").mkdir(exist_ok=True)
    (out / "golden" / "golden.json").write_text(json.dumps(g, ensure_ascii=False, indent=2))
    print(json.dumps(g, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
