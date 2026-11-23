"""指令列：chicago3 <指令> <設定檔> [--data-root 路徑] [--out 資料夾]

  reconcile  重建指標並和官方加權報酬指數勾稽（差距超過門檻就失敗）
  policy     年度 SAA：容量、中性、均衡與觀點、政策、區間、TE 預算、基金現況；結果寫到 --out
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .pipeline import ReconcileError, load_market, reconcile, run_policy

pd.set_option("display.width", 160)


def _policy_table(o) -> pd.DataFrame:
    t = pd.DataFrame(dict(benchmark=o["wB"], neutral=o["wN"], policy=o["wP"], tilt=o["wP"] - o["wN"],
                          band_lower=o["bands"].lower, band_upper=o["bands"].upper, capacity=o["capacity"]))
    if "wF" in o:
        t["fund"] = o["wF"]
        t["status"] = o["fund_status"]
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chicago3", description="Chicago 3.0 SAA 政策組合工具")
    ap.add_argument("command", choices=["reconcile", "policy"])
    ap.add_argument("config")
    ap.add_argument("--data-root")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.data_root)
    try:
        if a.command == "reconcile":
            print(reconcile(cfg, load_market(cfg)).round(4).to_string(index=False))
            return 0
        o = run_policy(cfg)
    except ReconcileError as e:
        print(f"勾稽失敗：{e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tab = _policy_table(o)
    tab.to_csv(out / f"{cfg.fund.code}_policy.csv", encoding="utf-8-sig")
    summary = dict(fund=cfg.fund.code, asof=str(cfg.dates.asof), te=o["te"], policy=o["wP"].to_dict(),
                   recon=o["recon"].to_dict("records"))
    if "wF" in o:
        summary.update(te_fund=o["te_fund"], beta_fund=o["beta_fund"], fund_expected=o["fund_expected"])
    (out / f"{cfg.fund.code}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(o["recon"].round(4).to_string(index=False), "\n")
    print(tab.round(4).to_string(), "\n")
    print({k: round(v, 4) for k, v in o["te"].items()})
    if "wF" in o:
        print(f"基金事前 TE {o['te_fund']:.4f}、β {o['beta_fund']:.2f}；預期主動 "
              f"{o['fund_expected']['expected_active']:.4f} = Beta 溢酬 {o['fund_expected']['beta_premium']:.4f}"
              f" ＋ 觀點 α {o['fund_expected']['view_alpha']:.4f}")
    print("輸出：", out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
