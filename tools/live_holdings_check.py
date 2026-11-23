"""實際上網下載持股（在 GitHub Actions 跑）：統一 00981A 最新一天、復華 00991A 指定日期。

用法：python tools/live_holdings_check.py <輸出資料夾> [復華日期 YYYYMMDD]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chicago3.holdings import ezmoney, fhtrust  # noqa: E402


def summary(h):
    st = h.stocks()
    return dict(fund=h.fund, date=h.date, source=h.source, nav=h.nav, n_stocks=len(st),
                stock_weight_pct=round(float(st.weight_pct.sum()), 2), top3=st.nlargest(3, "weight_pct")[["code", "name", "weight_pct"]].values.tolist())


def main(out, fh_date="20261120"):
    out = Path(out)
    out.mkdir(exist_ok=True)
    res = {}
    ez_list = ezmoney.fetch_fund_list()
    res["ezmoney_funds"] = ez_list
    h = ezmoney.fetch_holdings("00981A", ez_list["00981A"])
    h.to_csv(out / f"00981A_{h.date}.csv")
    res["00981A"] = summary(h)
    fh_list = fhtrust.fetch_fund_list()
    res["fhtrust_funds"] = fh_list
    h2 = fhtrust.fetch_holdings("00991A", fh_date, fh_list["00991A"])
    if h2:
        h2.to_csv(out / f"00991A_{h2.date}.csv")
        res["00991A"] = summary(h2)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:])
