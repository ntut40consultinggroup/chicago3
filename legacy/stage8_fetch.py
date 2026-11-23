"""SAA 課：下載市值前 200 大上市股兩年日價與除權息結果（FinMind）。

用法：python3 -I stage8_fetch.py <資料夾> <代號來源>
代號來源 = 資料夾（取其中 TaiwanStockPrice_*.json 的代號，沿用階段 6 的 164 檔），
或文字檔（每行一個代號，例如補抓的 36 檔大型持股）。兩者合起來 = 11/20 市值前 200 大上市股。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

URL = "https://api.finmindtrade.com/api/v4/data?dataset={ds}&data_id={id}&start_date=2024-11-01&end_date=2026-11-20"


def get(ds, code, fn):
    for i in range(6):
        subprocess.run(["curl", "-sS", "-m", "60", URL.format(ds=ds, id=code), "-o", str(fn)])
        try:
            st = json.loads(fn.read_text()).get("status")
        except Exception:
            st = None
        if st == 200:
            return True
        time.sleep(60 if st == 402 else 5)   # 402 = 超過頻率上限
    return False


def main(out, src):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)   # 允許中斷後續傳；已成功的檔不重抓
    src = Path(src)
    if src.is_file():                      # 也可給一個每行一個代號的文字檔
        codes = [c.strip() for c in src.read_text().split() if c.strip()]
    else:
        codes = sorted(p.stem.split("_")[-1] for p in src.glob("TaiwanStockPrice_*.json"))
    bad = []
    for c in codes:
        for ds in ("TaiwanStockPrice", "TaiwanStockDividendResult"):
            fn = out / f"{ds}_{c}.json"
            if fn.exists():
                try:
                    if json.loads(fn.read_text()).get("status") == 200:
                        continue
                except Exception:
                    pass
            if not get(ds, c, fn):
                bad.append(f"{ds}_{c}")
            time.sleep(0.8)
    print("完成", len(codes), "檔；失敗", bad)


if __name__ == "__main__":
    main(*sys.argv[1:])
