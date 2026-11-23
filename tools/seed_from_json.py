"""把課程的 FinMind 個股 JSON 資料夾轉成 sync 的儲存結構（合併 parquet＋快照＋manifest）。

用法：python tools/seed_from_json.py <儲存位置，例如 local:/tmp/store> <個股 JSON 資料夾> <tri.json> <快照資料夾> <資料截止日 YYYY-MM-DD>
用途：已經下載過的資料不必再花 20 分鐘重抓；也用來驗證「從合併檔計算」和「從個股檔計算」結果一致。
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chicago3.data import twse  # noqa: E402
from chicago3.storage import open_storage  # noqa: E402
from chicago3.sync import DIV_COLS, M, PRICE_COLS, _roc_to_iso, _write_manifest, _write_parquet, read_manifest  # noqa: E402


def main(spec, prices_dir, tri_file, snapshot_dir, through):
    store = open_storage(spec)
    pdir = Path(prices_dir)
    P, D, codes = [], [], []
    for f in sorted(pdir.glob("TaiwanStockPrice_*.json")):
        c = f.stem.split("_")[-1]
        p = pd.DataFrame(json.loads(f.read_text())["data"])
        if p.empty:
            continue
        codes.append(c)
        P.append(p.reindex(columns=PRICE_COLS).assign(stock_id=c))
        fd = pdir / f"TaiwanStockDividendResult_{c}.json"
        if fd.exists():
            d = pd.DataFrame(json.loads(fd.read_text())["data"])
            if len(d):
                D.append(d.reindex(columns=DIV_COLS).assign(stock_id=c))
    _write_parquet(store, f"{M}/prices.parquet", pd.concat(P).sort_values(["stock_id", "date"]))
    _write_parquet(store, f"{M}/dividends.parquet", pd.concat(D).sort_values(["stock_id", "date"]))
    t = pd.DataFrame(json.loads(Path(tri_file).read_text())["data"])[["date", "stock_id", "price"]]
    _write_parquet(store, f"{M}/tri.parquet", t)
    sdir = Path(snapshot_dir)
    iso = _roc_to_iso(json.loads((sdir / "exchangeReport_STOCK_DAY_ALL.json").read_text())[0]["Date"])
    day = iso.replace("-", "")
    for fn in twse.URLS:
        store.write_bytes(f"{M}/snapshot/{day}/{fn}", (sdir / fn).read_bytes())
    man = read_manifest(store)
    man.update(prices_through={c: through for c in codes}, tri_through=through, snapshot=iso, seeded_from=str(pdir))
    _write_manifest(store, man)
    print(f"完成：{len(codes)} 檔、快照 {day}")


if __name__ == "__main__":
    main(*sys.argv[1:])
