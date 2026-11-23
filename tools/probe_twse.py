"""在雲端抓證交所網站資料（開發沙盒連不到 www.twse.com.tw），存到輸出資料夾供比對。

用法：python tools/probe_twse.py <輸出資料夾>
"""
import json
import sys
import time
from pathlib import Path

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}
URLS = {
    "twt49u_a.json": "https://www.twse.com.tw/exchangeReport/TWT49U?response=json&strDate=20241101&endDate=20261120",
    "twt49u_b.json": "https://www.twse.com.tw/rwd/zh/exchangeReport/TWT49U?response=json&strDate=20241101&endDate=20261120",
    "twt49u_c.json": "https://www.twse.com.tw/exchangeReport/TWT49U?response=json&strDate=20261001&endDate=20261120",
}


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for fn, url in URLS.items():
        t0 = time.time()
        r = requests.get(url, headers=UA, timeout=60)
        (out / fn).write_bytes(r.content)
        try:
            js = r.json()
            info = dict(stat=js.get("stat"), keys=list(js)[:12], n=len(js.get("data", [])),
                        tables=[(t.get("title"), len(t.get("data", []))) for t in js.get("tables", [])][:12])
        except ValueError:
            info = dict(text=r.text[:200])
        print(fn, r.status_code, len(r.content), f"{time.time() - t0:.1f}s", json.dumps(info, ensure_ascii=False))
        time.sleep(3)          # 證交所網站對頻繁請求很敏感


if __name__ == "__main__":
    main(*sys.argv[1:])
