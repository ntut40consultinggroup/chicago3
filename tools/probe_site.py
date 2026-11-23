"""網站結構探測（在 GitHub Actions 上跑，因為開發沙盒連不到部分投信網站）。

用法：python tools/probe_site.py <輸出資料夾> <起始網址> <關鍵字> [其他網址…]
做的事：下載起始網址與額外網址；找出含關鍵字的連結並下載；在 HTML 與引用的 JS 裡找疑似 API 路徑。
所有結果寫到輸出資料夾，由 workflow 上傳成 artifact。
"""
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"}
API_RE = re.compile(r"""["'`](/?(?:api|Api|API|ajax|Ajax|ETF|etf|Fund|fund)[A-Za-z0-9_\-/.?=&{}$]*)["'`]""")


def save(out, name, r):
    p = out / re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:150]
    p.write_bytes(r.content)
    return p.name


def main(out, start, keyword, *extra):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    s = requests.Session()
    s.headers.update(UA)
    seen, apis = set(), set()
    queue = [start, *extra]
    while queue and len(seen) < 25:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = s.get(url, timeout=30)
        except Exception as e:
            log.append(dict(url=url, error=repr(e)))
            continue
        fn = save(out, urlparse(url).path + "_" + (urlparse(url).query or ""), r)
        ct = r.headers.get("content-type", "")
        log.append(dict(url=url, status=r.status_code, type=ct, size=len(r.content), file=fn))
        if "html" not in ct and "javascript" not in ct:
            continue
        text = r.text
        apis.update(API_RE.findall(text))
        if "html" in ct:
            for href in re.findall(r"""(?:href|src)=["']([^"']+)["']""", text):
                full = urljoin(url, href)
                if urlparse(full).netloc != urlparse(start).netloc:
                    continue
                if keyword.lower() in full.lower() or full.endswith(".js") and "/vendor" not in full:
                    queue.append(full)
    (out / "_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    (out / "_api_candidates.txt").write_text("\n".join(sorted(apis)))
    print(json.dumps(log, ensure_ascii=False, indent=1))
    print(len(apis), "API candidates")


if __name__ == "__main__":
    main(*sys.argv[1:])
