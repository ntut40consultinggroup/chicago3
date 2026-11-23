"""儲存層：LocalStorage（本機資料夾）、GDriveStorage（Google Drive）。用 open_storage("local:/路徑") 或 open_storage("gdrive")。"""
import os

from .base import LocalStorage, Storage
from .gdrive import GDriveStorage


def open_storage(spec: str, secrets: dict | None = None) -> Storage:
    """spec："local:<資料夾>" 或 "gdrive"。gdrive 的憑證來自 secrets 或環境變數
    GOOGLE_CLIENT_ID、GOOGLE_CLIENT_SECRET、GOOGLE_REFRESH_TOKEN。"""
    if spec.startswith("local:"):
        return LocalStorage(spec[len("local:"):])
    if spec == "gdrive":
        s = secrets or {}
        get = lambda k: s.get(k.lower().replace("google_", "")) or os.environ.get(k)
        cid, sec, tok = get("GOOGLE_CLIENT_ID"), get("GOOGLE_CLIENT_SECRET"), get("GOOGLE_REFRESH_TOKEN")
        if not (cid and sec and tok):
            raise ValueError("缺少 Google 憑證：GOOGLE_CLIENT_ID、GOOGLE_CLIENT_SECRET、GOOGLE_REFRESH_TOKEN")
        return GDriveStorage(cid, sec, tok)
    raise ValueError(f"不支援的儲存位置：{spec}（用 local:<資料夾> 或 gdrive）")
