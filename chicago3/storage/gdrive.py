"""Google Drive 儲存（Drive API v3，OAuth refresh token，權限 drive.file）。

- 根資料夾 chicago3.0 由程式自己建立（drive.file 只看得到程式建立的檔案）；之後可以手動移到別的資料夾。
- 路徑 "a/b/c.json"：逐層找資料夾，寫入時不存在就建立；同名檔案存在就更新內容。
"""
import json

import requests

from .base import Storage

API = "https://www.googleapis.com/drive/v3/files"
UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
FOLDER = "application/vnd.google-apps.folder"


class GDriveStorage(Storage):
    def __init__(self, client_id: str, client_secret: str, refresh_token: str, root_name: str = "chicago3.0",
                 session=None):
        self.cfg = dict(client_id=client_id, client_secret=client_secret, refresh_token=refresh_token)
        self.root_name = root_name
        self.s = session or requests.Session()
        self._token = None
        self._ids: dict[str, str] = {}

    # ---------- 底層 ----------
    def _auth(self) -> dict:
        if self._token is None:
            r = self.s.post("https://oauth2.googleapis.com/token", data=dict(**self.cfg, grant_type="refresh_token"),
                            timeout=30)
            r.raise_for_status()
            self._token = r.json()["access_token"]
        return {"Authorization": f"Bearer {self._token}"}

    def _q(self, q: str) -> list[dict]:
        r = self.s.get(API, headers=self._auth(), params=dict(q=q + " and trashed=false", fields="files(id,name,mimeType)",
                                                               pageSize=1000), timeout=30)
        r.raise_for_status()
        return r.json()["files"]

    @staticmethod
    def _esc(name: str) -> str:
        return name.replace("\\", "\\\\").replace("'", "\\'")

    def _child(self, parent: str | None, name: str, folder: bool) -> str | None:
        q = f"name='{self._esc(name)}'" + (f" and '{parent}' in parents" if parent else "")
        q += f" and mimeType{'=' if folder else '!='}'{FOLDER}'"
        hits = self._q(q)
        return hits[0]["id"] if hits else None

    def _mkdir(self, parent: str | None, name: str) -> str:
        body = dict(name=name, mimeType=FOLDER, **({"parents": [parent]} if parent else {}))
        r = self.s.post(API, headers=self._auth(), json=body, params=dict(fields="id"), timeout=30)
        r.raise_for_status()
        return r.json()["id"]

    def _folder(self, parts: list[str], create: bool) -> str | None:
        key, parent = "", None
        for name in [self.root_name, *parts]:
            key = f"{key}/{name}"
            if key not in self._ids:
                fid = self._child(parent, name, folder=True)
                if fid is None:
                    if not create:
                        return None
                    fid = self._mkdir(parent, name)
                self._ids[key] = fid
            parent = self._ids[key]
        return parent

    def _file_id(self, path: str) -> str | None:
        *dirs, name = path.split("/")
        parent = self._folder(dirs, create=False)
        return self._child(parent, name, folder=False) if parent else None

    # ---------- 介面 ----------
    def exists(self, path):
        return self._file_id(path) is not None

    def read_bytes(self, path):
        fid = self._file_id(path)
        if fid is None:
            raise FileNotFoundError(path)
        r = self.s.get(f"{API}/{fid}", headers=self._auth(), params=dict(alt="media"), timeout=120)
        r.raise_for_status()
        return r.content

    def write_bytes(self, path, data):
        *dirs, name = path.split("/")
        parent = self._folder(dirs, create=True)
        fid = self._child(parent, name, folder=False)
        if fid:
            r = self.s.patch(f"{UPLOAD}/{fid}", headers=self._auth(), params=dict(uploadType="media"), data=data, timeout=300)
        else:
            meta = json.dumps(dict(name=name, parents=[parent]))
            files = {"metadata": ("metadata", meta, "application/json"), "file": (name, data, "application/octet-stream")}
            r = self.s.post(UPLOAD, headers=self._auth(), params=dict(uploadType="multipart"), files=files, timeout=300)
        r.raise_for_status()

    def list(self, folder):
        fid = self._folder([p for p in folder.split("/") if p], create=False)
        return sorted(f["name"] for f in self._q(f"'{fid}' in parents")) if fid else []
