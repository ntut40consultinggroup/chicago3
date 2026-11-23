"""儲存介面：路徑一律用 "/" 分隔的相對路徑，例如 "_market/prices.parquet"、"00981A/holdings/2026-11-20.csv"。"""
from pathlib import Path


class Storage:
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def write_bytes(self, path: str, data: bytes) -> None:
        raise NotImplementedError

    def list(self, folder: str) -> list[str]:
        """folder 底下的檔名與子資料夾名（不遞迴）；資料夾不存在時回傳空清單。"""
        raise NotImplementedError

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8")

    def write_text(self, path: str, text: str) -> None:
        self.write_bytes(path, text.encode("utf-8"))

    def download(self, path: str, local: Path) -> Path:
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self.read_bytes(path))
        return local


class LocalStorage(Storage):
    """本機資料夾（CLI、業界自行安裝、測試用）。"""

    def __init__(self, root):
        self.root = Path(root)

    def _p(self, path: str) -> Path:
        return self.root / path

    def exists(self, path):
        return self._p(path).exists()

    def read_bytes(self, path):
        return self._p(path).read_bytes()

    def write_bytes(self, path, data):
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def list(self, folder):
        p = self._p(folder)
        return sorted(x.name for x in p.iterdir()) if p.is_dir() else []
