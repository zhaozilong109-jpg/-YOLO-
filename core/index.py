"""上传去重索引。

同一张图被重复上传（换了个文件名、或拖了两次）会污染数据集：同一张图两条标注、
train/val 里出现两个副本。这里按内容 sha256 建索引，重复的直接跳过。

索引持久化在数据目录下，重启后依然有效；图片被删除时会同步清理记录。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    """分块计算文件哈希，避免大文件一次性读进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class UploadIndex:
    """文件名 ↔ 内容哈希的双向索引，线程安全。"""

    VERSION = 1

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._items: dict[str, dict] = {}
        self.load()

    # ---------- 磁盘 ----------

    def load(self) -> None:
        with self._lock:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                self._items = {}
                return

            items = raw.get("items") if isinstance(raw, dict) else None
            if not isinstance(items, dict):
                self._items = {}
                return
            self._items = {
                str(name): {"sha256": str(meta.get("sha256")), "size": int(meta.get("size", 0))}
                for name, meta in items.items()
                if isinstance(meta, dict) and meta.get("sha256")
            }

    def _save(self) -> None:
        payload = {"version": self.VERSION, "items": self._items}
        tmp = self._path.with_name(f"{self._path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, self._path)
        finally:
            tmp.unlink(missing_ok=True)

    # ---------- 查询 ----------

    def find(self, sha256: str) -> str | None:
        with self._lock:
            for filename, meta in self._items.items():
                if meta["sha256"] == sha256:
                    return filename
        return None

    def has(self, filename: str) -> bool:
        with self._lock:
            return filename in self._items

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    # ---------- 维护 ----------

    def add(self, filename: str, sha256: str, size: int = 0) -> None:
        with self._lock:
            self._items[filename] = {"sha256": sha256, "size": int(size)}
            self._save()

    def remove(self, filename: str) -> None:
        with self._lock:
            if self._items.pop(filename, None) is not None:
                self._save()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._save()

    def sync(self, existing: set[str]) -> int:
        """丢掉磁盘上已不存在的文件记录，返回清理条数。"""
        with self._lock:
            stale = [name for name in self._items if name not in existing]
            for name in stale:
                self._items.pop(name, None)
            if stale:
                self._save()
            return len(stale)
