"""文件名安全处理 + 标注结果持久化。

标注结果按「一张图片一个 JSON」保存在 annotations/ 下，文件名是
``<图片文件名>.json``，进程重启后会自动回填，不会再出现「图片还在、标注全丢」。
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Iterable


class StorageError(RuntimeError):
    """存储层可预期的错误（会以 400 返回给前端）。"""


class PathTraversalError(StorageError):
    """文件名试图跳出目标目录。"""


# Windows 非法字符 + 控制字符
_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
}
_MAX_STEM_LEN = 80


def _clean_stem(stem: str, fallback: str) -> str:
    """清洗不含扩展名的文件名主干（允许中文）。"""
    stem = _ILLEGAL_CHARS.sub("_", stem).strip().strip(".")
    if not stem:
        stem = fallback
    if stem.upper() in _RESERVED_NAMES:
        stem = f"_{stem}"
    # 末尾的点和空格在 Windows 上会被静默截断，先处理掉
    stem = stem.rstrip(". ")[:_MAX_STEM_LEN].rstrip(". ")
    return stem or fallback


def sanitize_stem(raw: str, fallback: str = "video") -> str:
    """从任意文件名里取出安全的主干，用于视频抽帧命名。"""
    name = str(raw or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFC", name).strip()
    stem = name.rpartition(".")[0] if "." in name else name
    return _clean_stem(stem, fallback)


def sanitize_filename(raw: str, allowed_extensions: Iterable[str]) -> str:
    """把浏览器给的文件名清洗成安全、可跨平台保存的名字，同时保留中文。

    与 werkzeug 的 ``secure_filename`` 不同，这里不会丢弃非 ASCII 字符——
    ``猫.jpg`` 存成 ``jpg`` 那种事故（扩展名丢失、图片在列表里消失）不会再发生。
    """
    name = str(raw or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFC", name).strip()

    # 拆出扩展名（兼容 a.b.jpg 这种）
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""

    ext = ext.lower().strip()
    allowed = {e.lower().lstrip(".") for e in allowed_extensions}
    if ext not in allowed:
        raise StorageError(f"不支持的文件格式: {ext or '未知'}")

    return f"{_clean_stem(stem, 'image')}.{ext}"


def resolve_within(base: Path, name: str) -> Path:
    """把用户提供的名字解析到 base 目录内，越界一律拒绝。"""
    if not name or name in {".", ".."}:
        raise PathTraversalError("文件名无效")
    if "/" in name or "\\" in name:
        raise PathTraversalError(f"文件名不能包含路径分隔符: {name}")

    base = Path(base).resolve()
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise PathTraversalError(f"路径越界: {name}") from exc
    return target


def unique_path(folder: Path, filename: str) -> Path:
    """同名文件自动加 ``_1``、``_2`` 后缀，避免互相覆盖。"""
    folder = Path(folder)
    candidate = folder / filename
    if not candidate.exists():
        return candidate

    stem, suffix = os.path.splitext(filename)
    for index in range(1, 10000):
        candidate = folder / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise StorageError(f"同名文件过多: {filename}")


MIN_POLYGON_POINTS = 3
MAX_POLYGON_POINTS = 400


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalise_points(
    raw_points: Any, img_w: float, img_h: float
) -> list[list[float]] | None:
    """校验多边形顶点：裁剪到图像范围、去掉相邻重复点；少于 3 点返回 None。"""
    if not isinstance(raw_points, (list, tuple)):
        return None

    points: list[list[float]] = []
    for pair in raw_points:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            x = float(pair[0])
            y = float(pair[1])
        except (TypeError, ValueError):
            continue

        if img_w > 0:
            x = _clamp(x, 0.0, img_w)
        if img_h > 0:
            y = _clamp(y, 0.0, img_h)
        if points and abs(points[-1][0] - x) < 0.5 and abs(points[-1][1] - y) < 0.5:
            continue
        points.append([round(x, 2), round(y, 2)])

    # 首尾重合时去掉最后一个点（前端闭合时常见）
    if (
        len(points) > 1
        and abs(points[0][0] - points[-1][0]) < 0.5
        and abs(points[0][1] - points[-1][1]) < 0.5
    ):
        points.pop()

    if len(points) < MIN_POLYGON_POINTS:
        return None

    # 点太多会让 JSON、画布和导出都变慢，等间隔抽样到上限
    if len(points) > MAX_POLYGON_POINTS:
        step = len(points) / MAX_POLYGON_POINTS
        points = [points[int(index * step)] for index in range(MAX_POLYGON_POINTS)]
    return points


def _normalise_box(raw: Any, img_w: float, img_h: float) -> dict | None:
    """校验并裁剪一个标注形状（矩形框或多边形）；非法则返回 None。

    多边形同样会算出外接矩形，这样检测格式导出、按框排序等旧逻辑不用改。
    """
    if not isinstance(raw, dict):
        return None

    points = normalise_points(raw.get("points"), img_w, img_h)
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
    else:
        try:
            x1 = float(raw["x1"])
            y1 = float(raw["y1"])
            x2 = float(raw["x2"])
            y2 = float(raw["y2"])
        except (KeyError, TypeError, ValueError):
            return None

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if img_w > 0:
            x1 = _clamp(x1, 0.0, img_w)
            x2 = _clamp(x2, 0.0, img_w)
        if img_h > 0:
            y1 = _clamp(y1, 0.0, img_h)
            y2 = _clamp(y2, 0.0, img_h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None

    try:
        confidence = _clamp(float(raw.get("confidence", 1.0)), 0.0, 1.0)
    except (TypeError, ValueError):
        confidence = 1.0
    try:
        class_id = int(raw.get("class_id", 0))
    except (TypeError, ValueError):
        class_id = 0

    shape = {
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "confidence": round(confidence, 4),
        "class_id": max(class_id, 0),
        "class_name": str(raw.get("class_name") or f"class_{class_id}"),
        "shape": "polygon" if points else "box",
    }
    if points:
        shape["points"] = points
    return shape


class AnnotationStore:
    """线程安全的标注仓库，负责落盘与启动回填。"""

    VERSION = 2

    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._items: dict[str, dict] = {}
        self.load()

    # ---------- 磁盘 ----------

    def _json_path(self, filename: str) -> Path:
        return resolve_within(self._dir, f"{filename}.json")

    def load(self) -> int:
        """扫描 annotations/ 回填内存，返回加载条数。"""
        loaded = 0
        with self._lock:
            self._items.clear()
            for path in sorted(self._dir.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    # 损坏的文件跳过，不影响其它标注
                    continue
                filename = str(data.get("filename") or path.name[: -len(".json")])
                if not filename:
                    continue
                self._items[filename] = self._clean(filename, data)
                loaded += 1
        return loaded

    def _clean(self, filename: str, data: dict) -> dict:
        try:
            img_w = float(data.get("img_w") or 0)
            img_h = float(data.get("img_h") or 0)
        except (TypeError, ValueError):
            img_w = img_h = 0.0

        boxes = []
        for raw in data.get("boxes") or []:
            box = _normalise_box(raw, img_w, img_h)
            if box is not None:
                boxes.append(box)

        return {
            "filename": filename,
            "img_w": int(img_w),
            "img_h": int(img_h),
            "boxes": boxes,
            "updated_at": float(data.get("updated_at") or 0),
            "version": self.VERSION,
        }

    def _write(self, data: dict) -> None:
        path = self._json_path(data["filename"])
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ---------- 读写 ----------

    def get(self, filename: str) -> dict | None:
        with self._lock:
            item = self._items.get(filename)
            return copy.deepcopy(item) if item else None

    def put(
        self,
        filename: str,
        boxes: list[dict],
        img_w: int,
        img_h: int,
    ) -> dict:
        """写入（或覆盖）一张图片的标注，立即落盘。"""
        if not filename:
            raise StorageError("缺少图片文件名")

        data = {
            "filename": filename,
            "img_w": int(img_w or 0),
            "img_h": int(img_h or 0),
            "boxes": [
                box
                for box in (
                    _normalise_box(raw, float(img_w or 0), float(img_h or 0))
                    for raw in (boxes or [])
                )
                if box is not None
            ],
            "updated_at": time.time(),
            "version": self.VERSION,
        }
        with self._lock:
            self._write(data)
            self._items[filename] = data
            return copy.deepcopy(data)

    def delete(self, filename: str) -> bool:
        with self._lock:
            existed = self._items.pop(filename, None) is not None
            try:
                self._json_path(filename).unlink(missing_ok=True)
            except (OSError, PathTraversalError):
                pass
            return existed

    def clear(self) -> None:
        with self._lock:
            for path in self._dir.glob("*.json"):
                try:
                    path.unlink()
                except OSError:
                    pass
            self._items.clear()

    def all_items(self) -> dict[str, dict]:
        with self._lock:
            return copy.deepcopy(self._items)

    def has(self, filename: str) -> bool:
        with self._lock:
            return filename in self._items

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def class_names(self) -> dict[int, str]:
        """已出现过的类别，用于离线补标时也能选类别。"""
        found: dict[int, str] = {}
        with self._lock:
            for item in self._items.values():
                for box in item["boxes"]:
                    found.setdefault(int(box["class_id"]), str(box["class_name"]))
        return found
