"""SAM 辅助标注：在图上点一下目标，返回精确轮廓多边形。

用 Ultralytics 的 SAM 封装（默认 ``mobile_sam.pt``，约 40 MB，CPU 上单点约 1–2 秒）。
``masks.xy`` 已经是原图像素坐标的轮廓，这里再做一次多边形简化，
避免把上百个点塞进 JSON 和画布。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

DEFAULT_MODEL = "mobile_sam.pt"
MIN_AREA = 12.0        # 噪点轮廓（面积太小）直接丢弃
MAX_RESULTS = 3        # 一次最多返回几个候选轮廓
MAX_POINTS = 200       # 单个轮廓最多保留的顶点数


class SamError(RuntimeError):
    """SAM 相关错误（会以 400 返回给前端）。"""


class SamAssistant:
    """懒加载的 SAM 单例，推理全程持锁。"""

    def __init__(self, model_dir: Path, model_name: str = DEFAULT_MODEL):
        self._dir = Path(model_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._name = model_name
        self._lock = threading.RLock()
        self._model = None
        self._loaded_name: str | None = None

    # ---------- 状态 ----------

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def status(self) -> dict:
        with self._lock:
            return {"loaded": self._model is not None, "model": self._loaded_name}

    # ---------- 加载 ----------

    def load(self, name: str | None = None) -> str:
        safe = os.path.basename(str(name or self._name).strip())
        if not safe:
            raise SamError("未指定 SAM 模型")
        if not safe.lower().endswith(".pt"):
            raise SamError("SAM 模型文件名必须以 .pt 结尾")

        with self._lock:
            if self._model is not None and self._loaded_name == safe:
                return safe

            path = self._dir / safe
            if not path.exists():
                self._download(safe)

            try:
                from ultralytics import SAM

                model = SAM(str(path))
            except Exception as exc:  # noqa: BLE001
                raise SamError(f"SAM 模型加载失败: {exc}") from exc

            self._model = model
            self._loaded_name = safe
            self._name = safe
            return safe

    def _download(self, name: str) -> None:
        """让 ultralytics 把权重下到 models/ 目录（加载期间持锁，chdir 安全）。"""
        workdir = os.getcwd()
        try:
            from ultralytics import SAM

            os.chdir(self._dir)
            SAM(name)
        except Exception as exc:  # noqa: BLE001
            raise SamError(
                f"无法获取 SAM 权重 {name}：{exc}。可手动下载后放入 models/ 目录。"
            ) from exc
        finally:
            os.chdir(workdir)

        if not (self._dir / name).exists():
            raise SamError(f"SAM 权重 {name} 下载后仍未找到，请手动放入 models/ 目录")

    # ---------- 推理 ----------

    def segment(
        self,
        image_path: Path,
        points: list[list[float]],
        labels: list[int] | None = None,
        simplify: float = 1.5,
    ) -> list[list[list[float]]]:
        """返回若干轮廓，每个轮廓是 ``[[x, y], ...]``（原图像素坐标）。"""
        if not points:
            raise SamError("缺少提示点")
        if self._model is None:
            self.load()

        with self._lock:
            try:
                results = self._model(
                    str(image_path),
                    points=points,
                    labels=labels,
                    verbose=False,
                )
            except Exception as exc:  # noqa: BLE001
                raise SamError(f"SAM 推理失败: {exc}") from exc

        if not results:
            return []

        masks = getattr(results[0], "masks", None)
        raw_polygons = getattr(masks, "xy", None) if masks is not None else None
        if not raw_polygons:
            return []

        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - 依赖缺失时才触发
            raise SamError("轮廓简化需要 opencv 与 numpy") from exc

        candidates: list[tuple[float, list[list[float]]]] = []
        for polygon in raw_polygons:
            array = np.asarray(polygon, dtype=np.float32)
            if array.ndim != 2 or array.shape[0] < 3:
                continue

            contour = array.reshape(-1, 1, 2)
            perimeter = cv2.arcLength(contour, True)
            epsilon = max(float(simplify), 0.002 * perimeter)
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(approx) < 3:
                continue

            area = float(cv2.contourArea(approx.reshape(-1, 1, 2)))
            if area < MIN_AREA:
                continue

            coords = [[round(float(x), 1), round(float(y), 1)] for x, y in approx]
            if len(coords) > MAX_POINTS:
                step = len(coords) / MAX_POINTS
                coords = [coords[int(index * step)] for index in range(MAX_POINTS)]

            candidates.append((area, coords))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [coords for _, coords in candidates[:MAX_RESULTS]]
