"""模型加载、单图推理与批量标注任务。

要点：
* 推理全程持锁——YOLO/PyTorch 模型不是线程安全的，多标签页同时点「标注」不会再打崩进程。
* 批量标注改成服务端单任务：一次扫描、顺序推进，前端轮询进度，
  不再每张图都全目录扫描（原来 1000 张图要做 1000 次 O(n) 扫描）。
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

from .storage import AnnotationStore


class InferenceError(RuntimeError):
    """推理相关错误（会以 400/500 返回给前端）。"""


class ModelNotLoaded(InferenceError):
    """还没加载模型。"""


class BusyError(InferenceError):
    """已有批量任务在跑。"""


PRETRAINED_MODELS = [
    {"name": "yolov8n.pt", "label": "YOLOv8n（最轻，CPU 友好）"},
    {"name": "yolov8s.pt", "label": "YOLOv8s（小型）"},
    {"name": "yolov8m.pt", "label": "YOLOv8m（中等）"},
    {"name": "yolov8l.pt", "label": "YOLOv8l（较大）"},
    {"name": "yolov8x.pt", "label": "YOLOv8x（最准，最慢）"},
    {"name": "yolov5su.pt", "label": "YOLOv5su（轻量）"},
]


class ModelManager:
    """单个模型的加载与推理，全局一把锁。"""

    def __init__(self, model_dir: Path):
        self._dir = Path(model_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._model = None
        self._name: str | None = None
        self._classes: dict[int, str] = {}

    # ---------- 状态 ----------

    @property
    def model_name(self) -> str | None:
        with self._lock:
            return self._name

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def classes(self) -> dict[int, str]:
        with self._lock:
            return dict(self._classes)

    def available(self) -> dict:
        local = sorted(p.name for p in self._dir.glob("*.pt"))
        return {
            "pretrained": PRETRAINED_MODELS,
            "custom": [{"name": name, "label": f"自定义: {name}"} for name in local],
            "current": self.model_name,
        }

    # ---------- 加载 ----------

    def load(self, name: str) -> dict[int, str]:
        safe = os.path.basename(str(name or "").strip())
        if not safe:
            raise InferenceError("未指定模型")
        if not safe.lower().endswith(".pt"):
            raise InferenceError("模型文件名必须以 .pt 结尾")

        with self._lock:
            path = self._dir / safe
            if not path.exists():
                self._download(safe)
            try:
                from ultralytics import YOLO

                model = YOLO(str(path))
            except Exception as exc:  # noqa: BLE001 - 交给上层转成提示
                raise InferenceError(f"模型加载失败: {exc}") from exc

            names = dict(getattr(model, "names", {}) or {})
            self._model = model
            self._name = safe
            self._classes = {int(k): str(v) for k, v in names.items()}
            return dict(self._classes)

    def _download(self, name: str) -> None:
        """让 ultralytics 把预训练权重下到 models/ 目录。

        ultralytics 只会下载到当前工作目录，这里临时切换并在 finally 里还原；
        因为整个加载过程都持有 self._lock，不存在其它线程被连带影响的问题。
        """
        workdir = os.getcwd()
        try:
            from ultralytics import YOLO

            os.chdir(self._dir)
            YOLO(name)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(
                f"无法获取模型权重 {name}：{exc}。可手动下载后放入 models/ 目录。"
            ) from exc
        finally:
            os.chdir(workdir)

        if not (self._dir / name).exists():
            raise InferenceError(f"模型 {name} 下载后仍未找到，请手动放入 models/ 目录")

    # ---------- 推理 ----------

    def annotate(
        self,
        image_path: Path,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: list[int] | None = None,
    ) -> tuple[list[dict], int, int]:
        """返回 (boxes, img_w, img_h)，坐标是原图像素值。"""
        with self._lock:
            if self._model is None:
                raise ModelNotLoaded("请先加载模型")

            kwargs: dict = {
                "conf": min(max(float(conf), 0.01), 0.99),
                "iou": min(max(float(iou), 0.01), 0.99),
                "verbose": False,
            }
            if classes:
                kwargs["classes"] = [int(c) for c in classes]

            try:
                results = self._model(str(image_path), **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(f"推理失败: {exc}") from exc

            if not results:
                return [], 0, 0

            result = results[0]
            shape = getattr(result, "orig_shape", (0, 0)) or (0, 0)
            img_h, img_w = int(shape[0]), int(shape[1])

            boxes: list[dict] = []
            raw_boxes = getattr(result, "boxes", None)
            if raw_boxes is not None:
                for box in raw_boxes:
                    coords = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    boxes.append(
                        {
                            "x1": round(float(coords[0]), 2),
                            "y1": round(float(coords[1]), 2),
                            "x2": round(float(coords[2]), 2),
                            "y2": round(float(coords[3]), 2),
                            "confidence": round(float(box.conf[0]), 4),
                            "class_id": class_id,
                            "class_name": self._classes.get(
                                class_id, f"class_{class_id}"
                            ),
                        }
                    )

            return boxes, img_w, img_h


class BatchRunner:
    """批量标注：同一时刻只允许一个任务，进度可轮询、可取消。"""

    def __init__(
        self,
        models: ModelManager,
        store: AnnotationStore,
        upload_dir: Path,
    ):
        self._models = models
        self._store = store
        self._uploads = Path(upload_dir)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._active: str | None = None
        self._keep = 20

    # ---------- 查询 ----------

    def snapshot(self, job_id: str) -> dict | None:
        with self._lock:
            return self._snapshot_locked(job_id)

    def _snapshot_locked(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {
            key: job[key]
            for key in (
                "id",
                "status",
                "total",
                "done",
                "failed",
                "current",
                "errors",
                "started_at",
            )
        }

    def active_job(self) -> dict | None:
        with self._lock:
            if not self._active:
                return None
            return self._snapshot_locked(self._active)

    # ---------- 控制 ----------

    def start(
        self,
        files: list[str],
        conf: float,
        iou: float,
        classes: list[int] | None,
    ) -> dict:
        if not files:
            raise InferenceError("没有待标注的图片")

        with self._lock:
            if self._active:
                job = self._jobs.get(self._active)
                if job and job["status"] == "running":
                    raise BusyError("已有批量任务在运行，请等待完成或取消")

            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {
                "id": job_id,
                "status": "running",
                "total": len(files),
                "done": 0,
                "failed": 0,
                "current": None,
                "errors": [],
                "started_at": time.time(),
                "cancel": False,
            }
            self._active = job_id
            self._prune_locked()

        thread = threading.Thread(
            target=self._run,
            args=(job_id, list(files), conf, iou, classes),
            name=f"batch-{job_id}",
            daemon=True,
        )
        thread.start()
        return self.snapshot(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job["cancel"] = True
            return True

    # ---------- 执行 ----------

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self._keep:
            return
        for job_id, job in list(self._jobs.items()):
            if job["status"] != "running":
                self._jobs.pop(job_id, None)
            if len(self._jobs) <= self._keep:
                break

    def _run(
        self,
        job_id: str,
        files: list[str],
        conf: float,
        iou: float,
        classes: list[int] | None,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]

        for filename in files:
            with self._lock:
                if job["cancel"]:
                    job["status"] = "cancelled"
                    break
                job["current"] = filename

            try:
                boxes, img_w, img_h = self._models.annotate(
                    self._uploads / filename, conf=conf, iou=iou, classes=classes
                )
                self._store.put(filename, boxes, img_w, img_h)
                with self._lock:
                    job["done"] += 1
            except Exception as exc:  # noqa: BLE001 - 单张失败不中断整体
                with self._lock:
                    job["failed"] += 1
                    if len(job["errors"]) < 20:
                        job["errors"].append(
                            {"filename": filename, "error": str(exc)}
                        )
            finally:
                with self._lock:
                    job["current"] = None

        with self._lock:
            if job["status"] == "running":
                job["status"] = "done"
            if self._active == job_id:
                self._active = None
