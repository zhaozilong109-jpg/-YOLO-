"""运行期配置。

所有路径都可以用环境变量覆盖，方便测试和部署到别处：

    YOLO_LABELER_DATA           数据根目录（默认项目根目录）
    YOLO_LABELER_HOST           监听地址（默认 127.0.0.1）
    YOLO_LABELER_PORT           监听端口（默认 5000）
    YOLO_LABELER_MAX_UPLOAD_MB  单次请求体积上限，单位 MB（默认 50）
"""

from __future__ import annotations

import os
from pathlib import Path

# 项目根目录（app.py 所在目录）
BASE_DIR = Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    raw = (os.environ.get("YOLO_LABELER_DATA") or "").strip()
    if not raw:
        return BASE_DIR
    return Path(raw).expanduser().resolve()


DATA_DIR = _data_root()

UPLOAD_DIR = DATA_DIR / "uploads"
ANNOTATION_DIR = DATA_DIR / "annotations"
MODEL_DIR = DATA_DIR / "models"
EXPORT_DIR = DATA_DIR / "exports"

# 允许上传的图片格式
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# 导出时训练集/验证集的划分比例上限，防止传 1.0 把验证集塞满
MAX_VAL_RATIO = 0.5


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MAX_CONTENT_LENGTH = max(1, _int_env("YOLO_LABELER_MAX_UPLOAD_MB", 50)) * 1024 * 1024
HOST = os.environ.get("YOLO_LABELER_HOST", "127.0.0.1")
PORT = _int_env("YOLO_LABELER_PORT", 5000)


def ensure_dirs() -> None:
    """首次运行时创建全部数据目录。"""
    for folder in (UPLOAD_DIR, ANNOTATION_DIR, MODEL_DIR, EXPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
