"""视频抽帧：把视频按固定时间间隔抽成图片，直接进入标注流程。

用 OpenCV 读视频、用 ``imencode`` 编码成 JPEG 字节，全程不经过图片文件路径，
因此在中文目录下也不会出现 ``cv2.imwrite`` 那种写不进去的问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

VIDEO_EXTENSIONS = {
    "mp4", "avi", "mov", "mkv", "webm", "flv", "m4v", "mpg", "mpeg", "wmv",
}

# 抽帧参数的硬性边界（前端的选项是软限制）
MIN_INTERVAL = 0.1
MAX_INTERVAL = 60.0
MAX_FRAMES = 300


class VideoError(RuntimeError):
    """视频相关错误（会以 400 返回给前端）。"""


def _cv2():
    try:
        import cv2  # noqa: PLC0415 - 延迟导入，没装 opencv 时不影响其它功能
    except ImportError as exc:  # pragma: no cover - 依赖缺失时才触发
        raise VideoError("抽帧需要 opencv，请先执行 pip install opencv-python") from exc
    return cv2


def probe(video: Path) -> dict:
    """读取视频元信息。"""
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise VideoError("无法打开视频，可能是编码不受支持")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    if fps <= 0:
        fps = 25.0

    return {
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": round(total_frames / fps, 2) if total_frames else 0.0,
    }


def iter_frames(
    video: Path,
    interval: float = 1.0,
    max_frames: int = 60,
    start: float = 0.0,
    quality: int = 90,
) -> Iterator[tuple[int, bytes]]:
    """按 ``interval`` 秒抽帧，逐帧 yield ``(帧序号, JPEG 字节)``。

    跳帧用 ``grab()``（只移动游标不解码），比逐帧 ``read()`` 快很多。
    """
    cv2 = _cv2()
    interval = min(max(float(interval), MIN_INTERVAL), MAX_INTERVAL)
    max_frames = max(1, min(int(max_frames), MAX_FRAMES))

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise VideoError("无法打开视频，可能是编码不受支持")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        step = max(1, int(round(fps * interval)))
        start_frame = max(0, int(round(float(start) * fps)))
        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        index = start_frame
        emitted = 0
        while emitted < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            )
            if ok:
                yield index, buffer.tobytes()
                emitted += 1

            index += step
            # 跳过中间的帧
            for _ in range(step - 1):
                if not capture.grab():
                    return
    finally:
        capture.release()
