"""视频抽帧的单元测试与接口测试。"""

import io

import pytest

from core.video import VideoError, iter_frames, probe

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def make_video(path, frames=30, fps=10, size=(64, 48), static=False):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    if not writer.isOpened():
        pytest.skip("当前 OpenCV 构建不支持写 MJPG 视频")
    try:
        for index in range(frames):
            if static:
                color = (40, 80, 120)
            else:
                color = ((index * 8) % 255, (index * 5) % 255, (index * 3) % 255)
            writer.write(np.full((size[1], size[0], 3), color, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_probe_reads_metadata(tmp_path):
    video = make_video(tmp_path / "demo.avi", frames=30, fps=10)
    meta = probe(video)
    assert abs(meta["fps"] - 10) < 0.5
    assert meta["total_frames"] >= 25  # 不同编码器略有差异
    assert (meta["width"], meta["height"]) == (64, 48)
    assert meta["duration"] > 1.5


def test_iter_frames_respects_interval(tmp_path):
    video = make_video(tmp_path / "demo.avi", frames=30, fps=10)
    frames = list(iter_frames(video, interval=0.5, max_frames=100))
    # 0.5 秒 @ 10fps → 每 5 帧取一张，30 帧大约 6 张
    assert 5 <= len(frames) <= 7
    assert all(data[:2] == b"\xff\xd8" for _, data in frames)


def test_iter_frames_respects_max_frames(tmp_path):
    video = make_video(tmp_path / "demo.avi", frames=30, fps=10)
    frames = list(iter_frames(video, interval=0.1, max_frames=4))
    assert len(frames) == 4


def test_broken_video_raises(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not really a video")
    with pytest.raises(VideoError):
        probe(broken)


def test_upload_video_endpoint(client, tmp_path):
    video = make_video(tmp_path / "demo.avi", frames=30, fps=10)
    response = client.post(
        "/api/upload_video",
        data={
            "video": (io.BytesIO(video.read_bytes()), "演示视频.avi"),
            "interval": "0.1",
            "max_frames": "5",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 5
    assert payload["frames"][0].startswith("演示视频_")
    assert payload["video"]["fps"] > 0

    listed = client.get("/api/images").get_json()["images"]
    assert len(listed) == 5
    assert all(item["filename"].endswith(".jpg") for item in listed)


def test_static_video_dedupes_frames(client, tmp_path):
    video = make_video(tmp_path / "static.avi", frames=10, fps=10, static=True)
    response = client.post(
        "/api/upload_video",
        data={
            "video": (io.BytesIO(video.read_bytes()), "静止画面.avi"),
            "interval": "0.1",
            "max_frames": "5",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1
    assert payload["duplicates"] == 4


def test_invalid_video_rejected(client):
    response = client.post(
        "/api/upload_video",
        data={"video": (io.BytesIO(b"garbage"), "bad.mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "视频" in response.get_json()["error"]


def test_unsupported_video_extension_rejected(client):
    response = client.post(
        "/api/upload_video",
        data={"video": (io.BytesIO(b"data"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "不支持" in response.get_json()["error"]
