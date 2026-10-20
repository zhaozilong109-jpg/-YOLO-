"""SAM 辅助接口：不加载真实模型，验证参数校验与形状约定。"""

import io
import zipfile

from core.sam import SamAssistant, SamError


def upload_png(client, name="a.png"):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), (30, 60, 90)).save(buffer, format="PNG")
    buffer.seek(0)
    return client.post(
        "/api/upload",
        data={"files": (buffer, name)},
        content_type="multipart/form-data",
    )


def test_sam_status_initially_not_loaded(client):
    payload = client.get("/api/sam/status").get_json()
    assert payload["loaded"] is False
    assert payload["model"] is None


def test_sam_segment_returns_polygons(client, monkeypatch):
    upload_png(client)

    assistant = client.application.extensions["sam"]
    captured = {}

    def fake_segment(image_path, points, labels=None, simplify=1.5):
        captured["points"] = points
        captured["labels"] = labels
        captured["path"] = str(image_path)
        return [
            [[10.0, 10.0], [100.0, 10.0], [100.0, 80.0]],
            [[20.0, 20.0], [30.0, 20.0], [30.0, 30.0]],
        ]

    monkeypatch.setattr(assistant, "segment", fake_segment)

    response = client.post(
        "/api/sam",
        json={"filename": "a.png", "points": [[50, 40]], "labels": [1]},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["count"] == 2
    assert payload["polygons"][0][0] == [10.0, 10.0]
    assert captured["points"] == [[50.0, 40.0]]
    assert captured["labels"] == [1]


def test_sam_requires_points(client):
    upload_png(client)
    response = client.post("/api/sam", json={"filename": "a.png"})
    assert response.status_code == 400
    assert "提示点" in response.get_json()["error"]


def test_sam_requires_existing_image(client):
    response = client.post("/api/sam", json={"filename": "nope.png", "points": [[1, 1]]})
    assert response.status_code == 400
    assert "不存在" in response.get_json()["error"]


def test_sam_error_is_reported_as_400(client, monkeypatch):
    upload_png(client)
    assistant = client.application.extensions["sam"]

    def boom(*args, **kwargs):
        raise SamError("模型下载失败")

    monkeypatch.setattr(assistant, "segment", boom)
    response = client.post("/api/sam", json={"filename": "a.png", "points": [[1, 1]]})
    assert response.status_code == 400
    assert "下载失败" in response.get_json()["error"]


def test_sam_labels_default_to_foreground(client, monkeypatch):
    upload_png(client)
    assistant = client.application.extensions["sam"]
    captured = {}

    def fake_segment(image_path, points, labels=None, simplify=1.5):
        captured["labels"] = labels
        return []

    monkeypatch.setattr(assistant, "segment", fake_segment)
    client.post("/api/sam", json={"filename": "a.png", "points": [[1, 1], [2, 2]]})
    assert captured["labels"] == [1, 1]


def test_sam_load_rejects_bad_name(client):
    assistant = client.application.extensions["sam"]
    try:
        assistant.load("model.onnx")
    except SamError as exc:
        assert ".pt" in str(exc)
    else:  # pragma: no cover - 不应该走到这里
        raise AssertionError("非法模型名应当被拒绝")


def test_polygon_annotation_roundtrip_and_seg_export(client):
    upload_png(client)

    points = [[10, 10], [200, 10], [200, 160], [10, 160]]
    saved = client.put(
        "/api/annotation/a.png",
        json={"img_w": 640, "img_h": 480, "boxes": [
            {"points": points, "class_id": 0, "class_name": "cat", "confidence": 1}
        ]},
    )
    assert saved.status_code == 200
    box = saved.get_json()["boxes"][0]
    assert box["shape"] == "polygon"
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (10, 10, 200, 160)

    response = client.post("/api/export", json={"format": "yolo_seg", "val_ratio": 0})
    assert response.status_code == 200

    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        label = archive.read("labels/train/a.txt").decode("utf-8").strip()
        tokens = label.split()
        assert tokens[0] == "0"
        assert len(tokens) == 9
        assert "task: segment" in archive.read("data.yaml").decode("utf-8")
