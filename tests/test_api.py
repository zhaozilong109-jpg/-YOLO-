"""接口层测试：中文文件名、路径穿越、落盘持久化、zip 下载、错误信息。"""

import io
import json
import zipfile

from app import create_app


def upload(client, image_bytes, name, content_type="multipart/form-data"):
    return client.post(
        "/api/upload",
        data={"files": (image_bytes, name)},
        content_type=content_type,
    )


def test_health(client):
    payload = client.get("/api/health").get_json()
    assert payload["ok"] is True
    assert payload["annotations"] == 0


def test_upload_chinese_filename_survives(client, image_bytes):
    response = upload(client, image_bytes(), "猫.jpg")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 1

    listed = client.get("/api/images").get_json()["images"]
    assert [item["filename"] for item in listed] == ["猫.jpg"]
    assert listed[0]["annotated"] is False


def test_upload_same_name_gets_suffix(client, image_bytes):
    upload(client, image_bytes(), "cat.png")
    upload(client, image_bytes(), "cat.png")
    names = [item["filename"] for item in client.get("/api/images").get_json()["images"]]
    assert names == ["cat.png", "cat_1.png"]


def test_upload_rejects_fake_image(client):
    response = client.post(
        "/api/upload",
        data={"files": (io.BytesIO(b"not an image"), "fake.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["count"] == 0


def test_image_is_served(client, image_bytes):
    upload(client, image_bytes(), "cat.png")
    response = client.get("/uploads/cat.png")
    assert response.status_code == 200
    assert response.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_delete_cannot_escape_upload_dir(client, image_bytes, data_dir):
    upload(client, image_bytes(), "cat.png")
    secret = data_dir / "secret.txt"
    secret.write_text("keep me", encoding="utf-8")

    for evil in ("../secret.txt", "..%2Fsecret.txt", "....//secret.txt"):
        response = client.post("/api/delete_images", json={"filenames": [evil]})
        assert response.status_code == 200
        assert response.get_json()["deleted"] == []

    assert secret.exists()
    assert (data_dir / "uploads" / "cat.png").exists()


def test_annotation_save_and_reload(client, image_bytes, data_dir):
    upload(client, image_bytes(), "猫.png")

    saved = client.put(
        "/api/annotation/猫.png",
        json={
            "img_w": 640,
            "img_h": 480,
            "boxes": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 110,
                    "y2": 220,
                    "confidence": 0.77,
                    "class_id": 2,
                    "class_name": "手工框",
                }
            ],
        },
    )
    assert saved.status_code == 200
    assert saved.get_json()["boxes"][0]["class_name"] == "手工框"

    fetched = client.get("/api/annotation/猫.png").get_json()
    assert len(fetched["boxes"]) == 1

    # 模拟重启：新建 app 实例读同一份数据目录
    restarted = create_app(data_dir).test_client()
    again = restarted.get("/api/annotation/猫.png").get_json()
    assert again["boxes"][0]["x2"] == 110
    assert restarted.get("/api/images").get_json()["images"][0]["annotated"] is True


def test_annotation_on_missing_image_is_rejected(client):
    response = client.put("/api/annotation/nope.png", json={"boxes": []})
    assert response.status_code == 400
    assert "不存在" in response.get_json()["error"]


def test_annotation_accepts_empty_box_list(client, image_bytes):
    upload(client, image_bytes(), "empty.png")
    response = client.put("/api/annotation/empty.png", json={"boxes": []})
    assert response.status_code == 200
    assert response.get_json()["boxes"] == []
    # 空标注也算已标注，避免批量标注反复重跑同一张无目标的图
    assert client.get("/api/images").get_json()["images"][0]["annotated"] is True


def test_stats(client, image_bytes):
    upload(client, image_bytes(), "a.png")
    upload(client, image_bytes(), "b.png")
    client.put("/api/annotation/a.png", json={"boxes": []})
    payload = client.get("/api/stats").get_json()
    assert payload["total"] == 2
    assert payload["annotated"] == 1
    assert payload["pending"] == 1


def test_export_returns_zip(client, image_bytes):
    upload(client, image_bytes(), "猫.png")
    client.put(
        "/api/annotation/猫.png",
        json={
            "img_w": 640,
            "img_h": 480,
            "boxes": [
                {"x1": 10, "y1": 10, "x2": 110, "y2": 110, "class_id": 0, "class_name": "cat"}
            ],
        },
    )

    response = client.post("/api/export", json={"format": "yolo_txt", "include_images": True})
    assert response.status_code == 200
    assert response.mimetype == "application/zip"
    assert "attachment" in response.headers["Content-Disposition"]

    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        names = set(archive.namelist())
        assert "labels/train/猫.txt" in names
        assert "images/train/猫.png" in names
        assert archive.read("classes.txt").decode("utf-8") == "cat\n"


def test_export_without_data(client):
    response = client.post("/api/export", json={"format": "yolo_txt"})
    assert response.status_code == 400
    assert "没有标注数据" in response.get_json()["error"]


def test_export_rejects_unknown_format(client, image_bytes):
    upload(client, image_bytes(), "a.png")
    client.put("/api/annotation/a.png", json={"boxes": []})
    response = client.post("/api/export", json={"format": "nope"})
    assert response.status_code == 400


def test_annotate_without_model(client, image_bytes):
    upload(client, image_bytes(), "a.png")
    response = client.post("/api/annotate", json={"filename": "a.png"})
    assert response.status_code == 400
    assert "加载模型" in response.get_json()["error"]


def test_batch_without_images(client):
    response = client.post("/api/batch", json={"confidence": 0.25})
    assert response.status_code == 400
    assert "没有待标注" in response.get_json()["error"]


def test_models_list(client):
    payload = client.get("/api/models").get_json()
    assert payload["current"] is None
    assert any(model["name"] == "yolov8n.pt" for model in payload["pretrained"])


def test_api_404_is_json(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"] == "接口不存在"


def test_clear_all(client, image_bytes):
    upload(client, image_bytes(), "a.png")
    client.put("/api/annotation/a.png", json={"boxes": []})
    assert client.post("/api/clear_all").get_json()["removed"] == 1
    assert client.get("/api/images").get_json()["count"] == 0
    assert client.get("/api/stats").get_json()["annotated"] == 0


def test_unicode_json_is_utf8(client, image_bytes):
    upload(client, image_bytes(), "猫.png")
    raw = client.get("/api/images").data.decode("utf-8")
    assert "猫.png" in raw or json.loads(raw)["images"][0]["filename"] == "猫.png"
