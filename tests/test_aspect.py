"""图片比例：EXIF 方向导致的尺寸不一致，应该被自动矫正。"""

import io

from PIL import Image


def exif_rotated_jpeg(width=640, height=480):
    """生成像素 640x480、EXIF 标记顺时针旋转 90° 的 JPEG。

    浏览器按 EXIF 显示成 480x640（竖图）；PIL 直接读 size 仍是 640x480。
    """
    image = Image.new("RGB", (width, height), (90, 120, 150))
    exif = image.getexif()
    exif[274] = 6  # Orientation: Rotate 90 CW
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)
    return buffer


def upload_bytes(client, buffer, name):
    return client.post(
        "/api/upload",
        data={"files": (buffer, name)},
        content_type="multipart/form-data",
    )


def test_annotation_reports_exif_oriented_size(client):
    upload_bytes(client, exif_rotated_jpeg(), "rot.jpg")
    payload = client.get("/api/annotation/rot.jpg").get_json()
    assert (payload["img_w"], payload["img_h"]) == (480, 640)


def test_save_with_stale_dimensions_is_corrected(client):
    """前端若还传着未旋转的 640x480，服务端要把坐标一起转到 480x640。"""
    upload_bytes(client, exif_rotated_jpeg(), "rot.jpg")

    response = client.put(
        "/api/annotation/rot.jpg",
        json={
            "img_w": 640,
            "img_h": 480,
            "boxes": [
                {
                    "x1": 0,
                    "y1": 0,
                    "x2": 640,
                    "y2": 480,
                    "class_id": 0,
                    "class_name": "full",
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert (payload["img_w"], payload["img_h"]) == (480, 640)

    box = payload["boxes"][0]
    assert box["x1"] == 0
    assert box["y1"] == 0
    assert box["x2"] == 480
    assert box["y2"] == 640


def test_polygon_is_rescaled_too(client):
    upload_bytes(client, exif_rotated_jpeg(), "rot.jpg")

    response = client.put(
        "/api/annotation/rot.jpg",
        json={
            "img_w": 640,
            "img_h": 480,
            "boxes": [
                {
                    "points": [[0, 0], [640, 0], [640, 480]],
                    "class_id": 0,
                    "class_name": "tri",
                }
            ],
        },
    )
    box = response.get_json()["boxes"][0]
    assert box["shape"] == "polygon"
    assert box["points"][1] == [480.0, 0.0]
    assert box["points"][2] == [480.0, 640.0]


def test_correct_dimensions_are_left_alone(client):
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), (10, 20, 30)).save(buffer, format="PNG")
    buffer.seek(0)
    upload_bytes(client, buffer, "normal.png")

    response = client.put(
        "/api/annotation/normal.png",
        json={
            "img_w": 640,
            "img_h": 480,
            "boxes": [
                {"x1": 10, "y1": 10, "x2": 110, "y2": 110, "class_id": 0, "class_name": "cat"}
            ],
        },
    )
    payload = response.get_json()
    assert (payload["img_w"], payload["img_h"]) == (640, 480)
    assert payload["boxes"][0]["x2"] == 110


def test_missing_dimensions_fall_back_to_real_size(client):
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), (10, 20, 30)).save(buffer, format="PNG")
    buffer.seek(0)
    upload_bytes(client, buffer, "nodim.png")

    response = client.put(
        "/api/annotation/nodim.png",
        json={"boxes": [{"x1": 5, "y1": 5, "x2": 100, "y2": 100, "class_id": 0, "class_name": "x"}]},
    )
    payload = response.get_json()
    assert (payload["img_w"], payload["img_h"]) == (320, 240)
