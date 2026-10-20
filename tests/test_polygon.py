"""多边形标注：存储规范化与分割格式导出。"""

import io
import zipfile

import pytest

from core.exporter import build_export, coco_document, voc_xml, yolo_seg_text, yolo_label_text
from core.storage import AnnotationStore


def make_ann(boxes, width=640, height=480, filename="cat.jpg"):
    annotations = []
    for box in boxes:
        item = {key: box[key] for key in ("x1", "y1", "x2", "y2")}
        item["class_id"] = box.get("class_id", 0)
        item["class_name"] = box.get("class_name", "cat")
        item["confidence"] = box.get("confidence", 1.0)
        if "points" in box:
            item["points"] = box["points"]
        annotations.append(item)
    return {
        "filename": filename,
        "img_w": width,
        "img_h": height,
        "boxes": annotations,
    }


SQUARE = [[10, 10], [110, 10], [110, 110], [10, 110]]


# ---------------------------------------------------------------- 存储


def test_polygon_is_kept_and_bbox_derived(tmp_path):
    store = AnnotationStore(tmp_path)
    saved = store.put(
        "p.jpg",
        [{"points": SQUARE, "class_id": 0, "class_name": "poly"}],
        640,
        480,
    )
    box = saved["boxes"][0]
    assert box["shape"] == "polygon"
    assert len(box["points"]) == 4
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (10, 10, 110, 110)


def test_polygon_points_are_clamped_and_deduped(tmp_path):
    store = AnnotationStore(tmp_path)
    saved = store.put(
        "p.jpg",
        [
            {
                "points": [[-20, 10], [200, 10], [200, 10], [190, 900]],
                "class_id": 0,
                "class_name": "poly",
            }
        ],
        100,
        100,
    )
    points = saved["boxes"][0]["points"]
    assert points[0] == [0, 10]
    assert points[1] == [100, 10]
    assert len(points) == 3  # 重复点被合并
    assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in points)


def test_polygon_with_too_few_points_is_dropped(tmp_path):
    store = AnnotationStore(tmp_path)
    saved = store.put(
        "p.jpg",
        [{"points": [[10, 10], [20, 20]], "class_id": 0, "class_name": "poly"}],
        640,
        480,
    )
    assert saved["boxes"] == []


def test_polygon_persists_across_restart(tmp_path):
    store = AnnotationStore(tmp_path)
    store.put("p.jpg", [{"points": SQUARE, "class_id": 0, "class_name": "poly"}], 640, 480)

    reloaded = AnnotationStore(tmp_path)
    box = reloaded.get("p.jpg")["boxes"][0]
    assert box["shape"] == "polygon"
    assert box["points"][2] == [110, 110]


# ---------------------------------------------------------------- 导出


def test_yolo_seg_text_is_normalised():
    annotation = make_ann([{"x1": 10, "y1": 10, "x2": 110, "y2": 110, "points": SQUARE}])
    text = yolo_seg_text(annotation, {0: 0})

    tokens = text.split()
    assert tokens[0] == "0"
    assert len(tokens) == 9  # class + 4 个点 × 2 个坐标

    values = [float(value) for value in tokens[1:]]
    assert values[0] == pytest.approx(10 / 640, abs=1e-5)
    assert values[1] == pytest.approx(10 / 480, abs=1e-5)
    assert values[2] == pytest.approx(110 / 640, abs=1e-5)
    assert values[3] == pytest.approx(10 / 480, abs=1e-5)


def test_yolo_seg_falls_back_to_rect_corners():
    annotation = make_ann([{"x1": 0, "y1": 0, "x2": 640, "y2": 480}])
    tokens = yolo_seg_text(annotation, {0: 0}).split()
    assert tokens[0] == "0"
    assert len(tokens) == 9
    assert all(float(value) in (0.0, 1.0) for value in tokens[1:])


def test_yolo_label_still_uses_bbox_for_polygons():
    annotation = make_ann([{"x1": 10, "y1": 10, "x2": 110, "y2": 110, "points": SQUARE}])
    text = yolo_label_text(annotation, {0: 0})
    values = [float(value) for value in text.split()[1:]]
    # 中心点 (60,60)，宽高 (100,100)
    assert values[0] == pytest.approx(60 / 640, abs=1e-5)
    assert values[1] == pytest.approx(60 / 480, abs=1e-5)
    assert values[2] == pytest.approx(100 / 640, abs=1e-5)
    assert values[3] == pytest.approx(100 / 480, abs=1e-5)


def test_coco_contains_segmentation_and_polygon_area():
    annotation = make_ann([{"x1": 10, "y1": 10, "x2": 110, "y2": 110, "points": SQUARE}])
    document = coco_document({"a.jpg": annotation}, {0: 0}, ["cat"])
    item = document["annotations"][0]

    assert item["segmentation"] == [[10, 10, 110, 10, 110, 110, 10, 110]]
    assert item["area"] == pytest.approx(10000.0)
    assert item["bbox"] == [10.0, 10.0, 100.0, 100.0]


def test_coco_rect_still_has_four_corner_segmentation():
    annotation = make_ann([{"x1": 0, "y1": 0, "x2": 10, "y2": 20}])
    document = coco_document({"a.jpg": annotation}, {0: 0}, ["cat"])
    segmentation = document["annotations"][0]["segmentation"]
    assert len(segmentation) == 1
    assert len(segmentation[0]) == 8


def test_voc_uses_bbox_of_polygon():
    annotation = make_ann([{"x1": 10, "y1": 10, "x2": 110, "y2": 110, "points": SQUARE}])
    from xml.etree import ElementTree

    root = ElementTree.fromstring(voc_xml("a.jpg", annotation))
    box = root.find("object/bndbox")
    assert box.findtext("xmin") == "10"
    assert box.findtext("xmax") == "110"


def test_export_yolo_seg_zip(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "cat.jpg").write_bytes(b"fake")

    annotations = {
        "cat.jpg": make_ann([{"x1": 10, "y1": 10, "x2": 110, "y2": 110, "points": SQUARE}])
    }
    data, names = build_export(
        "yolo_seg",
        annotations,
        {0: "cat"},
        upload_dir,
        include_images=True,
        val_ratio=0.0,
    )

    assert names.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = set(archive.namelist())
        assert "labels/train/cat.txt" in entries
        assert "images/train/cat.jpg" in entries
        assert "task: segment" in archive.read("data.yaml").decode("utf-8")

        line = archive.read("labels/train/cat.txt").decode("utf-8").strip()
        assert line.split()[0] == "0"
        assert len(line.split()) == 9
