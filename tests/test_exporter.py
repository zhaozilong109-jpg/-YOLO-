"""导出格式的数值正确性与 zip 结构测试。"""

import io
import json
import zipfile
from xml.etree import ElementTree

import pytest

from core.exporter import (
    ExportError,
    _class_table,
    build_export,
    coco_document,
    voc_xml,
    yolo_label_text,
)


def make_ann(boxes, width=640, height=480, filename="cat.jpg"):
    return {
        "filename": filename,
        "img_w": width,
        "img_h": height,
        "boxes": [
            {
                "class_id": box.get("class_id", 0),
                "class_name": box.get("class_name", "cat"),
                "confidence": box.get("confidence", 0.9),
                **{k: box[k] for k in ("x1", "y1", "x2", "y2")},
            }
            for box in boxes
        ],
    }


def test_yolo_label_is_normalised_centre_size():
    annotation = make_ann([{"x1": 100, "y1": 100, "x2": 300, "y2": 300}])
    text = yolo_label_text(annotation, {0: 0})
    assert text == "0 0.312500 0.416667 0.312500 0.416667"


def test_yolo_label_clamps_out_of_bounds_and_remaps_ids():
    annotation = make_ann(
        [{"x1": -100, "y1": -100, "x2": 740, "y2": 580, "class_id": 9}]
    )
    text = yolo_label_text(annotation, {9: 0})
    assert text.startswith("0 ")
    values = [float(v) for v in text.split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert abs(values[2] - 1.0) < 1e-6  # 宽度铺满


def test_yolo_label_skips_degenerate_boxes():
    annotation = make_ann([{"x1": 10, "y1": 10, "x2": 10, "y2": 200}])
    assert yolo_label_text(annotation, {0: 0}) == ""


def test_yolo_txt_zip_structure(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "cat.jpg").write_bytes(b"fake-jpeg")

    annotations = {"cat.jpg": make_ann([{"x1": 10, "y1": 10, "x2": 110, "y2": 110}])}
    data, filename = build_export(
        "yolo_txt",
        annotations,
        {0: "cat"},
        upload_dir,
        include_images=True,
        val_ratio=0.0,
    )

    assert filename.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        assert "labels/train/cat.txt" in names
        assert "images/train/cat.jpg" in names
        assert "classes.txt" in names
        assert "data.yaml" in names
        assert archive.read("classes.txt").decode("utf-8") == "cat\n"
        assert archive.read("images/train/cat.jpg") == b"fake-jpeg"
        yaml_text = archive.read("data.yaml").decode("utf-8")
        assert "nc: 1" in yaml_text
        assert 'names: ["cat"]' in yaml_text


def test_train_val_split_is_deterministic_and_splits(tmp_path):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    annotations = {}
    for index in range(40):
        name = f"img{index:02d}.jpg"
        annotations[name] = make_ann([{"x1": 1, "y1": 1, "x2": 50, "y2": 50}], filename=name)

    first, _ = build_export("yolo_txt", annotations, {0: "cat"}, upload_dir, val_ratio=0.2)
    second, _ = build_export("yolo_txt", annotations, {0: "cat"}, upload_dir, val_ratio=0.2)

    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        names = archive.namelist()
    with zipfile.ZipFile(io.BytesIO(second)) as archive:
        assert archive.namelist() == names

    train = [n for n in names if n.startswith("labels/train/")]
    val = [n for n in names if n.startswith("labels/val/")]
    assert train and val
    assert 0.05 < len(val) / (len(train) + len(val)) < 0.45


def test_coco_categories_are_stable_and_contiguous():
    annotations = {
        "a.jpg": make_ann([{"x1": 10, "y1": 10, "x2": 60, "y2": 60, "class_id": 9, "class_name": "dog"}]),
        "b.jpg": make_ann([{"x1": 20, "y1": 20, "x2": 80, "y2": 80, "class_id": 5, "class_name": "cat"}]),
    }
    ids, id_map, names = _class_table({}, annotations)
    assert ids == [5, 9]
    assert names == ["cat", "dog"]
    assert id_map == {5: 0, 9: 1}

    document = coco_document(annotations, id_map, names)
    assert [c["id"] for c in document["categories"]] == [1, 2]
    assert [c["name"] for c in document["categories"]] == ["cat", "dog"]
    assert len(document["images"]) == 2
    assert len(document["annotations"]) == 2

    # 同一份数据重复导出，类别顺序不能变
    again = coco_document(annotations, id_map, names)
    assert again["categories"] == document["categories"]


def test_coco_bbox_and_area():
    annotations = {"a.jpg": make_ann([{"x1": 10, "y1": 20, "x2": 60, "y2": 120}])}
    document = coco_document(annotations, {0: 0}, ["cat"])
    item = document["annotations"][0]
    assert item["bbox"] == [10.0, 20.0, 50.0, 100.0]
    assert item["area"] == 5000.0
    assert item["category_id"] == 1


def test_voc_xml_is_parseable():
    annotation = make_ann(
        [{"x1": 10.4, "y1": 20.6, "x2": 60.2, "y2": 120.8, "class_name": "猫 & 狗"}]
    )
    xml_text = voc_xml("cat.jpg", annotation)
    root = ElementTree.fromstring(xml_text)
    assert root.findtext("filename") == "cat.jpg"
    assert root.findtext("size/width") == "640"
    obj = root.find("object")
    assert obj.findtext("name") == "猫 & 狗"
    assert obj.findtext("bndbox/xmin") == "10"
    assert obj.findtext("bndbox/ymax") == "121"


def test_export_rejects_empty_data():
    with pytest.raises(ExportError):
        build_export("yolo_txt", {}, {}, "/tmp")


def test_export_rejects_unknown_format(tmp_path):
    annotations = {"a.jpg": make_ann([{"x1": 1, "y1": 1, "x2": 10, "y2": 10}])}
    with pytest.raises(ExportError):
        build_export("xml_v99", annotations, {0: "cat"}, tmp_path)
