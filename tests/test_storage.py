"""文件名清洗与标注落盘的测试。"""

import json

import pytest

from core.storage import (
    AnnotationStore,
    PathTraversalError,
    StorageError,
    resolve_within,
    sanitize_filename,
    unique_path,
)

ALLOWED = {"jpg", "jpeg", "png"}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("cat.jpg", "cat.jpg"),
        ("猫.jpg", "猫.jpg"),
        ("数据集_001.png", "数据集_001.png"),
        ("小狗 (1).JPG", "小狗 (1).jpg"),
        ("a.b.png", "a.b.png"),
        ("../evil.png", "evil.png"),
        ("C:\\tmp\\x.jpg", "x.jpg"),
        ("bad<>name.jpg", "bad__name.jpg"),
    ],
)
def test_sanitize_keeps_unicode(raw, expected):
    assert sanitize_filename(raw, ALLOWED) == expected


def test_sanitize_rejects_unknown_extension():
    with pytest.raises(StorageError):
        sanitize_filename("shell.py", ALLOWED)


def test_sanitize_handles_reserved_and_empty_names():
    assert sanitize_filename("CON.jpg", ALLOWED) == "_CON.jpg"
    assert sanitize_filename(".jpg", ALLOWED) == "image.jpg"


def test_sanitize_limits_length():
    long_name = "长" * 200 + ".jpg"
    assert len(sanitize_filename(long_name, ALLOWED)) <= 84


def test_resolve_within_blocks_traversal(tmp_path):
    base = tmp_path / "uploads"
    base.mkdir()
    assert resolve_within(base, "a.jpg") == (base / "a.jpg").resolve()

    for bad in ("../a.jpg", "..\\a.jpg", "sub/a.jpg", "..", ""):
        with pytest.raises(PathTraversalError):
            resolve_within(base, bad)


def test_unique_path_adds_suffix(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert unique_path(tmp_path, "a.jpg").name == "a_1.jpg"
    (tmp_path / "a_1.jpg").write_bytes(b"x")
    assert unique_path(tmp_path, "a.jpg").name == "a_2.jpg"


def test_store_roundtrip_and_persistence(tmp_path):
    store = AnnotationStore(tmp_path)
    saved = store.put(
        "猫.jpg",
        [
            {"x1": 10, "y1": 20, "x2": 110, "y2": 220, "confidence": 0.9, "class_id": 3, "class_name": "cat"},
            # 退化框应被丢弃
            {"x1": 50, "y1": 50, "x2": 50, "y2": 50, "class_id": 1, "class_name": "x"},
        ],
        640,
        480,
    )
    assert len(saved["boxes"]) == 1
    assert saved["boxes"][0]["class_id"] == 3

    # 落盘了
    json_file = tmp_path / "猫.jpg.json"
    assert json_file.exists()
    assert json.loads(json_file.read_text(encoding="utf-8"))["filename"] == "猫.jpg"

    # 新实例（模拟重启）能读回来
    reloaded = AnnotationStore(tmp_path)
    assert reloaded.count() == 1
    assert reloaded.get("猫.jpg")["boxes"][0]["x2"] == 110


def test_store_clamps_boxes_to_image_bounds(tmp_path):
    store = AnnotationStore(tmp_path)
    saved = store.put(
        "a.jpg",
        [{"x1": -50, "y1": -20, "x2": 900, "y2": 700, "class_id": 0, "class_name": "obj"}],
        640,
        480,
    )
    box = saved["boxes"][0]
    assert box["x1"] == 0 and box["y1"] == 0
    assert box["x2"] == 640 and box["y2"] == 480


def test_store_delete_and_clear(tmp_path):
    store = AnnotationStore(tmp_path)
    store.put("a.jpg", [{"x1": 1, "y1": 1, "x2": 100, "y2": 100, "class_id": 0, "class_name": "o"}], 640, 480)
    assert store.delete("a.jpg") is True
    assert store.get("a.jpg") is None
    assert not (tmp_path / "a.jpg.json").exists()

    store.put("b.jpg", [], 640, 480)
    store.clear()
    assert store.count() == 0
    assert list(tmp_path.glob("*.json")) == []


def test_store_ignores_corrupt_json(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    store = AnnotationStore(tmp_path)
    assert store.count() == 0
