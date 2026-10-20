"""把标注结果导出成主流数据集格式，并打包为 zip。

支持：YOLO TXT、YOLO JSON、COCO JSON、Pascal VOC，以及带 train/val 划分的
YOLO 训练集（images/ + labels/ + data.yaml + classes.txt），解压即可 `yolo train`。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from .storage import resolve_within

YOLO_FORMATS = {"yolo_txt", "yolo_json", "yolo_seg"}
SUPPORTED_FORMATS = {
    "yolo_txt",
    "yolo_seg",
    "yolo_json",
    "coco_json",
    "pascal_voc",
}
FORMAT_LABELS = {
    "yolo_txt": "YOLO TXT",
    "yolo_seg": "YOLO SEG",
    "yolo_json": "YOLO JSON",
    "coco_json": "COCO JSON",
    "pascal_voc": "Pascal VOC",
}


class ExportError(RuntimeError):
    """导出失败（会以 400 返回给前端）。"""


# ---------------------------------------------------------------- 基础处理


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _clean_polygon(raw_points, img_w: float, img_h: float) -> list[list[float]] | None:
    """把多边形顶点裁剪到图像范围，少于 3 点视为无效。"""
    if not isinstance(raw_points, (list, tuple)):
        return None

    points: list[list[float]] = []
    for pair in raw_points:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            x = float(pair[0])
            y = float(pair[1])
        except (TypeError, ValueError):
            continue
        if img_w > 0:
            x = _clamp(x, 0, img_w)
        if img_h > 0:
            y = _clamp(y, 0, img_h)
        points.append([x, y])

    return points if len(points) >= 3 else None


def polygon_of(box: dict) -> list[list[float]]:
    """取多边形的顶点；矩形框则返回它的四个角。"""
    return box.get("points") or [
        [box["x1"], box["y1"]],
        [box["x2"], box["y1"]],
        [box["x2"], box["y2"]],
        [box["x1"], box["y2"]],
    ]


def polygon_area(points: list[list[float]]) -> float:
    """鞋带公式求多边形面积。"""
    total = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def normalised_boxes(annotation: dict) -> list[dict]:
    """裁剪到图像范围内，丢弃退化框，返回统一结构的列表。"""
    try:
        img_w = float(annotation.get("img_w") or 0)
        img_h = float(annotation.get("img_h") or 0)
    except (TypeError, ValueError):
        return []

    result = []
    for raw in annotation.get("boxes") or []:
        try:
            x1 = float(raw["x1"])
            y1 = float(raw["y1"])
            x2 = float(raw["x2"])
            y2 = float(raw["y2"])
        except (KeyError, TypeError, ValueError):
            continue

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if img_w > 0:
            x1, x2 = _clamp(x1, 0, img_w), _clamp(x2, 0, img_w)
        if img_h > 0:
            y1, y2 = _clamp(y1, 0, img_h), _clamp(y2, 0, img_h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue

        item = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class_id": int(raw.get("class_id", 0)),
            "class_name": str(raw.get("class_name") or "object"),
            "confidence": float(raw.get("confidence", 1.0)),
        }
        points = _clean_polygon(raw.get("points"), img_w, img_h)
        if points:
            item["points"] = points
        result.append(item)
    return result


def _class_table(
    class_map: dict[int, str],
    annotations: dict[str, dict],
    only_used: bool = False,
) -> tuple[list[int], dict[int, int], list[str]]:
    """生成稳定的类别表。

    类别 id 按旧 id 升序重新编号成 0..N-1，这样同一份数据分几次导出，
    类别顺序也不会变（原来的实现按出现顺序编号，多次导出 COCO 会错位）。

    ``only_used=True`` 时只保留真正出现过的类别，避免把 COCO 的 80 类
    全写进 data.yaml（只有 3 类数据的训练集不该声明 80 类）。
    """
    used: set[int] = set()
    fallback_names: dict[int, str] = {}
    for annotation in annotations.values():
        for box in annotation.get("boxes") or []:
            try:
                class_id = int(box.get("class_id", 0))
            except (TypeError, ValueError):
                continue
            used.add(class_id)
            fallback_names.setdefault(class_id, str(box.get("class_name") or "object"))

    if only_used and used:
        ids = sorted(used)
    else:
        ids = sorted(set(int(k) for k in class_map) | used)

    names = [class_map.get(i) or fallback_names.get(i) or f"class_{i}" for i in ids]
    return ids, {old: new for new, old in enumerate(ids)}, names


def _stem(filename: str) -> str:
    return Path(filename).stem


def _is_val(filename: str, val_ratio: float) -> bool:
    """按文件名哈希划分，保证多次导出结果一致。"""
    if val_ratio <= 0:
        return False
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    threshold = int(_clamp(val_ratio, 0, 1) * 1000)
    return int(digest[:8], 16) % 1000 < threshold


# ---------------------------------------------------------------- 单格式内容


def yolo_label_text(annotation: dict, id_map: dict[int, int]) -> str:
    img_w = float(annotation.get("img_w") or 0)
    img_h = float(annotation.get("img_h") or 0)
    if img_w <= 0 or img_h <= 0:
        return ""

    lines = []
    for box in normalised_boxes(annotation):
        if box["class_id"] not in id_map:
            continue
        x_center = (box["x1"] + box["x2"]) / 2 / img_w
        y_center = (box["y1"] + box["y2"]) / 2 / img_h
        width = (box["x2"] - box["x1"]) / img_w
        height = (box["y2"] - box["y1"]) / img_h
        lines.append(
            f"{id_map[box['class_id']]} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )
    return "\n".join(lines)


def yolo_seg_text(annotation: dict, id_map: dict[int, int]) -> str:
    """YOLO 分割标注：``class x1 y1 x2 y2 ...``（归一化多边形顶点）。

    矩形框也能导出——按四个角当作四边形处理。
    """
    img_w = float(annotation.get("img_w") or 0)
    img_h = float(annotation.get("img_h") or 0)
    if img_w <= 0 or img_h <= 0:
        return ""

    lines = []
    for box in normalised_boxes(annotation):
        if box["class_id"] not in id_map:
            continue

        coords = []
        for x, y in polygon_of(box):
            coords.append(f"{_clamp(x / img_w, 0.0, 1.0):.6f}")
            coords.append(f"{_clamp(y / img_h, 0.0, 1.0):.6f}")
        if len(coords) < 6:
            continue

        lines.append(f"{id_map[box['class_id']]} " + " ".join(coords))
    return "\n".join(lines)


def yolo_json_document(filename: str, annotation: dict) -> dict:
    document = {
        "version": "1.0",
        "filename": filename,
        "image_width": int(annotation.get("img_w") or 0),
        "image_height": int(annotation.get("img_h") or 0),
        "annotations": [
            {
                "class_id": box["class_id"],
                "class_name": box["class_name"],
                "bbox": [
                    round(box["x1"], 2),
                    round(box["y1"], 2),
                    round(box["x2"], 2),
                    round(box["y2"], 2),
                ],
                "confidence": round(box["confidence"], 4),
            }
            for box in normalised_boxes(annotation)
        ],
    }

    # 有多边形就一并带上，消费方按需取用
    polygons = [
        {
            "class_id": box["class_id"],
            "class_name": box["class_name"],
            "points": [[round(x, 2), round(y, 2)] for x, y in box["points"]],
        }
        for box in normalised_boxes(annotation)
        if box.get("points")
    ]
    if polygons:
        document["polygons"] = polygons
    return document


def coco_document(
    annotations: dict[str, dict],
    id_map: dict[int, int],
    names: list[str],
) -> dict:
    images = []
    anns = []
    ann_id = 1

    for index, (filename, annotation) in enumerate(sorted(annotations.items()), start=1):
        images.append(
            {
                "id": index,
                "file_name": filename,
                "width": int(annotation.get("img_w") or 0),
                "height": int(annotation.get("img_h") or 0),
            }
        )
        for box in normalised_boxes(annotation):
            if box["class_id"] not in id_map:
                continue
            width = box["x2"] - box["x1"]
            height = box["y2"] - box["y1"]
            polygon = box.get("points")
            segmentation = (
                [[round(coord, 2) for point in polygon for coord in point]]
                if polygon
                else [[
                    round(box["x1"], 2), round(box["y1"], 2),
                    round(box["x2"], 2), round(box["y1"], 2),
                    round(box["x2"], 2), round(box["y2"], 2),
                    round(box["x1"], 2), round(box["y2"], 2),
                ]]
            )
            anns.append(
                {
                    "id": ann_id,
                    "image_id": index,
                    "category_id": id_map[box["class_id"]] + 1,
                    "bbox": [
                        round(box["x1"], 2),
                        round(box["y1"], 2),
                        round(width, 2),
                        round(height, 2),
                    ],
                    "area": round(
                        polygon_area(polygon) if polygon else width * height, 2
                    ),
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": "YOLO 智能标注工具导出",
            "version": "1.0",
            "date_created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "none"}
            for index, name in enumerate(names)
        ],
    }


def voc_xml(filename: str, annotation: dict) -> str:
    img_w = int(annotation.get("img_w") or 0)
    img_h = int(annotation.get("img_h") or 0)

    parts = [
        "<annotation>",
        f"  <folder>images</folder>",
        f"  <filename>{escape(filename)}</filename>",
        "  <source><database>YOLO 智能标注工具</database></source>",
        "  <size>",
        f"    <width>{img_w}</width>",
        f"    <height>{img_h}</height>",
        "    <depth>3</depth>",
        "  </size>",
        "  <segmented>0</segmented>",
    ]

    for box in normalised_boxes(annotation):
        xmin = int(math.floor(box["x1"]))
        ymin = int(math.floor(box["y1"]))
        xmax = int(math.ceil(box["x2"]))
        ymax = int(math.ceil(box["y2"]))
        if img_w:
            xmin, xmax = min(max(xmin, 0), img_w), min(max(xmax, 0), img_w)
        if img_h:
            ymin, ymax = min(max(ymin, 0), img_h), min(max(ymax, 0), img_h)
        parts += [
            "  <object>",
            f"    <name>{escape(box['class_name'])}</name>",
            "    <pose>Unspecified</pose>",
            "    <truncated>0</truncated>",
            "    <difficult>0</difficult>",
            "    <bndbox>",
            f"      <xmin>{xmin}</xmin>",
            f"      <ymin>{ymin}</ymin>",
            f"      <xmax>{xmax}</xmax>",
            f"      <ymax>{ymax}</ymax>",
            "    </bndbox>",
            "  </object>",
        ]

    parts.append("</annotation>")
    return "\n".join(parts)


def _classes_txt(names: list[str]) -> str:
    return "\n".join(names) + ("\n" if names else "")


def _data_yaml(names: list[str], task: str | None = None) -> str:
    quoted = json.dumps(names, ensure_ascii=False)
    lines = ["# 由 YOLO 智能标注工具生成"]
    if task:
        lines.append(f"task: {task}")
    lines += [
        "path: .",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        f"names: {quoted}",
        "",
    ]
    return "\n".join(lines)


def _read_image(upload_dir: Path, filename: str) -> bytes | None:
    try:
        path = resolve_within(upload_dir, filename)
    except Exception:  # noqa: BLE001 - 名字非法就跳过
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


# ---------------------------------------------------------------- 打包


def build_export(
    fmt: str,
    annotations: dict[str, dict],
    class_map: dict[int, str],
    upload_dir: Path,
    include_images: bool = False,
    val_ratio: float = 0.0,
    only_used_classes: bool = False,
) -> tuple[bytes, str]:
    """返回 (zip 字节流, 建议文件名)。"""
    if fmt not in SUPPORTED_FORMATS:
        raise ExportError(f"不支持的导出格式: {fmt}")
    if not annotations:
        raise ExportError("没有标注数据可导出，请先标注至少一张图片")

    ids, id_map, names = _class_table(class_map, annotations, only_used_classes)
    files: dict[str, bytes] = {}

    if fmt == "yolo_txt":
        has_val = False
        for filename, annotation in sorted(annotations.items()):
            split = "val" if _is_val(filename, val_ratio) else "train"
            has_val = has_val or split == "val"
            label = yolo_label_text(annotation, id_map)
            files[f"labels/{split}/{_stem(filename)}.txt"] = label.encode("utf-8")
            if include_images:
                content = _read_image(upload_dir, filename)
                if content is not None:
                    files[f"images/{split}/{filename}"] = content

        files["classes.txt"] = _classes_txt(names).encode("utf-8")
        files["data.yaml"] = _data_yaml(names).encode("utf-8")
        files["README.txt"] = _dataset_readme(
            names, include_images, val_ratio, has_val
        ).encode("utf-8")

    elif fmt == "yolo_seg":
        has_val = False
        for filename, annotation in sorted(annotations.items()):
            split = "val" if _is_val(filename, val_ratio) else "train"
            has_val = has_val or split == "val"
            label = yolo_seg_text(annotation, id_map)
            files[f"labels/{split}/{_stem(filename)}.txt"] = label.encode("utf-8")
            if include_images:
                content = _read_image(upload_dir, filename)
                if content is not None:
                    files[f"images/{split}/{filename}"] = content

        files["classes.txt"] = _classes_txt(names).encode("utf-8")
        files["data.yaml"] = _data_yaml(names, task="segment").encode("utf-8")
        files["README.txt"] = _dataset_readme(
            names, include_images, val_ratio, has_val, task="segment"
        ).encode("utf-8")

    elif fmt == "yolo_json":
        for filename, annotation in sorted(annotations.items()):
            document = yolo_json_document(filename, annotation)
            files[f"labels/{_stem(filename)}.json"] = json.dumps(
                document, ensure_ascii=False, indent=2
            ).encode("utf-8")
            if include_images:
                content = _read_image(upload_dir, filename)
                if content is not None:
                    files[f"images/{filename}"] = content
        files["classes.txt"] = _classes_txt(names).encode("utf-8")

    elif fmt == "coco_json":
        document = coco_document(annotations, id_map, names)
        files["annotations/instances.json"] = json.dumps(
            document, ensure_ascii=False, indent=2
        ).encode("utf-8")
        if include_images:
            for filename in sorted(annotations):
                content = _read_image(upload_dir, filename)
                if content is not None:
                    files[f"images/{filename}"] = content
        files["classes.txt"] = _classes_txt(names).encode("utf-8")

    else:  # pascal_voc
        for filename, annotation in sorted(annotations.items()):
            files[f"Annotations/{_stem(filename)}.xml"] = voc_xml(
                filename, annotation
            ).encode("utf-8")
            if include_images:
                content = _read_image(upload_dir, filename)
                if content is not None:
                    files[f"JPEGImages/{filename}"] = content
        files["classes.txt"] = _classes_txt(names).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, content in files.items():
            archive.writestr(arcname, content)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    return buffer.getvalue(), f"{fmt}_{stamp}.zip"


def _dataset_readme(
    names: list[str],
    include_images: bool,
    val_ratio: float,
    has_val: bool,
    task: str = "detect",
) -> str:
    if task == "segment":
        label_line = "  labels/train, labels/val   对应的 YOLO 分割标注（class x1 y1 x2 y2 …，归一化）"
        train_line = "  yolo segment train data=data.yaml model=yolov8n-seg.pt imgsz=640 epochs=100"
    else:
        label_line = "  labels/train, labels/val   对应的 YOLO 标注（class cx cy w h，归一化）"
        train_line = "  yolo detect train data=data.yaml model=yolov8n.pt imgsz=640 epochs=100"

    lines = [
        "YOLO 智能标注工具 —— 数据集导出说明",
        "=" * 40,
        f"类别数: {len(names)}",
        f"类别: {', '.join(names) if names else '（无）'}",
        "",
        "目录结构:",
        "  images/train, images/val   原始图片",
        label_line,
        "  data.yaml                  ultralytics 训练配置",
        "  classes.txt                类别名，行号即 class id",
        "",
    ]
    if include_images:
        lines.append(
            f"训练集/验证集按文件名哈希划分（val_ratio={val_ratio:.2f}），"
            f"本次{'包含' if has_val else '不包含'}验证集图片。"
        )
        if not has_val:
            lines.append("提示：本次没有图片被分到 val，可在导出时调大划分比例。")
    else:
        lines.append(
            "本次导出不含图片，labels/ 下的标注请与同名图片放在一起后使用。"
        )
    lines += [
        "",
        "训练示例:",
        train_line,
        "",
    ]
    return "\n".join(lines)
