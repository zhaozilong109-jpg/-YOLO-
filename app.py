"""YOLO 智能标注工具 —— Flask 路由层。

启动：python app.py    然后访问 http://127.0.0.1:5000
"""

from __future__ import annotations

import io
import os
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from core import config
from core.exporter import FORMAT_LABELS, SUPPORTED_FORMATS, ExportError, build_export
from core.index import UploadIndex, bytes_sha256, file_sha256
from core.inference import BatchRunner, InferenceError, ModelManager
from core.storage import (
    AnnotationStore,
    PathTraversalError,
    StorageError,
    resolve_within,
    sanitize_filename,
    sanitize_stem,
    unique_path,
)
from core.video import VIDEO_EXTENSIONS, VideoError, iter_frames, probe

APP_VERSION = "0.3.0"


def _clean_class_filter(raw) -> list[int] | None:
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    result = []
    for item in raw:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(result)) or None


def _float_value(raw, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def create_app(data_dir: Path | str | None = None) -> Flask:
    """应用工厂：测试时可以把数据目录指到临时路径。"""
    root = Path(data_dir).resolve() if data_dir else config.DATA_DIR

    dirs = {
        "UPLOAD_DIR": root / "uploads",
        "ANNOTATION_DIR": root / "annotations",
        "MODEL_DIR": root / "models",
        "EXPORT_DIR": root / "exports",
    }
    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    app = Flask(__name__)
    app.config.update(dirs)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["ALLOWED_EXTENSIONS"] = config.ALLOWED_EXTENSIONS
    app.config["MAX_IMAGE_BYTES"] = config.MAX_IMAGE_BYTES
    app.config["MAX_VIDEO_BYTES"] = config.MAX_VIDEO_BYTES
    app.config["DEDUPE"] = config.DEDUPE
    try:
        app.json.ensure_ascii = False
    except AttributeError:  # Flask < 2.2
        app.config["JSON_AS_ASCII"] = False

    upload_dir: Path = app.config["UPLOAD_DIR"]

    store = AnnotationStore(app.config["ANNOTATION_DIR"])
    models = ModelManager(app.config["MODEL_DIR"])
    batch = BatchRunner(models, store, upload_dir)
    upload_index = UploadIndex(root / "upload_index.json")

    app.extensions["store"] = store
    app.extensions["models"] = models
    app.extensions["batch"] = batch
    app.extensions["upload_index"] = upload_index

    # ------------------------------------------------------------ 工具函数

    def image_path(filename: str) -> Path:
        """把图片名解析成 uploads/ 下的真实路径，越界直接抛错。"""
        path = resolve_within(upload_dir, filename)
        if not path.is_file():
            raise StorageError(f"图片不存在: {filename}")
        return path

    def is_image_name(name: str) -> bool:
        if "." not in name:
            return False
        ext = name.rsplit(".", 1)[1].lower()
        return ext in app.config["ALLOWED_EXTENSIONS"]

    def list_image_names() -> list[str]:
        try:
            entries = list(upload_dir.iterdir())
        except OSError:
            return []
        names = [p.name for p in entries if p.is_file() and is_image_name(p.name)]
        return sorted(names, key=str.lower)

    # 清理被外部删除的图片留下的哈希记录，避免误判「重复」而拒绝上传
    upload_index.sync(set(list_image_names()))

    # ------------------------------------------------------------ 页面

    @app.get("/")
    def index():
        return render_template("index.html", version=APP_VERSION)

    @app.get("/api/health")
    def health():
        return jsonify(
            ok=True,
            version=APP_VERSION,
            model=models.model_name,
            images=len(list_image_names()),
            annotations=store.count(),
        )

    # ------------------------------------------------------------ 图片管理

    @app.post("/api/upload")
    def upload_images():
        files = request.files.getlist("files")
        if not files:
            return jsonify(error="没有选择文件"), 400

        uploaded: list[dict] = []
        skipped: list[dict] = []

        for storage_file in files:
            raw_name = storage_file.filename or ""
            try:
                safe_name = sanitize_filename(
                    raw_name, app.config["ALLOWED_EXTENSIONS"]
                )
            except StorageError as exc:
                skipped.append({"filename": raw_name, "reason": str(exc)})
                continue

            tmp_path = upload_dir / f".upload-{uuid.uuid4().hex}.tmp"
            try:
                storage_file.save(tmp_path)
                size = tmp_path.stat().st_size
                if size == 0:
                    raise StorageError("文件为空")
                if size > app.config["MAX_IMAGE_BYTES"]:
                    raise StorageError(f"图片超过 {config.MAX_IMAGE_MB} MB 上限")

                # 内容去重：同一张图换个名字再传一次，不该变成两份数据
                sha = file_sha256(tmp_path)
                if app.config["DEDUPE"]:
                    existing = upload_index.find(sha)
                    if existing and (upload_dir / existing).exists():
                        skipped.append(
                            {
                                "filename": raw_name,
                                "reason": f"与已有图片重复：{existing}",
                                "duplicate": existing,
                            }
                        )
                        continue

                # 只信任真实的图片内容，不信任扩展名
                with Image.open(tmp_path) as probe_image:
                    probe_image.verify()

                target = unique_path(upload_dir, safe_name)
                os.replace(tmp_path, target)
                upload_index.add(target.name, sha, size)
                uploaded.append(
                    {
                        "filename": target.name,
                        "original": raw_name,
                        "renamed": target.name != raw_name,
                    }
                )
            except (StorageError, UnidentifiedImageError, OSError) as exc:
                skipped.append({"filename": raw_name, "reason": str(exc)})
            finally:
                tmp_path.unlink(missing_ok=True)

        if not uploaded and skipped:
            duplicates = [item for item in skipped if item.get("duplicate")]
            return (
                jsonify(
                    error=(
                        f"{len(duplicates)} 个文件与已有图片重复"
                        if len(duplicates) == len(skipped)
                        else "所有文件都上传失败"
                    ),
                    skipped=skipped,
                    count=0,
                ),
                400,
            )

        return jsonify(
            success=True,
            uploaded=uploaded,
            skipped=skipped,
            count=len(uploaded),
        )

    @app.post("/api/upload_video")
    def upload_video():
        """从视频按固定间隔抽帧，抽出的帧直接进图片列表。"""
        storage_file = request.files.get("video")
        if storage_file is None or not storage_file.filename:
            return jsonify(error="没有选择视频文件"), 400

        raw_name = storage_file.filename
        ext = Path(raw_name.replace("\\", "/")).suffix.lower().lstrip(".")
        if ext not in VIDEO_EXTENSIONS:
            return jsonify(error=f"不支持的视频格式: {ext or '未知'}"), 400

        interval = _float_value(request.form.get("interval"), 1.0)
        max_frames = int(_float_value(request.form.get("max_frames"), 60))

        # 用 ASCII 临时文件名，避免 OpenCV 在中文路径下打不开视频
        tmp_path = upload_dir / f".video-{uuid.uuid4().hex}.{ext}"
        try:
            storage_file.save(tmp_path)
            size = tmp_path.stat().st_size
            if size == 0:
                raise VideoError("视频文件为空")
            if size > app.config["MAX_VIDEO_BYTES"]:
                raise VideoError(f"视频超过 {config.MAX_VIDEO_MB} MB 上限")

            meta = probe(tmp_path)
            stem = sanitize_stem(raw_name, "video")

            created: list[str] = []
            duplicates = 0
            for frame_index, data in iter_frames(
                tmp_path, interval=interval, max_frames=max_frames
            ):
                sha = bytes_sha256(data)
                existing = upload_index.find(sha)
                if existing and (upload_dir / existing).exists():
                    duplicates += 1
                    continue

                target = unique_path(upload_dir, f"{stem}_{frame_index:05d}.jpg")
                target.write_bytes(data)
                upload_index.add(target.name, sha, len(data))
                created.append(target.name)

            if not created and not duplicates:
                raise VideoError("没有抽到任何帧，可能是视频损坏或编码不受支持")

            return jsonify(
                success=True,
                count=len(created),
                duplicates=duplicates,
                frames=created,
                video={
                    "filename": raw_name,
                    "size": size,
                    **meta,
                },
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.get("/api/images")
    def get_images():
        items = []
        for name in list_image_names():
            annotation = store.get(name)
            path = upload_dir / name
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            items.append(
                {
                    "filename": name,
                    "url": url_for("uploaded_file", filename=name),
                    "annotated": annotation is not None,
                    "boxes": len(annotation["boxes"]) if annotation else 0,
                    "size": size,
                }
            )
        return jsonify(images=items, count=len(items))

    @app.get("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        # send_from_directory 自身已做目录穿越防护
        return send_from_directory(upload_dir, filename)

    @app.post("/api/delete_images")
    def delete_images():
        payload = request.get_json(silent=True) or {}
        names = payload.get("filenames") or []
        if not isinstance(names, list):
            return jsonify(error="filenames 必须是数组"), 400

        deleted = []
        for name in names:
            try:
                path = image_path(str(name))
            except (StorageError, PathTraversalError):
                continue
            try:
                path.unlink()
                store.delete(path.name)
                upload_index.remove(path.name)
                deleted.append(path.name)
            except OSError:
                continue

        return jsonify(success=True, deleted=deleted, count=len(deleted))

    @app.post("/api/clear_all")
    def clear_all():
        removed = 0
        for name in list_image_names():
            try:
                (upload_dir / name).unlink()
                removed += 1
            except OSError:
                continue
        store.clear()
        upload_index.clear()
        return jsonify(success=True, removed=removed)

    # ------------------------------------------------------------ 模型

    @app.get("/api/models")
    def get_models():
        return jsonify(models.available())

    @app.post("/api/upload_model")
    def upload_model():
        storage_file = request.files.get("model")
        if storage_file is None or not storage_file.filename:
            return jsonify(error="没有选择模型文件"), 400

        raw_name = os.path.basename(storage_file.filename.replace("\\", "/"))
        if not raw_name.lower().endswith(".pt"):
            return jsonify(error="仅支持 .pt 格式的模型文件"), 400

        safe_name = f"{Path(raw_name).stem[:80]}.pt"
        target = unique_path(Path(app.config["MODEL_DIR"]), safe_name)
        storage_file.save(target)
        return jsonify(success=True, name=target.name)

    @app.post("/api/load_model")
    def load_model():
        payload = request.get_json(silent=True) or {}
        name = payload.get("model_name")
        if not name:
            return jsonify(error="未指定模型"), 400
        classes = models.load(str(name))
        return jsonify(success=True, model_name=models.model_name, classes=classes)

    @app.get("/api/classes")
    def get_classes():
        return jsonify(classes=models.classes(), model=models.model_name)

    # ------------------------------------------------------------ 标注

    @app.get("/api/annotation/<path:filename>")
    def get_annotation(filename: str):
        annotation = store.get(filename)
        if annotation is not None:
            return jsonify(annotation)

        # 没有标注时返回图片的真实尺寸，方便前端直接画框
        img_w = img_h = 0
        try:
            with Image.open(image_path(filename)) as image:
                img_w, img_h = image.size
        except (StorageError, UnidentifiedImageError, OSError):
            pass
        return jsonify(filename=filename, img_w=img_w, img_h=img_h, boxes=[])

    @app.put("/api/annotation/<path:filename>")
    def save_annotation(filename: str):
        """保存人工修正后的标注（前端拖动/新增/删除后调用）。"""
        image_path(filename)  # 确保这张图确实存在，避免写入垃圾数据
        payload = request.get_json(silent=True) or {}

        img_w = int(_float_value(payload.get("img_w"), 0))
        img_h = int(_float_value(payload.get("img_h"), 0))
        if img_w <= 0 or img_h <= 0:
            try:
                with Image.open(image_path(filename)) as image:
                    img_w, img_h = image.size
            except (UnidentifiedImageError, OSError):
                return jsonify(error="读取图片尺寸失败"), 400

        boxes = payload.get("boxes")
        if not isinstance(boxes, list):
            return jsonify(error="boxes 必须是数组"), 400

        saved = store.put(filename, boxes, img_w, img_h)
        return jsonify(success=True, **saved)

    @app.post("/api/annotate")
    def annotate_single():
        payload = request.get_json(silent=True) or {}
        filename = payload.get("filename")
        if not filename:
            return jsonify(error="缺少图片文件名"), 400

        path = image_path(str(filename))
        boxes, img_w, img_h = models.annotate(
            path,
            conf=_float_value(payload.get("confidence"), 0.25),
            iou=_float_value(payload.get("iou"), 0.45),
            classes=_clean_class_filter(payload.get("classes")),
        )
        saved = store.put(path.name, boxes, img_w, img_h)
        return jsonify(success=True, **saved)

    @app.post("/api/batch")
    def start_batch():
        payload = request.get_json(silent=True) or {}
        include_annotated = bool(payload.get("include_annotated", False))

        files = [
            name
            for name in list_image_names()
            if include_annotated or not store.has(name)
        ]
        if not files:
            return (
                jsonify(
                    error="没有待标注的图片",
                    message="所有图片都已标注，可勾选「重新标注全部」再试",
                ),
                400,
            )

        job = batch.start(
            files,
            conf=_float_value(payload.get("confidence"), 0.25),
            iou=_float_value(payload.get("iou"), 0.45),
            classes=_clean_class_filter(payload.get("classes")),
        )
        return jsonify(success=True, **job)

    @app.get("/api/batch/<job_id>")
    def batch_status(job_id: str):
        job = batch.snapshot(job_id)
        if job is None:
            return jsonify(error="任务不存在"), 404
        return jsonify(job)

    @app.post("/api/batch/<job_id>/cancel")
    def batch_cancel(job_id: str):
        if not batch.cancel(job_id):
            return jsonify(error="任务不存在"), 404
        return jsonify(success=True, **batch.snapshot(job_id))

    @app.get("/api/stats")
    def stats():
        names = list_image_names()
        annotated = sum(1 for name in names if store.has(name))
        items = store.all_items()
        return jsonify(
            total=len(names),
            annotated=annotated,
            pending=len(names) - annotated,
            boxes=sum(len(item["boxes"]) for item in items.values()),
            class_names={str(k): v for k, v in store.class_names().items()},
        )

    # ------------------------------------------------------------ 导出

    @app.post("/api/export")
    def export_annotations():
        payload = request.get_json(silent=True) or {}
        fmt = str(payload.get("format") or "yolo_txt")
        include_images = bool(payload.get("include_images", False))
        val_ratio = min(
            max(_float_value(payload.get("val_ratio"), 0.0), 0.0),
            config.MAX_VAL_RATIO,
        )

        annotations = store.all_items()
        if not annotations:
            return jsonify(error="没有标注数据可导出，请先标注图片"), 400

        class_map = {**store.class_names(), **models.classes()}
        data, filename = build_export(
            fmt,
            annotations,
            class_map,
            upload_dir,
            include_images=include_images,
            val_ratio=val_ratio,
            only_used_classes=bool(payload.get("only_used_classes", True)),
        )

        return send_file(
            io.BytesIO(data),
            mimetype="application/zip",
            as_attachment=True,
            download_name=filename,
        )

    @app.get("/api/export/formats")
    def export_formats():
        return jsonify(
            formats=[
                {"value": key, "label": label}
                for key, label in FORMAT_LABELS.items()
                if key in SUPPORTED_FORMATS
            ]
        )

    # ------------------------------------------------------------ 关闭服务

    @app.post("/api/shutdown")
    def shutdown():
        def _exit_soon():
            time.sleep(0.4)
            os._exit(0)

        threading.Thread(target=_exit_soon, daemon=True).start()
        return jsonify(success=True, message="服务已关闭")

    # ------------------------------------------------------------ 错误处理

    @app.errorhandler(StorageError)
    def handle_storage_error(exc):  # noqa: ANN001
        return jsonify(error=str(exc)), 400

    @app.errorhandler(InferenceError)
    def handle_inference_error(exc):  # noqa: ANN001
        return jsonify(error=str(exc)), 400

    @app.errorhandler(ExportError)
    def handle_export_error(exc):  # noqa: ANN001
        return jsonify(error=str(exc)), 400

    @app.errorhandler(VideoError)
    def handle_video_error(exc):  # noqa: ANN001
        return jsonify(error=str(exc)), 400

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(exc):  # noqa: ANN001
        limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify(error=f"文件太大，单次上传上限 {limit_mb} MB"), 413

    @app.errorhandler(404)
    def handle_404(exc):  # noqa: ANN001
        if request.path.startswith("/api/"):
            return jsonify(error="接口不存在"), 404
        return jsonify(error="页面不存在"), 404

    @app.after_request
    def no_cache_api(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()


if __name__ == "__main__":
    print("=" * 56)
    print(f"  YOLO 智能标注工具 v{APP_VERSION}")
    print(f"  数据目录: {config.DATA_DIR}")
    print(f"  启动地址: http://{config.HOST}:{config.PORT}")
    print("=" * 56)
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
