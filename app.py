"""
YOLO 智能标注工具 - 后端服务
基于 Flask + Ultralytics YOLO
"""
import os
import json
import shutil
import threading
import time
import uuid

import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom

from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, url_for
)
from werkzeug.utils import secure_filename

# ========== 配置（使用绝对路径）==========
# 获取 app.py 所在的目录作为项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)


def _path(folder):
    """将相对路径转为项目根目录下的绝对路径"""
    return os.path.join(BASE_DIR, folder)


app.config['UPLOAD_FOLDER'] = _path('uploads')
app.config['ANNOTATION_FOLDER'] = _path('annotations')
app.config['MODEL_FOLDER'] = _path('models')
app.config['EXPORT_FOLDER'] = _path('exports')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 最大上传 50MB
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# 确保目录存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['ANNOTATION_FOLDER'],
               app.config['MODEL_FOLDER'], app.config['EXPORT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 全局变量：当前加载的模型
current_model = None
current_model_name = None

# 保存每张图片的标注结果
# 结构: { "图片文件名": {"boxes": [...], "img_w": 100, "img_h": 100} }
annotations_store = {}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ========== 页面路由 ==========

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


# ========== 图片管理 API ==========

@app.route('/api/upload', methods=['POST'])
def upload_images():
    """上传图片"""
    if 'files' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    files = request.files.getlist('files')
    uploaded = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # 避免重名
            name, ext = os.path.splitext(filename)
            save_name = filename
            counter = 1
            while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], save_name)):
                save_name = f"{name}_{counter}{ext}"
                counter += 1

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], save_name))
            uploaded.append(save_name)

    return jsonify({
        'success': True,
        'uploaded': uploaded,
        'count': len(uploaded)
    })


@app.route('/api/images', methods=['GET'])
def get_images():
    """获取所有已上传的图片列表"""
    image_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(image_dir):
        return jsonify({'images': []})

    images = []
    for f in sorted(os.listdir(image_dir)):
        if allowed_file(f):
            status = 'annotated' if f in annotations_store else 'pending'
            images.append({
                'filename': f,
                'url': url_for('uploaded_file', filename=f),
                'status': status
            })

    return jsonify({'images': images})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的图片文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/delete_images', methods=['POST'])
def delete_images():
    """删除选中的图片"""
    filenames = request.json.get('filenames', [])
    for f in filenames:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f)
        if os.path.exists(filepath):
            os.remove(filepath)
        # 同时清除标注
        if f in annotations_store:
            del annotations_store[f]
    return jsonify({'success': True})


@app.route('/api/clear_all', methods=['POST'])
def clear_all():
    """清空所有图片和标注"""
    for folder in [app.config['UPLOAD_FOLDER'], app.config['ANNOTATION_FOLDER']]:
        for f in os.listdir(folder):
            fpath = os.path.join(folder, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
    annotations_store.clear()
    return jsonify({'success': True})


# ========== 模型管理 API ==========

@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表（预训练 + 自定义）"""
    # 预训练模型
    pretrained = [
        {'name': 'yolov8n.pt', 'label': 'YOLOv8n (轻量)'},
        {'name': 'yolov8s.pt', 'label': 'YOLOv8s (小型)'},
        {'name': 'yolov8m.pt', 'label': 'YOLOv8m (中等)'},
        {'name': 'yolov5su.pt', 'label': 'YOLOv5su (轻量)'},
    ]

    # 自定义模型（用户上传的）
    custom = []
    model_dir = app.config['MODEL_FOLDER']
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith('.pt'):
                custom.append({
                    'name': f,
                    'label': f'自定义: {f}',
                    'custom': True
                })

    return jsonify({
        'pretrained': pretrained,
        'custom': custom
    })


@app.route('/api/upload_model', methods=['POST'])
def upload_model():
    """上传自定义模型文件 (.pt)"""
    if 'model' not in request.files:
        return jsonify({'error': '没有选择模型文件'}), 400

    file = request.files['model']
    if file and file.filename.endswith('.pt'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['MODEL_FOLDER'], filename))
        return jsonify({'success': True, 'name': filename})
    return jsonify({'error': '仅支持 .pt 格式的模型文件'}), 400


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """加载选中的模型"""
    global current_model, current_model_name

    model_name = request.json.get('model_name')
    if not model_name:
        return jsonify({'error': '未指定模型'}), 400

    try:
        from ultralytics import YOLO

        # 先检查 models 目录下是否有
        model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)

        if not os.path.exists(model_path):
            # 预训练模型：让 ultralytics 下载到项目根目录，再移到 models 目录
            print(f"[INFO] 下载预训练模型: {model_name}")
            # 先切到项目根目录下载，避免路径混乱
            old_cwd = os.getcwd()
            os.chdir(BASE_DIR)
            try:
                tmp_model = YOLO(model_name)
                # 下载完成后将 .pt 文件移到 models 目录
                src = os.path.join(BASE_DIR, model_name)
                if os.path.exists(src):
                    shutil.move(src, model_path)
                os.chdir(old_cwd)
            except Exception:
                os.chdir(old_cwd)
                raise

        # 从 models 目录加载
        current_model = YOLO(model_path)
        current_model_name = model_name

        return jsonify({
            'success': True,
            'model_name': model_name,
            'classes': current_model.names  # 返回类别名称
        })
    except Exception as e:
        return jsonify({'error': f'模型加载失败: {str(e)}'}), 500


# ========== 标注 API ==========

@app.route('/api/annotate', methods=['POST'])
def annotate_single():
    """标注单张图片"""
    global current_model

    if current_model is None:
        return jsonify({'error': '请先加载模型'}), 400

    filename = request.json.get('filename')
    confidence = float(request.json.get('confidence', 0.25))
    iou = float(request.json.get('iou', 0.45))

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(img_path):
        return jsonify({'error': '图片不存在'}), 404

    try:
        # 用 YOLO 进行推理
        results = current_model(
            img_path,
            conf=confidence,
            iou=iou,
            verbose=False
        )[0]

        # 读取图片尺寸（无论有无检测框都需要）
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # 解析检测结果
        boxes_data = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_data.append({
                    'x1': round(x1, 1),
                    'y1': round(y1, 1),
                    'x2': round(x2, 1),
                    'y2': round(y2, 1),
                    'confidence': round(float(box.conf[0]), 4),
                    'class_id': int(box.cls[0]),
                    'class_name': results.names[int(box.cls[0])]
                })

        # 保存标注结果
        annotations_store[filename] = {
            'boxes': boxes_data,
            'img_w': img_w,
            'img_h': img_h
        }

        return jsonify({
            'success': True,
            'filename': filename,
            'boxes': boxes_data,
            'img_w': img_w,
            'img_h': img_h
        })

    except Exception as e:
        return jsonify({'error': f'标注失败: {str(e)}'}), 500


@app.route('/api/annotate_batch', methods=['POST'])
def annotate_batch():
    """批量标注 - 每次请求标注一张，配合前端轮询进度"""
    global current_model

    if current_model is None:
        return jsonify({'error': '请先加载模型'}), 400

    confidence = float(request.json.get('confidence', 0.25))
    iou = float(request.json.get('iou', 0.45))

    # 找出所有未标注的图片
    image_dir = app.config['UPLOAD_FOLDER']
    unannotated = []
    for f in sorted(os.listdir(image_dir)):
        if allowed_file(f) and f not in annotations_store:
            unannotated.append(f)

    total = len(unannotated)
    done = len([f for f in os.listdir(image_dir) if allowed_file(f)]) - total

    if total == 0:
        return jsonify({
            'success': True,
            'completed': True,
            'done': done,
            'total': done,
            'message': '所有图片均已标注完成'
        })

    # 标注下一张
    filename = unannotated[0]
    try:
        results = current_model(
            os.path.join(image_dir, filename),
            conf=confidence,
            iou=iou,
            verbose=False
        )[0]

        img = cv2.imread(os.path.join(image_dir, filename))
        img_h, img_w = img.shape[:2]

        boxes_data = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes_data.append({
                    'x1': round(x1, 1),
                    'y1': round(y1, 1),
                    'x2': round(x2, 1),
                    'y2': round(y2, 1),
                    'confidence': round(float(box.conf[0]), 4),
                    'class_id': int(box.cls[0]),
                    'class_name': results.names[int(box.cls[0])]
                })

        annotations_store[filename] = {
            'boxes': boxes_data,
            'img_w': img_w,
            'img_h': img_h
        }

        # 检查是否全部完成
        remaining = total - 1
        done_now = done + 1
        is_completed = remaining == 0

        return jsonify({
            'success': True,
            'completed': is_completed,
            'filename': filename,
            'boxes': boxes_data,
            'done': done_now,
            'remaining': remaining,
            'total': done_now + remaining
        })

    except Exception as e:
        return jsonify({'error': f'标注失败: {filename} - {str(e)}'}), 500


@app.route('/api/annotation/<filename>', methods=['GET'])
def get_annotation(filename):
    """获取某张图片的标注结果"""
    if filename in annotations_store:
        return jsonify(annotations_store[filename])
    return jsonify({'boxes': [], 'img_w': 0, 'img_h': 0})


@app.route('/api/annotations_count', methods=['GET'])
def annotations_count():
    """获取标注进度统计"""
    image_dir = app.config['UPLOAD_FOLDER']
    total = 0
    for f in os.listdir(image_dir):
        if allowed_file(f):
            total += 1
    done = len(annotations_store)
    return jsonify({
        'total': total,
        'done': done,
        'remaining': total - done
    })


# ========== 导出 API ==========

def export_yolo_txt(filename, annotation, output_dir):
    """导出为 YOLO TXT 格式"""
    txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
    lines = []
    for box in annotation['boxes']:
        # YOLO格式: class_id x_center y_center width height (归一化)
        x_center = (box['x1'] + box['x2']) / 2 / annotation['img_w']
        y_center = (box['y1'] + box['y2']) / 2 / annotation['img_h']
        width = (box['x2'] - box['x1']) / annotation['img_w']
        height = (box['y2'] - box['y1']) / annotation['img_h']
        lines.append(f"{box['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_yolo_json(filename, annotation, output_dir):
    """导出为 YOLO JSON 格式"""
    data = {
        'version': '1.0',
        'filename': filename,
        'image_width': annotation['img_w'],
        'image_height': annotation['img_h'],
        'annotations': []
    }
    for box in annotation['boxes']:
        data['annotations'].append({
            'class_id': box['class_id'],
            'class_name': box['class_name'],
            'bbox': [box['x1'], box['y1'], box['x2'], box['y2']],
            'confidence': box['confidence']
        })
    json_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_coco_json(all_data, output_dir):
    """导出为 COCO JSON 格式（所有图片合并为一个文件）"""
    images = []
    annotations = []
    categories = {}
    ann_id = 1

    for filename, annotation in all_data.items():
        img_id = len(images) + 1
        images.append({
            'id': img_id,
            'file_name': filename,
            'width': annotation['img_w'],
            'height': annotation['img_h']
        })

        for box in annotation['boxes']:
            if box['class_name'] not in categories:
                cat_id = len(categories) + 1
                categories[box['class_name']] = {
                    'id': cat_id,
                    'name': box['class_name']
                }
            cat_id = categories[box['class_name']]['id']

            x = box['x1']
            y = box['y1']
            w = box['x2'] - box['x1']
            h = box['y2'] - box['y1']

            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'confidence': box['confidence']
            })
            ann_id += 1

    coco_data = {
        'images': images,
        'annotations': annotations,
        'categories': list(categories.values())
    }

    json_path = os.path.join(output_dir, 'coco_annotations.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)

    return json_path


def export_pascal_voc(filename, annotation, output_dir):
    """导出为 Pascal VOC XML 格式"""
    root = ET.Element('annotation')
    ET.SubElement(root, 'filename').text = filename

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(int(annotation['img_w']))
    ET.SubElement(size, 'height').text = str(int(annotation['img_h']))
    ET.SubElement(size, 'depth').text = '3'

    for box in annotation['boxes']:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = box['class_name']
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(box['x1']))
        ET.SubElement(bndbox, 'ymin').text = str(int(box['y1']))
        ET.SubElement(bndbox, 'xmax').text = str(int(box['x2']))
        ET.SubElement(bndbox, 'ymax').text = str(int(box['y2']))

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')
    xml_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.xml')
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)


@app.route('/api/export', methods=['POST'])
def export_annotations():
    """导出标注结果"""
    format_type = request.json.get('format', 'yolo_txt')
    if not annotations_store:
        return jsonify({'error': '没有标注数据可导出'}), 400

    # 创建导出目录
    export_dir = os.path.join(
        app.config['EXPORT_FOLDER'],
        f'export_{uuid.uuid4().hex[:8]}'
    )
    os.makedirs(export_dir, exist_ok=True)

    try:
        if format_type == 'yolo_txt':
            for filename, annotation in annotations_store.items():
                export_yolo_txt(filename, annotation, export_dir)
            return jsonify({
                'success': True,
                'path': export_dir,
                'count': len(annotations_store)
            })

        elif format_type == 'yolo_json':
            for filename, annotation in annotations_store.items():
                export_yolo_json(filename, annotation, export_dir)
            return jsonify({
                'success': True,
                'path': export_dir,
                'count': len(annotations_store)
            })

        elif format_type == 'coco_json':
            path = export_coco_json(annotations_store, export_dir)
            return jsonify({
                'success': True,
                'path': path,
                'count': len(annotations_store)
            })

        elif format_type == 'pascal_voc':
            for filename, annotation in annotations_store.items():
                export_pascal_voc(filename, annotation, export_dir)
            return jsonify({
                'success': True,
                'path': export_dir,
                'count': len(annotations_store)
            })

        else:
            return jsonify({'error': f'不支持的导出格式: {format_type}'}), 400

    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500


# ========== 关闭服务 ==========

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """关闭服务器（前端点击关闭按钮时调用）"""
    # 先返回响应，再退出
    shutdown_response = jsonify({'success': True, 'message': '服务已关闭'})

    # 延迟退出，确保响应先发送到浏览器
    def _delayed_exit():
        time.sleep(0.5)          # 等响应发送完毕
        # 如果存在 reloader 父进程（debug 模式），也杀掉它
        try:
            ppid = os.getppid()
            if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' and ppid > 1:
                os.kill(ppid, 9)  # 杀 reloader 父进程
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=False).start()

    return shutdown_response


# ========== 启动 ==========

if __name__ == '__main__':
    print("=" * 50)
    print("  YOLO 智能标注工具")
    print("  启动服务...")
    print("  打开浏览器访问: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='127.0.0.1', port=5000, debug=False)
