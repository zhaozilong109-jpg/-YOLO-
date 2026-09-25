# YOLO 智能标注工具

基于 **Flask + Ultralytics YOLO** 的浏览器端图像标注工具：上传图片后，用预训练 YOLO 模型一键自动生成标注框，人工微调后导出为多种数据集格式。

## 功能特性

- **图片管理**：批量上传、浏览、删除图片
- **AI 预标注**：内置 YOLOv8n / YOLOv8s / YOLOv8m / YOLOv5su 预训练模型，支持单张与批量自动标注（模型首次使用时由 ultralytics 自动下载）
- **自定义模型**：上传自己的 `.pt` 权重文件即可使用
- **人工修正**：在网页上调整检测框的位置、类别与置信度
- **多格式导出**：
  - YOLO TXT（归一化坐标）
  - YOLO JSON
  - COCO JSON（全图合并）
  - Pascal VOC XML

## 快速开始

### 环境要求

- Python 3.9+

### 安装

```bash
git clone https://github.com/<你的用户名>/yolo-tool.git
cd yolo-tool
pip install -r requirements.txt
```

### 运行

```bash
python app.py
```

Windows 下也可直接双击 `start.bat`（会先清理占用 5000 端口的旧进程）。

启动后浏览器访问：

```
http://127.0.0.1:5000
```

## 使用流程

1. **上传图片** —— 支持多选批量上传
2. **选择模型** —— 从模型列表选择预训练模型，或上传自定义 `.pt`
3. **自动标注** —— 单张标注或批量标注全部图片
4. **人工微调** —— 拖动调整检测框、修改类别
5. **导出数据集** —— 选择格式导出到 `exports/` 目录

## 项目结构

```
yolo-tool/
├── app.py              # Flask 后端（全部 API 路由）
├── requirements.txt    # Python 依赖
├── start.bat           # Windows 一键启动
├── templates/
│   └── index.html      # 页面模板
├── static/
│   ├── css/style.css   # 样式
│   └── js/main.js      # 前端逻辑（标注画布交互）
├── uploads/            # 上传的图片（运行时生成）
├── annotations/        # 标注数据（运行时生成）
├── models/             # 模型权重（运行时下载/上传）
└── exports/            # 导出结果（运行时生成）
```

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/upload` | 上传图片 |
| GET | `/api/images` | 图片列表 |
| POST | `/api/delete_images` | 删除图片 |
| POST | `/api/clear_all` | 清空全部图片与标注 |
| GET | `/api/models` | 可用模型列表 |
| POST | `/api/upload_model` | 上传自定义模型 |
| POST | `/api/load_model` | 加载模型 |
| POST | `/api/annotate` | 单张自动标注 |
| POST | `/api/annotate_batch` | 批量自动标注 |
| GET | `/api/annotation/<filename>` | 获取单张标注 |
| GET | `/api/annotations_count` | 标注数量统计 |
| POST | `/api/export` | 导出数据集 |
| POST | `/api/shutdown` | 关闭服务 |

## 技术栈

- **后端**：Flask、Ultralytics YOLO、OpenCV、Pillow、NumPy
- **前端**：原生 HTML / CSS / JavaScript（Canvas 标注交互）

## License

[MIT](LICENSE)
