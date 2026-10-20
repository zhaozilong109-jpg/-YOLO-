# YOLO 智能标注工具

基于 **Flask + Ultralytics YOLO** 的浏览器端图像标注工具：把图片拖进来，模型先自动画框，你在网页上拖一拖改一改，最后一键导出**能直接拿去训练**的数据集（zip 直接下载）。

> v0.3：新增视频抽帧导入与内容去重；v0.2 补齐了「标注」这件事缺失的另一半 —— 人工修正、结果落盘、导出下载。

## v0.3 更新了什么

| 原问题 | 现在的行为 |
| --- | --- |
| 同一张图换个名字再传一次，会变成两份数据 | 按内容 sha256 去重，重复直接跳过并提示「与已有图片重复：xxx」；索引持久化，重启后依然生效 |
| 只能标图片，视频要自己先拆帧再一张张传 | 网页里直接选视频，按 0.5–10 秒抽帧，抽出的帧自动进图片列表；静止画面自动去重 |
| 大图/大视频把上传接口卡死 | 图片与视频分别限制体积（默认 50 MB / 512 MB），超限给出明确提示 |

## v0.2 更新了什么

| 原问题 | 现在的行为 |
| --- | --- |
| 标注只存内存，重启后「图片还在、标注全丢」 | 每张图一个 `annotations/<文件名>.json`，启动自动回填 |
| 中文文件名被 `secure_filename` 吃成 `jpg`，图片在列表里消失 | 自己实现 unicode 安全清洗，`猫.jpg` 原样保存、原样导出 |
| README 说能「人工修正」，但画布没绑任何鼠标事件 | 画布支持新建 / 拖动 / 八向缩放 / 删除 / 改类别 / 撤销重做 / 方向键微调，改动自动保存 |
| 导出只返回服务器路径，浏览器根本拿不到文件 | 服务端打包 zip，浏览器直接下载 |
| `delete_images` 可用 `../` 删掉任意文件 | 统一走 `resolve_within()` 校验，越界一律拒绝 |
| 全局模型无锁，多标签页并发推理会崩 | 推理全程持锁，批量标注改服务端单任务队列 |
| 批量标注每张都全目录扫描（O(n²)） | 任务开始一次性建列表，前端轮询进度，可随时取消 |
| 只能输出 COCO 的 80 类，做不了小类别数据集 | 类别过滤 + 「只导出有标注的类别」，自动生成 `data.yaml` |
| 导出的是散落的标注文件 | YOLO 格式自带 `classes.txt` / `data.yaml`，可选 train/val 划分和原图 |
| 大图缩在屏幕里，小目标没法修 | 滚轮以光标为中心缩放、空格/中键平移，「F」一键适应窗口 |

## 功能特性

- **自动预标注**：内置 YOLOv8n/s/m/l/x 与 YOLOv5su，也可以上传自己的 `.pt` 权重
- **视频抽帧**：直接选视频按固定间隔抽帧，静止画面自动跳过，不用先拿剪辑软件拆图
- **上传去重**：按内容哈希判重，同一张图重复上传只保留一份
- **可编辑标注**：拖拽移动、八向缩放、空白处框选新建、Delete 删除、下拉改类别、Ctrl+Z 撤销
- **类别过滤**：只输出关心的类别，避免把整张图的椅子、杯子都标出来
- **批量标注**：服务端任务队列，进度条 + 失败原因，可中途取消
- **多格式导出**：YOLO TXT / YOLO JSON / COCO JSON / Pascal VOC，全部打包成 zip
- **导出即可训练**：`data.yaml` + `classes.txt` + 自动划分 train/val，解压后 `yolo train` 就能跑
- **数据不丢**：标注实时落盘，关掉服务、重启电脑都还在
- **中文友好**：中文文件名、中文类别名全程正常

## 快速开始

### 环境要求

- Python 3.9+
- 首次加载预训练模型需要联网下载权重（约 6–130 MB，取决于所选模型）

### 安装与运行

```bash
git clone https://github.com/zhaozilong109-jpg/-YOLO-.git yolo-tool
cd yolo-tool
python -m pip install -r requirements.txt
python app.py
```

Windows 下也可以直接双击 `start.bat`：它会检查依赖、清理占用 5000 端口的旧进程、自动打开浏览器。

启动后访问 <http://127.0.0.1:5000>。

> 只看 CPU 也能跑：`yolov8n.pt` 在普通笔记本上一张图约 0.1–0.5 秒。

## 使用流程

1. **上传图片**：点「上传图片」或直接把文件/文件夹拖进画布区域
2. **导入视频**（可选）：点「视频抽帧」，选视频与抽帧间隔，抽出的帧会直接加进列表
3. **加载模型**：下拉选一个预训练模型，点「加载模型」
4. **自动标注**：`标注当前图` 只跑当前这张，`批量标注` 跑所有未标注的图（勾「重标全部」则全部重跑）
5. **人工修正**：拖框、拉控制点、空白处画新框、在右侧「标注」面板改类别
6. **导出**：点「导出」，选格式与选项，浏览器直接下载 zip

标注过程中不需要点保存：改动后 0.45 秒自动落盘，右下角会显示「已保存 / 保存中 / 保存失败」。

## 画布操作

| 操作 | 效果 |
| --- | --- |
| 在空白处拖动 | 新建检测框 |
| 拖动框内部 | 移动检测框 |
| 拖动 8 个控制点 | 缩放检测框 |
| `Delete` / `Backspace` | 删除选中的框 |
| `Ctrl+Z` / `Ctrl+Shift+Z` | 撤销 / 重做 |
| `Ctrl+S` | 立即保存 |
| 方向键（按住 `Shift` 加速） | 以像素为单位微调位置 |
| `1`–`9` | 给选中的框切换成第 N 个类别 |
| `空格` + 拖动 / 中键拖动 | 平移画布 |
| 滚轮 | 以光标为中心缩放 |
| `F` | 适应窗口 |
| `PageUp` / `PageDown` | 上一张 / 下一张 |
| `H` | 快捷键帮助 |

## 导出格式

| 格式 | 产出内容 | 说明 |
| --- | --- | --- |
| **YOLO TXT** | `labels/{train,val}/*.txt`、`images/`、`classes.txt`、`data.yaml`、`README.txt` | 归一化 `class cx cy w h`，解压即可训练 |
| **YOLO JSON** | `labels/*.json` + `classes.txt` | 每图一个 JSON，绝对坐标 `xyxy` |
| **COCO JSON** | `annotations/instances.json`、`images/`、`classes.txt` | 单文件全量，类别 id 稳定且从 1 连续编号 |
| **Pascal VOC** | `Annotations/*.xml`、`JPEGImages/`、`classes.txt` | 绝对坐标 `xmin/ymin/xmax/ymax` |

导出选项：

- **包含原始图片**：把原图一起打进 zip，避免图片和标注分家
- **只导出有标注的类别**：默认开启。只有 person 的数据集不会声明 80 个类别
- **划分验证集**：按文件名哈希确定性地切 train/val（默认 20%），多次导出结果一致

训练示例：

```bash
unzip yolo_txt_20261020_153000.zip -d dataset
cd dataset
yolo detect train data=data.yaml model=yolov8n.pt imgsz=640 epochs=100
```

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查（版本、模型、图片数） |
| POST | `/api/upload` | 上传图片（multipart，字段 `files`），自动去重 |
| POST | `/api/upload_video` | 视频抽帧（字段 `video`、`interval`、`max_frames`） |
| GET | `/api/images` | 图片列表与标注状态 |
| GET | `/uploads/<filename>` | 读取图片原文件 |
| POST | `/api/delete_images` | 删除指定图片（同时清标注） |
| POST | `/api/clear_all` | 清空全部图片与标注 |
| GET | `/api/models` | 预训练 + 自定义模型列表 |
| POST | `/api/upload_model` | 上传自定义 `.pt` |
| POST | `/api/load_model` | 加载模型，返回类别表 |
| GET | `/api/classes` | 当前模型的类别表 |
| POST | `/api/annotate` | 单图推理并落盘 |
| GET | `/api/annotation/<filename>` | 读取某图标注 |
| PUT | `/api/annotation/<filename>` | 保存人工修正后的标注 |
| POST | `/api/batch` | 创建批量标注任务 |
| GET | `/api/batch/<job_id>` | 查询任务进度 |
| POST | `/api/batch/<job_id>/cancel` | 取消任务 |
| GET | `/api/stats` | 图片 / 标注 / 框数量统计 |
| POST | `/api/export` | 导出数据集，返回 zip 附件 |
| GET | `/api/export/formats` | 支持的导出格式 |
| POST | `/api/shutdown` | 关闭本地服务 |

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `YOLO_LABELER_DATA` | 项目根目录 | 数据根目录（`uploads/`、`annotations/`、`models/`、`exports/` 都在其下） |
| `YOLO_LABELER_HOST` | `127.0.0.1` | 监听地址 |
| `YOLO_LABELER_PORT` | `5000` | 监听端口 |
| `YOLO_LABELER_MAX_IMAGE_MB` | `50` | 单张图片体积上限 |
| `YOLO_LABELER_MAX_VIDEO_MB` | `512` | 单个视频体积上限 |
| `YOLO_LABELER_DEDUPE` | `1` | 设为 `0` 可关闭按内容去重 |

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

覆盖文件名清洗、路径穿越、标注落盘与重启回填、四种导出格式的坐标换算、类别重编号、train/val 划分稳定性、内容去重、视频抽帧，以及全部 HTTP 接口（64 个用例，不需要 GPU 或模型权重）。

## 项目结构

```
yolo-tool/
├── app.py                  # Flask 应用工厂与全部路由
├── core/
│   ├── config.py           # 路径、上传限制、环境变量
│   ├── storage.py          # 文件名安全 + 标注持久化（AnnotationStore）
│   ├── index.py            # 内容哈希去重索引（UploadIndex）
│   ├── video.py            # 视频抽帧（OpenCV）
│   ├── inference.py        # 模型加载、单图推理、批量任务（BatchRunner）
│   └── exporter.py         # 四种格式导出 + zip 打包
├── templates/index.html    # 页面结构
├── static/
│   ├── css/style.css       # 深色主题样式
│   └── js/main.js          # Canvas 标注交互（无框架）
├── tests/                  # pytest 用例
├── requirements.txt
├── requirements-dev.txt
├── start.bat               # Windows 一键启动
├── uploads/                # 上传的图片（运行时生成）
├── annotations/            # 标注 JSON（运行时生成）
├── models/                 # 模型权重（运行时下载/上传）
├── upload_index.json       # 上传去重索引（运行时生成）
└── exports/                # 导出临时目录（运行时生成）
```

## 已知限制

- 只支持矩形框（bbox），多边形/分割标注尚未实现
- 单进程、单任务：批量标注同时只能跑一个，适合个人或小团队本地使用
- 没有鉴权，默认只监听 `127.0.0.1`，**不要直接暴露到公网**
- 去重只比较完全相同的字节；同一张图另存为不同压缩率仍会各存一份

## 后续计划

- YOLOv8-seg 掩膜 / 多边形标注
- SAM 辅助：点一下目标自动生成精确轮廓
- 主动学习：按置信度排序，优先人工复核低分框
- 感知哈希去重（识别「同一张图不同压缩率」的近似重复）
- ONNX / 半精度推理，CPU 上再快一截
- Dockerfile 与 PyInstaller 打包

## License

[MIT](LICENSE)
