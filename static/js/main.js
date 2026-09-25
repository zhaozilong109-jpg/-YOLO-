/**
 * YOLO 智能标注工具 - 前端逻辑
 * 纯 JavaScript，无需任何前端框架
 */

// ========== 状态管理 ==========
const state = {
    images: [],            // 所有图片列表
    currentImage: null,    // 当前选中的图片文件名
    annotations: {},       // 所有标注结果 { filename: { boxes: [...], img_w, img_h } }
    isBatchRunning: false, // 是否正在批量标注
    cancelBatch: false,    // 是否取消批量标注
    modelLoaded: false,    // 模型是否已加载
};

// ========== DOM 引用 ==========
const $ = (id) => document.getElementById(id);
const canvas = $('imageCanvas');
const ctx = canvas.getContext('2d');

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadImageList();

    // 绑定事件
    setupEventListeners();
});

// ========== 事件绑定 ==========
function setupEventListeners() {
    // 置信度滑块
    $('confidenceSlider').addEventListener('input', (e) => {
        $('confidenceVal').textContent = parseFloat(e.target.value).toFixed(2);
    });

    // IoU 滑块
    $('iouSlider').addEventListener('input', (e) => {
        $('iouVal').textContent = parseFloat(e.target.value).toFixed(2);
    });

    // 加载模型按钮
    $('loadModelBtn').addEventListener('click', loadModel);

    // 上传图片按钮
    $('uploadBtn').addEventListener('click', () => $('fileInput').click());
    $('fileInput').addEventListener('change', uploadImages);

    // 批量标注按钮
    $('batchAnnotateBtn').addEventListener('click', startBatchAnnotate);

    // 取消批量标注
    $('cancelBatchBtn').addEventListener('click', () => {
        state.cancelBatch = true;
        showToast('正在取消...', 'info');
    });

    // 导出按钮
    $('exportBtn').addEventListener('click', () => {
        $('export-modal').classList.remove('hidden');
    });

    // 导出格式选择
    document.querySelectorAll('.export-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const format = btn.dataset.format;
            $('export-modal').classList.add('hidden');
            exportAnnotations(format);
        });
    });

    // 清除所有图片
    $('clearAllBtn').addEventListener('click', clearAllImages);

    // 关闭导出弹窗
    $('closeExportModal').addEventListener('click', () => {
        $('export-modal').classList.add('hidden');
    });

    // 关闭服务按钮
    $('shutdownBtn').addEventListener('click', shutdownServer);
}

// ========== 模型管理 ==========

/**
 * 获取可用模型列表
 */
function loadModels() {
    fetch('/api/models')
        .then(res => res.json())
        .then(data => {
            const select = $('modelSelect');
            select.innerHTML = '';

            // 预训练模型分组
            if (data.pretrained && data.pretrained.length > 0) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = '预训练模型';
                data.pretrained.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.label;
                    optgroup.appendChild(opt);
                });
                select.appendChild(optgroup);
            }

            // 自定义模型分组
            if (data.custom && data.custom.length > 0) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = '自定义模型';
                data.custom.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.label;
                    optgroup.appendChild(opt);
                });
                select.appendChild(optgroup);
            }

            // 添加"上传自定义模型"选项
            const uploadOpt = document.createElement('option');
            uploadOpt.value = '__upload__';
            uploadOpt.textContent = '📤 上传自定义模型...';
            select.appendChild(uploadOpt);

            // 监听自定义模型上传
            select.addEventListener('change', (e) => {
                if (e.target.value === '__upload__') {
                    $('modelFileInput').click();
                }
            });

            $('modelFileInput').addEventListener('change', uploadModelFile);
        })
        .catch(err => showToast('获取模型列表失败: ' + err.message, 'error'));
}

/**
 * 上传自定义模型文件
 */
function uploadModelFile(e) {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('model', file);

    fetch('/api/upload_model', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showToast(`模型 "${data.name}" 上传成功！`, 'success');
                loadModels(); // 刷新列表
                // 自动选中刚上传的模型
                setTimeout(() => {
                    const select = $('modelSelect');
                    for (let opt of select.options) {
                        if (opt.value === data.name) {
                            select.value = data.name;
                            break;
                        }
                    }
                }, 100);
            } else {
                showToast('上传失败: ' + (data.error || '未知错误'), 'error');
            }
        })
        .catch(err => showToast('上传失败: ' + err.message, 'error'));

    e.target.value = ''; // 清空以便重复上传
}

/**
 * 加载选中的模型
 */
function loadModel() {
    const modelName = $('modelSelect').value;
    if (!modelName || modelName === '__upload__') {
        showToast('请先选择一个模型', 'error');
        return;
    }

    $('loadModelBtn').disabled = true;
    $('loadModelBtn').textContent = '加载中...';

    fetch('/api/load_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                state.modelLoaded = true;
                showToast(`✅ 模型 "${modelName}" 加载成功！`, 'success');
                if (data.classes) {
                    console.log('模型类别:', data.classes);
                }
            } else {
                showToast('❌ ' + (data.error || '加载失败'), 'error');
                state.modelLoaded = false;
            }
        })
        .catch(err => {
            showToast('❌ 加载失败: ' + err.message, 'error');
            state.modelLoaded = false;
        })
        .finally(() => {
            $('loadModelBtn').disabled = false;
            $('loadModelBtn').textContent = '加载模型';
        });
}

/**
 * 关闭服务器
 */
function shutdownServer() {
    if (!confirm('确定要关闭服务吗？关闭后需重新启动才能继续使用。')) return;

    const btn = $('shutdownBtn');
    btn.disabled = true;
    btn.textContent = '关闭中...';

    fetch('/api/shutdown', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            showToast('✅ 服务已关闭，可以关闭浏览器了', 'success');
            btn.textContent = '已关闭';
            // 禁用所有操作按钮
            $('loadModelBtn').disabled = true;
            $('uploadBtn').disabled = true;
            $('batchAnnotateBtn').disabled = true;
            $('exportBtn').disabled = true;
        })
        .catch(() => {
            // 服务关闭后 fetch 会失败，这是正常的
            showToast('✅ 服务已关闭，可以关闭浏览器了', 'success');
            btn.textContent = '已关闭';
            $('loadModelBtn').disabled = true;
            $('uploadBtn').disabled = true;
            $('batchAnnotateBtn').disabled = true;
            $('exportBtn').disabled = true;
        });
}

// ========== 图片管理 ==========

/**
 * 上传图片
 */
function uploadImages(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (let file of files) {
        formData.append('files', file);
    }

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showToast(`成功上传 ${data.count} 张图片`, 'success');
                loadImageList();
            } else {
                showToast('上传失败: ' + (data.error || '未知错误'), 'error');
            }
        })
        .catch(err => showToast('上传失败: ' + err.message, 'error'));

    e.target.value = ''; // 清空 input
}

/**
 * 删除单张图片
 */
function deleteImage(filename) {
    fetch('/api/delete_images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filenames: [filename] })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                // 如果删的是当前显示的图片，清空画布
                if (state.currentImage === filename) {
                    clearCanvas();
                    state.currentImage = null;
                    $('currentImageName').textContent = '未选择图片';
                    $('detectionCount').textContent = '';
                }
                showToast(`已移除: ${filename}`, 'info');
                loadImageList();
            }
        })
        .catch(err => showToast('删除失败: ' + err.message, 'error'));
}

/**
 * 清除所有图片
 */
function clearAllImages() {
    const count = state.images.length;
    if (count === 0) {
        showToast('没有图片可清除', 'info');
        return;
    }

    fetch('/api/clear_all', {
        method: 'POST'
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                clearCanvas();
                state.currentImage = null;
                state.annotations = {};
                $('currentImageName').textContent = '未选择图片';
                $('detectionCount').textContent = '';
                showToast(`已清除全部 ${count} 张图片`, 'info');
                loadImageList();
            }
        })
        .catch(err => showToast('清除失败: ' + err.message, 'error'));
}

/**
 * 加载图片列表（右侧边栏）
 */
function loadImageList() {
    fetch('/api/images')
        .then(res => res.json())
        .then(data => {
            state.images = data.images;
            renderImageList();
            updateProgressLabel();

            // 如果有图片且没有选中任何图片，自动选中第一张
            if (state.images.length > 0 && !state.currentImage) {
                selectImage(state.images[0].filename);
            } else if (state.images.length === 0) {
                // 没有图片，清空画布
                clearCanvas();
                state.currentImage = null;
                $('currentImageName').textContent = '未选择图片';
                $('detectionCount').textContent = '';
            }
        })
        .catch(err => showToast('加载图片列表失败', 'error'));
}

/**
 * 渲染图片列表
 */
function renderImageList() {
    const container = $('image-list');
    container.innerHTML = '';

    state.images.forEach(img => {
        const item = document.createElement('div');
        item.className = 'image-item';
        if (img.filename === state.currentImage) {
            item.classList.add('active');
        }

        // 删除按钮 ×
        const delBtn = document.createElement('button');
        delBtn.className = 'delete-btn';
        delBtn.textContent = '×';
        delBtn.title = '移除这张图片';

        // 缩略图
        const thumb = document.createElement('img');
        thumb.className = 'thumb';
        thumb.src = img.url;
        thumb.alt = img.filename;

        // 信息
        const info = document.createElement('div');
        info.className = 'item-info';

        const nameSpan = document.createElement('div');
        nameSpan.className = 'item-name';
        nameSpan.textContent = img.filename;

        const statusSpan = document.createElement('div');
        statusSpan.className = 'item-status';
        if (img.status === 'annotated') {
            statusSpan.className = 'item-status status-annotated';
            statusSpan.textContent = '✅ 已标注';
        } else {
            statusSpan.className = 'item-status status-pending';
            statusSpan.textContent = '⏳ 待标注';
        }

        info.appendChild(nameSpan);
        info.appendChild(statusSpan);
        item.appendChild(delBtn);
        item.appendChild(thumb);
        item.appendChild(info);

        // 点击删除按钮移除图片
        delBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteImage(img.filename);
        });

        // 点击选择图片
        item.addEventListener('click', () => {
            selectImage(img.filename);
        });

        container.appendChild(item);
    });

    // 更新列表计数
    $('listCount').textContent = `${state.images.length} 张`;
}

/**
 * 更新顶部进度标签
 */
function updateProgressLabel() {
    const total = state.images.length;
    const done = state.images.filter(img => img.status === 'annotated').length;
    if (total === 0) {
        $('progressLabel').textContent = '未加载图片';
    } else if (done === total) {
        $('progressLabel').textContent = `✅ 全部完成 (${total}张)`;
    } else {
        $('progressLabel').textContent = `已标注 ${done} 张，剩余 ${total - done} 张`;
    }
}

// ========== 图片选择与显示 ==========

/**
 * 选择并显示某张图片
 */
function selectImage(filename) {
    state.currentImage = filename;
    $('currentImageName').textContent = filename;

    // 更新列表高亮
    document.querySelectorAll('.image-item').forEach(el => {
        el.classList.remove('active');
        const nameEl = el.querySelector('.item-name');
        if (nameEl && nameEl.textContent === filename) {
            el.classList.add('active');
            // 滚动到可见区域
            el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    });

    // 显示图片
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        // 获取标注结果并绘制
        fetch(`/api/annotation/${encodeURIComponent(filename)}`)
            .then(res => res.json())
            .then(data => {
                state.annotations[filename] = data;
                drawCanvas(img, data);
                updateImageStatus(filename, data.boxes && data.boxes.length > 0);
            });
    };
    img.onerror = () => {
        showToast('图片加载失败: ' + filename, 'error');
    };
    img.src = `/uploads/${encodeURIComponent(filename)}`;
}

/**
 * 更新单张图片的标注状态
 */
function updateImageStatus(filename, hasBoxes) {
    // 更新 state.images 中的状态
    const imgObj = state.images.find(i => i.filename === filename);
    if (imgObj) {
        imgObj.status = hasBoxes ? 'annotated' : 'pending';
    }

    // 重新渲染列表（保持滚动位置）
    const scrollTop = $('image-list').scrollTop;
    renderImageList();
    $('image-list').scrollTop = scrollTop;

    updateProgressLabel();
}

/**
 * 在 Canvas 上绘制图片和检测框
 */
function drawCanvas(img, annotation) {
    // 设置 canvas 尺寸（限制最大显示尺寸）
    const container = $('image-container');
    const maxW = container.clientWidth - 20;
    const maxH = container.clientHeight - 20;

    // 计算缩放比例
    const scale = Math.min(maxW / img.naturalWidth, maxH / img.naturalHeight, 1);
    const displayW = img.naturalWidth * scale;
    const displayH = img.naturalHeight * scale;

    canvas.width = displayW;
    canvas.height = displayH;
    canvas.style.width = displayW + 'px';
    canvas.style.height = displayH + 'px';

    // 绘制图片
    ctx.drawImage(img, 0, 0, displayW, displayH);

    // 绘制检测框
    const boxes = annotation.boxes || [];
    if (boxes.length === 0) {
        $('detectionCount').textContent = '未检测到目标';
        return;
    }

    $('detectionCount').textContent = `检测到 ${boxes.length} 个目标`;

    // 缩放比例（原始图片尺寸 -> canvas 显示尺寸）
    const sx = displayW / annotation.img_w;
    const sy = displayH / annotation.img_h;

    // 预定义颜色（最多支持20个类别）
    const colors = [
        '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
        '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
        '#2980b9', '#27ae60', '#8e44ad', '#d35400', '#2c3e50',
        '#f1c40f', '#e91e63', '#00bcd4', '#ff5722', '#607d8b'
    ];

    boxes.forEach((box, index) => {
        const x = box.x1 * sx;
        const y = box.y1 * sy;
        const w = (box.x2 - box.x1) * sx;
        const h = (box.y2 - box.y1) * sy;

        const color = colors[box.class_id % colors.length];

        // 画边框
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // 画标签背景
        const label = `${box.class_name} ${(box.confidence * 100).toFixed(1)}%`;
        ctx.font = '13px sans-serif';
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - 20, textW + 8, 20);

        // 画标签文字
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x + 4, y - 6);
    });
}

/**
 * 清空画布
 */
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = 0;
    canvas.height = 0;
}

// ========== 标注功能 ==========

/**
 * 批量标注
 */
function startBatchAnnotate() {
    if (!state.modelLoaded) {
        showToast('请先加载模型！', 'error');
        return;
    }

    const pendingImages = state.images.filter(img => img.status === 'pending');
    if (state.images.length === 0) {
        showToast('请先上传图片！', 'error');
        return;
    }
    if (pendingImages.length === 0) {
        showToast('所有图片已标注完成', 'info');
        return;
    }

    // 显示进度弹窗
    state.isBatchRunning = true;
    state.cancelBatch = false;
    $('batch-progress-overlay').classList.remove('hidden');
    $('doneCount').textContent = '0';
    $('remainCount').textContent = pendingImages.length;
    $('progress-fill').style.width = '0%';

    // 开始逐张标注
    batchAnnotateNext();
}

/**
 * 批量标注下一张
 */
function batchAnnotateNext() {
    if (state.cancelBatch) {
        state.isBatchRunning = false;
        $('batch-progress-overlay').classList.add('hidden');
        showToast('已取消批量标注', 'info');
        return;
    }

    const confidence = parseFloat($('confidenceSlider').value);
    const iou = parseFloat($('iouSlider').value);

    fetch('/api/annotate_batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confidence, iou })
    })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                showToast('标注出错: ' + data.error, 'error');
                state.isBatchRunning = false;
                $('batch-progress-overlay').classList.add('hidden');
                return;
            }

            // 更新进度
            const done = data.done || 0;
            const remaining = data.remaining || 0;
            const total = done + remaining;

            $('doneCount').textContent = done;
            $('remainCount').textContent = remaining;
            const pct = total > 0 ? (done / total * 100) : 100;
            $('progress-fill').style.width = pct.toFixed(1) + '%';

            // 如果当前显示的是刚标注的图片，更新画布
            if (data.filename === state.currentImage) {
                const img = new Image();
                img.onload = () => drawCanvas(img, data);
                img.src = `/uploads/${encodeURIComponent(data.filename)}`;
            }

            // 更新列表
            loadImageList();

            if (data.completed) {
                // 全部完成
                state.isBatchRunning = false;
                $('batch-progress-overlay').classList.add('hidden');
                showToast(`✅ 批量标注完成！共标注 ${done} 张图片`, 'success');
            } else {
                // 延迟一点再标注下一张，让 UI 有时间更新
                setTimeout(batchAnnotateNext, 50);
            }
        })
        .catch(err => {
            showToast('批量标注出错: ' + err.message, 'error');
            state.isBatchRunning = false;
            $('batch-progress-overlay').classList.add('hidden');
        });
}

// ========== 导出功能 ==========

/**
 * 导出标注结果
 */
function exportAnnotations(format) {
    const formatNames = {
        'yolo_txt': 'YOLO TXT',
        'yolo_json': 'YOLO JSON',
        'coco_json': 'COCO JSON',
        'pascal_voc': 'Pascal VOC'
    };

    showToast(`正在导出 ${formatNames[format]} 格式...`, 'info');

    fetch('/api/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showToast(`✅ 导出成功！共 ${data.count} 张图片`, 'success');
                console.log('导出路径:', data.path);
            } else {
                showToast('❌ 导出失败: ' + (data.error || '未知错误'), 'error');
            }
        })
        .catch(err => showToast('导出失败: ' + err.message, 'error'));
}

// ========== Toast 消息提示 ==========

/**
 * 显示 Toast 消息
 */
function showToast(message, type = 'success') {
    // 移除旧 toast
    const old = document.querySelector('.toast');
    if (old) old.remove();

    const toast = document.createElement('div');
    toast.className = 'toast ' + (type || '');
    toast.textContent = message;
    document.body.appendChild(toast);

    // 3 秒后自动消失
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
