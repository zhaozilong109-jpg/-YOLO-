/**
 * YOLO 智能标注工具 —— 前端逻辑
 *
 * 标注坐标始终以「原图像素」存储，画布只是它的一个缩放视图，
 * 这样缩放、平移、改窗口大小都不会污染数据。
 */
(() => {
    'use strict';

    // ---------------------------------------------------------------- 常量

    const PALETTE = [
        '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7',
        '#06b6d4', '#f97316', '#64748b', '#14b8a6', '#e11d48',
        '#8b5cf6', '#84cc16', '#0ea5e9', '#d946ef', '#eab308',
        '#10b981', '#f43f5e', '#6366f1', '#a3e635', '#fb7185',
    ];

    const MIN_BOX_PX = 3;          // 小于这个尺寸的框不创建（原图像素）
    const HANDLE_SIZE = 8;         // 控制点边长（屏幕像素）
    const VERTEX_HIT = 7;          // 多边形顶点的命中半径
    const CLOSE_HIT = 10;          // 点回首点闭合多边形的命中半径
    const HISTORY_LIMIT = 100;
    const SAVE_DELAY = 450;

    const FORMATS = [
        { value: 'yolo_txt', label: 'YOLO TXT', desc: 'labels/ + data.yaml，可直接训练' },
        { value: 'yolo_seg', label: 'YOLO SEG', desc: '分割标注，多边形顶点归一化' },
        { value: 'yolo_json', label: 'YOLO JSON', desc: '每图一个 JSON，绝对坐标' },
        { value: 'coco_json', label: 'COCO JSON', desc: '带 segmentation 的通用格式' },
        { value: 'pascal_voc', label: 'Pascal VOC', desc: '每图一个 XML，绝对坐标' },
    ];

    // ---------------------------------------------------------------- 状态

    const state = {
        images: [],
        current: null,
        annotation: { boxes: [], img_w: 0, img_h: 0 },
        classes: {},
        modelLoaded: false,
        modelName: null,
        selected: -1,
        image: null,
        view: { scale: 1, tx: 0, ty: 0 },
        width: 1,
        height: 1,
        dpr: 1,
        drag: null,
        history: [],
        future: [],
        saveTimer: null,
        classFilter: new Set(),
        onlyPending: false,
        batchId: null,
        batchTimer: null,
        space: false,
        userZoomed: false,
        exportFormat: 'yolo_txt',
        tool: 'box',
        draft: null,
        hoverPoint: null,
        samReady: false,
        samBusy: false,
        loadToken: 0,
    };

    const $ = (id) => document.getElementById(id);
    let canvas;
    let ctx;
    let viewport;

    // ---------------------------------------------------------------- 通用工具

    function clamp(value, low, high) {
        return Math.max(low, Math.min(high, value));
    }

    function colorFor(classId) {
        return PALETTE[Math.abs(Number(classId) || 0) % PALETTE.length];
    }

    function classNameOf(classId) {
        return state.classes[String(classId)] || `class_${classId}`;
    }

    // 画面比例一律以图片自身的像素尺寸为准；标注里的 img_w/img_h 只作兜底，
    // 否则一旦两者不一致（EXIF 旋转、旧标注文件），图片就会被拉伸。
    function imageWidth() {
        if (state.image && state.image.naturalWidth) {
            return state.image.naturalWidth;
        }
        return state.annotation.img_w || 0;
    }

    function imageHeight() {
        if (state.image && state.image.naturalHeight) {
            return state.image.naturalHeight;
        }
        return state.annotation.img_h || 0;
    }

    function toast(message, type = 'success', timeout = 3200) {
        const host = $('toastHost');
        const node = document.createElement('div');
        node.className = `toast ${type}`;
        node.textContent = message;
        host.appendChild(node);
        setTimeout(() => {
            node.style.opacity = '0';
            node.style.transition = 'opacity .25s';
            setTimeout(() => node.remove(), 260);
        }, timeout);
    }

    async function api(url, options = {}) {
        const response = await fetch(url, options);
        const type = response.headers.get('Content-Type') || '';
        let payload = null;
        if (type.includes('application/json')) {
            payload = await response.json().catch(() => null);
        }
        if (!response.ok) {
            const message = (payload && (payload.error || payload.message)) ||
                `请求失败（HTTP ${response.status}）`;
            const error = new Error(message);
            error.payload = payload;
            error.status = response.status;
            throw error;
        }
        return payload;
    }

    function jsonRequest(body) {
        return {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        };
    }

    function setSaveState(text, kind) {
        const node = $('saveState');
        node.textContent = text;
        node.className = `save-state ${kind || ''}`.trim();
    }

    // ---------------------------------------------------------------- 画布几何

    function imgToScreen(x, y) {
        return {
            x: x * state.view.scale + state.view.tx,
            y: y * state.view.scale + state.view.ty,
        };
    }

    function screenToImg(x, y) {
        return {
            x: (x - state.view.tx) / state.view.scale,
            y: (y - state.view.ty) / state.view.scale,
        };
    }

    function resizeCanvas() {
        const rect = viewport.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        state.width = Math.max(1, Math.round(rect.width));
        state.height = Math.max(1, Math.round(rect.height));
        state.dpr = dpr;
        canvas.width = Math.round(state.width * dpr);
        canvas.height = Math.round(state.height * dpr);
        canvas.style.width = `${state.width}px`;
        canvas.style.height = `${state.height}px`;
        draw();
    }

    function fitView() {
        if (!state.image) {
            return;
        }
        const iw = imageWidth();
        const ih = imageHeight();
        if (!iw || !ih) {
            return;
        }
        const pad = 26;
        const scale = Math.min(
            (state.width - pad * 2) / iw,
            (state.height - pad * 2) / ih
        );
        state.view.scale = clamp(scale, 0.01, 40);
        state.view.tx = (state.width - iw * state.view.scale) / 2;
        state.view.ty = (state.height - ih * state.view.scale) / 2;
        state.userZoomed = false;
        draw();
        updateZoomLabel();
    }

    function updateZoomLabel() {
        $('zoomLabel').textContent = `${Math.round(state.view.scale * 100)}%`;
    }

    // ---------------------------------------------------------------- 绘制

    function draw() {
        ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
        ctx.clearRect(0, 0, state.width, state.height);
        if (!state.image) {
            return;
        }

        const iw = imageWidth();
        const ih = imageHeight();
        const origin = imgToScreen(0, 0);

        ctx.imageSmoothingEnabled = state.view.scale < 3;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(
            state.image,
            origin.x,
            origin.y,
            iw * state.view.scale,
            ih * state.view.scale
        );

        state.annotation.boxes.forEach((box, index) => drawBox(box, index === state.selected));

        if (state.drag && state.drag.type === 'new' && state.drag.temp) {
            drawBox(normaliseBox(state.drag.temp), true, true);
        }

        drawDraft();
    }

    function normaliseBox(box) {
        return {
            ...box,
            x1: Math.min(box.x1, box.x2),
            y1: Math.min(box.y1, box.y2),
            x2: Math.max(box.x1, box.x2),
            y2: Math.max(box.y1, box.y2),
        };
    }

    function isPolygon(shape) {
        return Array.isArray(shape.points) && shape.points.length >= 3;
    }

    function boundsOf(points) {
        const xs = points.map((point) => point[0]);
        const ys = points.map((point) => point[1]);
        return {
            x1: Math.min(...xs),
            y1: Math.min(...ys),
            x2: Math.max(...xs),
            y2: Math.max(...ys),
        };
    }

    function pointInPolygon(points, x, y) {
        let inside = false;
        for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
            const [xi, yi] = points[i];
            const [xj, yj] = points[j];
            const crosses = yi > y !== yj > y;
            if (crosses && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
                inside = !inside;
            }
        }
        return inside;
    }

    function drawBox(box, selected, provisional = false) {
        if (isPolygon(box)) {
            drawPolygon(box, selected, provisional);
            return;
        }

        const rect = normaliseBox(box);
        const topLeft = imgToScreen(rect.x1, rect.y1);
        const width = (rect.x2 - rect.x1) * state.view.scale;
        const height = (rect.y2 - rect.y1) * state.view.scale;
        if (width < 0.4 || height < 0.4) {
            return;
        }

        ctx.save();
        ctx.lineWidth = selected ? 2.5 : 1.8;
        ctx.strokeStyle = colorFor(box.class_id);
        ctx.setLineDash(provisional ? [5, 4] : []);
        ctx.strokeRect(topLeft.x, topLeft.y, width, height);
        ctx.setLineDash([]);

        drawLabel(box, rect, provisional);
        if (selected) {
            drawHandles(handlePositions(rect));
        }
        ctx.restore();
    }

    function drawPolygon(box, selected, provisional = false) {
        const screen = box.points.map(([x, y]) => imgToScreen(x, y));
        const color = colorFor(box.class_id);

        ctx.save();
        ctx.beginPath();
        screen.forEach((point, index) => {
            if (index === 0) {
                ctx.moveTo(point.x, point.y);
            } else {
                ctx.lineTo(point.x, point.y);
            }
        });
        ctx.closePath();

        ctx.globalAlpha = provisional ? 0.12 : 0.2;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1;

        ctx.lineWidth = selected ? 2.5 : 1.8;
        ctx.strokeStyle = color;
        ctx.setLineDash(provisional ? [5, 4] : []);
        ctx.stroke();
        ctx.setLineDash([]);

        drawLabel(box, normaliseBox(box), provisional);

        if (selected) {
            screen.forEach((point, index) => {
                const size = index === 0 ? HANDLE_SIZE + 2 : HANDLE_SIZE - 1;
                ctx.fillStyle = index === 0 ? '#ffffff' : '#0f1115';
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.2;
                ctx.fillRect(point.x - size / 2, point.y - size / 2, size, size);
                ctx.strokeRect(point.x - size / 2, point.y - size / 2, size, size);
            });
        }
        ctx.restore();
    }

    function drawLabel(box, rect, provisional = false) {
        const topLeft = imgToScreen(rect.x1, rect.y1);
        const label = `${box.class_name || classNameOf(box.class_id)}${
            box.confidence < 0.999 ? ` ${(box.confidence * 100).toFixed(0)}%` : ''
        }`;

        ctx.font = '12px "Segoe UI", "Microsoft YaHei", sans-serif';
        const textWidth = ctx.measureText(label).width;
        const chipHeight = 18;
        const above = topLeft.y >= chipHeight;
        const chipY = above ? topLeft.y - chipHeight : topLeft.y;

        ctx.fillStyle = colorFor(box.class_id);
        ctx.globalAlpha = provisional ? 0.7 : 0.92;
        ctx.fillRect(topLeft.x, chipY, textWidth + 10, chipHeight);
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#fff';
        ctx.fillText(label, topLeft.x + 5, chipY + 13);
    }

    function drawHandles(points) {
        ctx.fillStyle = '#0f1115';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        points.forEach((point) => {
            ctx.fillRect(
                point.x - HANDLE_SIZE / 2,
                point.y - HANDLE_SIZE / 2,
                HANDLE_SIZE,
                HANDLE_SIZE
            );
            ctx.strokeRect(
                point.x - HANDLE_SIZE / 2,
                point.y - HANDLE_SIZE / 2,
                HANDLE_SIZE,
                HANDLE_SIZE
            );
        });
    }

    function drawDraft() {
        const draft = state.draft;
        if (!draft || !draft.points.length) {
            return;
        }
        const screen = draft.points.map(([x, y]) => imgToScreen(x, y));
        const preview = state.hoverPoint ? imgToScreen(state.hoverPoint.x, state.hoverPoint.y) : null;

        ctx.save();
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 1.8;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        screen.forEach((point, index) => {
            if (index === 0) {
                ctx.moveTo(point.x, point.y);
            } else {
                ctx.lineTo(point.x, point.y);
            }
        });
        if (preview) {
            ctx.lineTo(preview.x, preview.y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        if (screen.length >= 3 && preview) {
            ctx.globalAlpha = 0.35;
            ctx.strokeStyle = '#22d3ee';
            ctx.beginPath();
            ctx.moveTo(screen[0].x, screen[0].y);
            ctx.lineTo(preview.x, preview.y);
            ctx.stroke();
            ctx.globalAlpha = 1;
        }

        screen.forEach((point, index) => {
            const size = index === 0 ? HANDLE_SIZE + 2 : HANDLE_SIZE - 1;
            ctx.fillStyle = index === 0 ? '#22d3ee' : '#0f1115';
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.2;
            ctx.fillRect(point.x - size / 2, point.y - size / 2, size, size);
            ctx.strokeRect(point.x - size / 2, point.y - size / 2, size, size);
        });
        ctx.restore();
    }

    function handlePositions(box) {
        const left = imgToScreen(box.x1, box.y1);
        const right = imgToScreen(box.x2, box.y2);
        const midX = (left.x + right.x) / 2;
        const midY = (left.y + right.y) / 2;
        return [
            { x: left.x, y: left.y },
            { x: midX, y: left.y },
            { x: right.x, y: left.y },
            { x: right.x, y: midY },
            { x: right.x, y: right.y },
            { x: midX, y: right.y },
            { x: left.x, y: right.y },
            { x: left.x, y: midY },
        ];
    }

    function hitHandle(box, sx, sy) {
        if (isPolygon(box)) {
            return -1;
        }
        const points = handlePositions(normaliseBox(box));
        for (let i = 0; i < points.length; i += 1) {
            if (
                Math.abs(points[i].x - sx) <= HANDLE_SIZE &&
                Math.abs(points[i].y - sy) <= HANDLE_SIZE
            ) {
                return i;
            }
        }
        return -1;
    }

    function hitVertex(shape, sx, sy) {
        if (!isPolygon(shape)) {
            return -1;
        }
        for (let index = 0; index < shape.points.length; index += 1) {
            const point = imgToScreen(shape.points[index][0], shape.points[index][1]);
            if (
                Math.abs(point.x - sx) <= VERTEX_HIT &&
                Math.abs(point.y - sy) <= VERTEX_HIT
            ) {
                return index;
            }
        }
        return -1;
    }

    function hitShape(sx, sy) {
        const boxes = state.annotation.boxes;
        const imagePoint = screenToImg(sx, sy);
        for (let i = boxes.length - 1; i >= 0; i -= 1) {
            const shape = boxes[i];
            if (isPolygon(shape)) {
                if (pointInPolygon(shape.points, imagePoint.x, imagePoint.y)) {
                    return i;
                }
                continue;
            }
            const rect = normaliseBox(shape);
            const a = imgToScreen(rect.x1, rect.y1);
            const b = imgToScreen(rect.x2, rect.y2);
            if (sx >= a.x && sx <= b.x && sy >= a.y && sy <= b.y) {
                return i;
            }
        }
        return -1;
    }

    // ---------------------------------------------------------------- 历史

    function snapshot() {
        return JSON.parse(JSON.stringify(state.annotation.boxes));
    }

    function commitChange(before) {
        if (JSON.stringify(before) === JSON.stringify(state.annotation.boxes)) {
            return;
        }
        state.history.push(before);
        if (state.history.length > HISTORY_LIMIT) {
            state.history.shift();
        }
        state.future.length = 0;
        refreshBoxPanel();
        scheduleSave();
    }

    function undo() {
        if (!state.history.length) {
            return;
        }
        state.future.push(snapshot());
        state.annotation.boxes = state.history.pop();
        state.selected = -1;
        draw();
        refreshBoxPanel();
        scheduleSave();
    }

    function redo() {
        if (!state.future.length) {
            return;
        }
        state.history.push(snapshot());
        state.annotation.boxes = state.future.pop();
        state.selected = -1;
        draw();
        refreshBoxPanel();
        scheduleSave();
    }

    // ---------------------------------------------------------------- 保存

    function scheduleSave() {
        setSaveState('未保存', '');
        window.clearTimeout(state.saveTimer);
        state.saveTimer = window.setTimeout(saveNow, SAVE_DELAY);
    }

    async function saveNow() {
        if (!state.current || !state.image) {
            return;
        }
        window.clearTimeout(state.saveTimer);
        setSaveState('保存中…', 'saving');
        try {
            const payload = await api(
                `/api/annotation/${encodeURIComponent(state.current)}`,
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        boxes: state.annotation.boxes,
                        img_w: imageWidth(),
                        img_h: imageHeight(),
                    }),
                }
            );
            state.annotation.boxes = payload.boxes;
            setSaveState('已保存', 'saved');
            markAnnotated(state.current, payload.boxes.length);
            draw();
        } catch (error) {
            setSaveState('保存失败', 'error');
            toast(`保存失败：${error.message}`, 'error', 5000);
        }
    }

    function markAnnotated(filename, boxCount) {
        const item = state.images.find((entry) => entry.filename === filename);
        if (item) {
            item.annotated = true;
            item.boxes = boxCount;
        }
        const row = document.querySelector(`.image-item[data-name="${CSS.escape(filename)}"]`);
        if (row) {
            row.classList.add('annotated');
            const status = row.querySelector('.item-status');
            if (status) {
                status.textContent = boxCount ? `${boxCount} 个框` : '已标注（无目标）';
            }
        }
        updateProgress();
    }

    // ---------------------------------------------------------------- 指针交互

    function pointerPos(event) {
        const rect = canvas.getBoundingClientRect();
        return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    function onPointerDown(event) {
        if (!state.image) {
            return;
        }
        const pos = pointerPos(event);

        if (event.button === 1 || event.button === 2 || state.space) {
            event.preventDefault();
            state.drag = {
                type: 'pan',
                startX: pos.x,
                startY: pos.y,
                tx: state.view.tx,
                ty: state.view.ty,
            };
            viewport.classList.add('panning');
            canvas.setPointerCapture(event.pointerId);
            return;
        }

        if (event.button !== 0) {
            return;
        }

        const boxes = state.annotation.boxes;
        const imagePoint = screenToImg(pos.x, pos.y);

        // SAM：单击即取轮廓
        if (state.tool === 'sam') {
            samSegment(imagePoint);
            return;
        }

        // 选中的多边形优先处理顶点拖动 / Alt+点击删点
        if (state.selected >= 0 && boxes[state.selected]) {
            const vertex = hitVertex(boxes[state.selected], pos.x, pos.y);
            if (vertex >= 0) {
                if (event.altKey) {
                    removeVertex(state.selected, vertex);
                    return;
                }
                state.drag = {
                    type: 'vertex',
                    index: state.selected,
                    vertex,
                    before: snapshot(),
                };
                canvas.setPointerCapture(event.pointerId);
                return;
            }
        }

        // 多边形：单击加点，点回首点闭合
        if (state.tool === 'polygon') {
            if (state.draft) {
                const first = state.draft.points[0];
                const firstScreen = imgToScreen(first[0], first[1]);
                if (
                    state.draft.points.length >= 3 &&
                    Math.abs(firstScreen.x - pos.x) <= CLOSE_HIT &&
                    Math.abs(firstScreen.y - pos.y) <= CLOSE_HIT
                ) {
                    closeDraft();
                    return;
                }
                addDraftPoint(imagePoint);
                return;
            }

            const hitPolygon = hitShape(pos.x, pos.y);
            if (hitPolygon >= 0) {
                selectBox(hitPolygon);
                const shape = boxes[hitPolygon];
                state.drag = {
                    type: 'move',
                    index: hitPolygon,
                    start: imagePoint,
                    before: snapshot(),
                    orig: { ...normaliseBox(shape) },
                    origPoints: isPolygon(shape)
                        ? shape.points.map((point) => [...point])
                        : null,
                };
                canvas.setPointerCapture(event.pointerId);
                return;
            }

            startDraft(imagePoint);
            return;
        }

        // 矩形框：先判断缩放手柄
        if (state.selected >= 0 && boxes[state.selected]) {
            const handle = hitHandle(boxes[state.selected], pos.x, pos.y);
            if (handle >= 0) {
                state.drag = {
                    type: 'resize',
                    handle,
                    index: state.selected,
                    before: snapshot(),
                    orig: { ...normaliseBox(boxes[state.selected]) },
                };
                canvas.setPointerCapture(event.pointerId);
                return;
            }
        }

        const hit = hitShape(pos.x, pos.y);
        if (hit >= 0) {
            selectBox(hit);
            const shape = boxes[hit];
            state.drag = {
                type: 'move',
                index: hit,
                start: imagePoint,
                before: snapshot(),
                orig: { ...normaliseBox(shape) },
                origPoints: isPolygon(shape) ? shape.points.map((point) => [...point]) : null,
            };
            canvas.setPointerCapture(event.pointerId);
            return;
        }

        // 在空白处拖拽 → 新建矩形框
        const classId = defaultClassId();
        state.drag = {
            type: 'new',
            start: imagePoint,
            before: snapshot(),
            temp: {
                x1: imagePoint.x,
                y1: imagePoint.y,
                x2: imagePoint.x,
                y2: imagePoint.y,
                class_id: classId,
                class_name: classNameOf(classId),
                confidence: 1,
            },
        };
        selectBox(-1);
        canvas.setPointerCapture(event.pointerId);
    }

    function onPointerMove(event) {
        const pos = pointerPos(event);
        if (!state.drag) {
            if (state.tool === 'polygon' && state.draft) {
                state.hoverPoint = screenToImg(pos.x, pos.y);
                draw();
            }
            return;
        }
        const img = screenToImg(pos.x, pos.y);
        const iw = imageWidth();
        const ih = imageHeight();
        const drag = state.drag;

        if (drag.type === 'pan') {
            state.view.tx = drag.tx + (pos.x - drag.startX);
            state.view.ty = drag.ty + (pos.y - drag.startY);
            draw();
            return;
        }

        if (drag.type === 'new') {
            drag.temp.x2 = clamp(img.x, 0, iw);
            drag.temp.y2 = clamp(img.y, 0, ih);
            draw();
            return;
        }

        if (drag.type === 'vertex') {
            const shape = state.annotation.boxes[drag.index];
            if (!isPolygon(shape)) {
                return;
            }
            const points = shape.points.map((point, index) =>
                index === drag.vertex
                    ? [clamp(img.x, 0, iw), clamp(img.y, 0, ih)]
                    : point
            );
            state.annotation.boxes[drag.index] = { ...shape, points, ...boundsOf(points) };
            draw();
            return;
        }

        if (drag.type === 'move') {
            const dx = img.x - drag.start.x;
            const dy = img.y - drag.start.y;

            if (drag.origPoints) {
                const bounds = boundsOf(drag.origPoints);
                const width = bounds.x2 - bounds.x1;
                const height = bounds.y2 - bounds.y1;
                const offsetX = clamp(
                    dx,
                    -bounds.x1,
                    Math.max(0, iw - width) - bounds.x1
                );
                const offsetY = clamp(
                    dy,
                    -bounds.y1,
                    Math.max(0, ih - height) - bounds.y1
                );
                const points = drag.origPoints.map(([x, y]) => [x + offsetX, y + offsetY]);
                state.annotation.boxes[drag.index] = {
                    ...state.annotation.boxes[drag.index],
                    points,
                    ...boundsOf(points),
                };
                draw();
                return;
            }

            const width = drag.orig.x2 - drag.orig.x1;
            const height = drag.orig.y2 - drag.orig.y1;
            const x1 = clamp(drag.orig.x1 + dx, 0, Math.max(0, iw - width));
            const y1 = clamp(drag.orig.y1 + dy, 0, Math.max(0, ih - height));
            state.annotation.boxes[drag.index] = {
                ...state.annotation.boxes[drag.index],
                x1,
                y1,
                x2: x1 + width,
                y2: y1 + height,
            };
            draw();
            return;
        }

        if (drag.type === 'resize') {
            const box = { ...drag.orig };
            const x = clamp(img.x, 0, iw);
            const y = clamp(img.y, 0, ih);
            const handle = drag.handle;

            if ([0, 6, 7].includes(handle)) {
                box.x1 = Math.min(x, box.x2 - MIN_BOX_PX);
            }
            if ([2, 3, 4].includes(handle)) {
                box.x2 = Math.max(x, box.x1 + MIN_BOX_PX);
            }
            if ([0, 1, 2].includes(handle)) {
                box.y1 = Math.min(y, box.y2 - MIN_BOX_PX);
            }
            if ([4, 5, 6].includes(handle)) {
                box.y2 = Math.max(y, box.y1 + MIN_BOX_PX);
            }

            state.annotation.boxes[drag.index] = {
                ...state.annotation.boxes[drag.index],
                ...box,
            };
            draw();
        }
    }

    function onPointerUp(event) {
        const drag = state.drag;
        if (!drag) {
            return;
        }

        if (drag.type === 'new') {
            const box = normaliseBox(drag.temp);
            if (box.x2 - box.x1 >= MIN_BOX_PX && box.y2 - box.y1 >= MIN_BOX_PX) {
                state.annotation.boxes.push(box);
                state.selected = state.annotation.boxes.length - 1;
                commitChange(drag.before);
            }
        } else if (
            drag.type === 'move' ||
            drag.type === 'resize' ||
            drag.type === 'vertex'
        ) {
            const box = normaliseBox(state.annotation.boxes[drag.index]);
            state.annotation.boxes[drag.index] = box;
            commitChange(drag.before);
        }

        if (drag.type === 'pan') {
            viewport.classList.remove('panning');
        }

        state.drag = null;
        if (event.pointerId !== undefined && canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
        draw();
        refreshBoxPanel();
    }

    function onWheel(event) {
        if (!state.image) {
            return;
        }
        event.preventDefault();
        const pos = pointerPos(event);
        const before = screenToImg(pos.x, pos.y);
        const factor = Math.exp(-event.deltaY * 0.0013);
        state.view.scale = clamp(state.view.scale * factor, 0.02, 40);
        const after = imgToScreen(before.x, before.y);
        state.view.tx += pos.x - after.x;
        state.view.ty += pos.y - after.y;
        state.userZoomed = true;
        draw();
        updateZoomLabel();
    }

    function defaultClassId() {
        const boxes = state.annotation.boxes;
        if (state.selected >= 0 && boxes[state.selected]) {
            return boxes[state.selected].class_id;
        }
        const ids = Object.keys(state.classes).map(Number).sort((a, b) => a - b);
        if (ids.length) {
            return ids[0];
        }
        return boxes.length ? Number(boxes[boxes.length - 1].class_id) : 0;
    }

    function round1(value) {
        return Math.round(value * 10) / 10;
    }

    // ---------------------------------------------------------------- 多边形 / SAM

    const TOOL_HINTS = {
        box: '拖空白处画框 · 拖动框移动 · 空格或中键平移 · 滚轮缩放',
        polygon: '单击加点 · 点回首点或双击闭合 · Alt+点击顶点删除 · Esc 取消',
        sam: '点一下目标，SAM 自动出轮廓 · 首次使用会下载约 40 MB 模型',
    };

    function makePolygonShape(points, classId) {
        return {
            ...boundsOf(points),
            points,
            class_id: classId,
            class_name: classNameOf(classId),
            confidence: 1,
            shape: 'polygon',
        };
    }

    function startDraft(point) {
        state.draft = {
            points: [[clamp(point.x, 0, imageWidth()), clamp(point.y, 0, imageHeight())]],
        };
        state.hoverPoint = { ...point };
        state.selected = -1;
        draw();
        refreshBoxPanel();
    }

    function addDraftPoint(point) {
        if (!state.draft) {
            return;
        }
        state.draft.points.push([
            clamp(point.x, 0, imageWidth()),
            clamp(point.y, 0, imageHeight()),
        ]);
        state.hoverPoint = { ...point };
        draw();
    }

    function closeDraft() {
        if (!state.draft || state.draft.points.length < 3) {
            state.draft = null;
            state.hoverPoint = null;
            draw();
            return;
        }
        const points = state.draft.points.map(([x, y]) => [round1(x), round1(y)]);
        const before = snapshot();
        state.annotation.boxes.push(makePolygonShape(points, defaultClassId()));
        state.selected = state.annotation.boxes.length - 1;
        state.draft = null;
        state.hoverPoint = null;
        commitChange(before);
        draw();
        refreshBoxPanel();
    }

    function cancelDraft() {
        if (!state.draft) {
            return;
        }
        state.draft = null;
        state.hoverPoint = null;
        draw();
    }

    function removeVertex(index, vertex) {
        const shape = state.annotation.boxes[index];
        if (!isPolygon(shape)) {
            return;
        }
        if (shape.points.length <= 3) {
            toast('多边形至少保留 3 个顶点，整体删除请按 Delete', 'info');
            return;
        }
        const before = snapshot();
        const points = shape.points.filter((_, position) => position !== vertex);
        state.annotation.boxes[index] = { ...shape, points, ...boundsOf(points) };
        commitChange(before);
        draw();
        refreshBoxPanel();
    }

    async function samSegment(point) {
        if (!state.current || state.samBusy) {
            return;
        }
        if (!state.samReady) {
            toast('首次使用 SAM 会下载约 40 MB 模型，请稍候…', 'info', 6000);
        }

        state.samBusy = true;
        viewport.classList.add('sam-busy');
        try {
            const payload = await api('/api/sam', jsonRequest({
                filename: state.current,
                points: [[round1(point.x), round1(point.y)]],
                labels: [1],
            }));
            state.samReady = true;

            const polygons = payload.polygons || [];
            if (!polygons.length) {
                toast('这个位置没识别出目标，换个点试试', 'info');
                return;
            }

            const classId = defaultClassId();
            const before = snapshot();
            polygons.forEach((polygon) => {
                const points = polygon.map(([x, y]) => [round1(x), round1(y)]);
                if (points.length >= 3) {
                    state.annotation.boxes.push(makePolygonShape(points, classId));
                }
            });
            state.selected = state.annotation.boxes.length - 1;
            commitChange(before);
            draw();
            refreshBoxPanel();
            toast(`已生成 ${polygons.length} 个轮廓`, 'success');
        } catch (error) {
            toast(`SAM 失败：${error.message}`, 'error', 7000);
        } finally {
            state.samBusy = false;
            viewport.classList.remove('sam-busy');
        }
    }

    function setTool(tool) {
        state.tool = tool;
        if (tool !== 'polygon') {
            cancelDraft();
        }
        document.querySelectorAll('.tool-btn').forEach((button) => {
            button.classList.toggle('active', button.dataset.tool === tool);
        });
        viewport.classList.toggle('tool-polygon', tool === 'polygon');
        viewport.classList.toggle('tool-sam', tool === 'sam');
        $('toolHint').textContent = TOOL_HINTS[tool] || TOOL_HINTS.box;
        draw();
    }

    // ---------------------------------------------------------------- 标注框列表

    function selectBox(index) {
        state.selected = index;
        draw();
        refreshBoxPanel();
    }

    function refreshBoxPanel() {
        const host = $('boxList');
        host.innerHTML = '';
        const boxes = state.annotation.boxes;

        renderClassOptions();
        $('boxCount').textContent = `${boxes.length} 个框`;

        if (!boxes.length) {
            const empty = document.createElement('div');
            empty.className = 'empty-list';
            empty.textContent = state.current
                ? '这张图还没有框，在画布上拖一下就能画'
                : '先选一张图片';
            host.appendChild(empty);
            return;
        }

        boxes.forEach((box, index) => {
            const row = document.createElement('div');
            row.className = `box-row${index === state.selected ? ' selected' : ''}`;
            row.dataset.index = String(index);

            const badge = document.createElement('span');
            badge.className = 'box-index';
            badge.textContent = String(index + 1);

            const swatch = document.createElement('span');
            swatch.className = 'swatch';
            swatch.style.background = colorFor(box.class_id);

            const main = document.createElement('div');
            main.className = 'box-main';

            const title = document.createElement('div');
            title.className = 'box-title';

            if (index === state.selected) {
                // 用「可输入的类别框」而不是 <select>：
                // 1) <select> 的原生下拉会被整行重绘立刻销毁，点开就没了；
                // 2) 没加载模型时 select 里只有当前这一项，根本无从切换。
                const input = document.createElement('input');
                input.className = 'class-input';
                input.type = 'text';
                input.setAttribute('list', 'classOptions');
                input.value = box.class_name || classNameOf(box.class_id);
                input.title = '输入或选择类别名，回车确认；输新名字会自动新建类别';

                const stop = (event) => event.stopPropagation();
                input.addEventListener('pointerdown', stop);
                input.addEventListener('click', stop);
                input.addEventListener('keydown', (event) => {
                    event.stopPropagation();
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        input.blur();
                    } else if (event.key === 'Escape') {
                        input.value = box.class_name || classNameOf(box.class_id);
                        input.blur();
                    }
                });
                input.addEventListener('change', () => applyClassName(index, input));
                input.addEventListener('blur', () => applyClassName(index, input));
                title.appendChild(input);
            } else {
                const name = document.createElement('span');
                name.textContent = box.class_name || classNameOf(box.class_id);
                title.appendChild(name);
                if (box.confidence < 0.999) {
                    const conf = document.createElement('em');
                    conf.className = 'box-coord';
                    conf.textContent = `${(box.confidence * 100).toFixed(0)}%`;
                    title.appendChild(conf);
                }
            }

            const coord = document.createElement('div');
            coord.className = 'box-coord';
            coord.textContent = isPolygon(box)
                ? `${box.points.length} 个顶点 · x ${Math.round(box.x1)}–${Math.round(box.x2)}`
                : `x ${Math.round(box.x1)}–${Math.round(box.x2)} · ` +
                  `y ${Math.round(box.y1)}–${Math.round(box.y2)}`;

            main.appendChild(title);
            main.appendChild(coord);

            const remove = document.createElement('button');
            remove.className = 'remove';
            remove.title = '删除这个框';
            remove.textContent = '×';
            remove.addEventListener('click', (event) => {
                event.stopPropagation();
                removeBox(index);
            });

            row.appendChild(badge);
            row.appendChild(swatch);
            row.appendChild(main);
            row.appendChild(remove);
            row.addEventListener('click', (event) => {
                // 点输入框/按钮时不要触发整行重绘，否则正在编辑的控件会被销毁
                if (event.target.closest('input, select, button')) {
                    return;
                }
                selectBox(index);
            });

            host.appendChild(row);
        });
    }

    function nextClassId() {
        const ids = Object.keys(state.classes).map(Number);
        const used = state.annotation.boxes.map((box) => Number(box.class_id));
        const max = Math.max(-1, ...ids, ...used.filter((id) => Number.isFinite(id)));
        return max + 1;
    }

    /** 把输入框里的类别名应用到某个框；名字不存在就新建一个类别 id。 */
    function applyClassName(index, input) {
        const box = state.annotation.boxes[index];
        if (!box) {
            return;
        }

        const name = input.value.trim();
        const currentName = box.class_name || classNameOf(box.class_id);
        if (!name) {
            input.value = currentName;
            return;
        }

        let classId = Object.keys(state.classes).find((id) => state.classes[id] === name);
        if (classId === undefined) {
            const sameName = state.annotation.boxes.find((item) => item.class_name === name);
            classId = sameName ? String(sameName.class_id) : String(nextClassId());
            state.classes[classId] = name;
            renderClassList();
        }

        if (Number(classId) === Number(box.class_id) && name === currentName) {
            return;
        }

        const before = snapshot();
        state.annotation.boxes[index].class_id = Number(classId);
        state.annotation.boxes[index].class_name = name;
        commitChange(before);
        draw();
        refreshBoxPanel();
    }

    function renderClassOptions() {
        const host = $('classOptions');
        if (!host) {
            return;
        }
        const names = new Set(Object.values(state.classes));
        state.annotation.boxes.forEach((box) => {
            if (box.class_name) {
                names.add(box.class_name);
            }
        });

        host.innerHTML = '';
        [...names].sort((a, b) => a.localeCompare(b, 'zh-CN')).forEach((name) => {
            const option = document.createElement('option');
            option.value = name;
            host.appendChild(option);
        });
    }

    function removeBox(index) {
        const before = snapshot();
        state.annotation.boxes.splice(index, 1);
        state.selected = -1;
        commitChange(before);
        draw();
        refreshBoxPanel();
    }

    function nudgeSelected(dx, dy) {
        if (state.selected < 0) {
            return;
        }
        const box = normaliseBox(state.annotation.boxes[state.selected]);
        const width = box.x2 - box.x1;
        const height = box.y2 - box.y1;
        const x1 = clamp(box.x1 + dx, 0, Math.max(0, imageWidth() - width));
        const y1 = clamp(box.y1 + dy, 0, Math.max(0, imageHeight() - height));
        const before = snapshot();
        state.annotation.boxes[state.selected] = {
            ...state.annotation.boxes[state.selected],
            x1,
            y1,
            x2: x1 + width,
            y2: y1 + height,
        };
        commitChange(before);
        draw();
        refreshBoxPanel();
    }

    // ---------------------------------------------------------------- 图片

    async function refreshImages() {
        const payload = await api('/api/images');
        state.images = payload.images || [];
        renderImageList();
        updateProgress();
        if (state.images.length && !state.current) {
            selectImage(state.images[0].filename);
        }
        if (!state.images.length) {
            clearViewer();
        }
    }

    function renderImageList() {
        const host = $('imageList');
        host.innerHTML = '';
        const list = state.onlyPending
            ? state.images.filter((item) => !item.annotated)
            : state.images;

        $('listCount').textContent = state.onlyPending
            ? `${list.length} / ${state.images.length} 张`
            : `${state.images.length} 张`;

        if (!list.length) {
            const empty = document.createElement('div');
            empty.className = 'empty-list';
            empty.textContent = state.images.length ? '没有未标注的图片' : '还没有图片，先上传几张';
            host.appendChild(empty);
            return;
        }

        const fragment = document.createDocumentFragment();
        list.forEach((item) => {
            const row = document.createElement('div');
            row.className = 'image-item';
            row.dataset.name = item.filename;
            if (item.filename === state.current) {
                row.classList.add('active');
            }
            if (item.annotated) {
                row.classList.add('annotated');
            }

            const thumb = document.createElement('img');
            thumb.className = 'thumb';
            thumb.loading = 'lazy';
            thumb.src = item.url;
            thumb.alt = item.filename;

            const info = document.createElement('div');
            info.className = 'item-info';

            const name = document.createElement('div');
            name.className = 'item-name';
            name.textContent = item.filename;
            name.title = item.filename;

            const status = document.createElement('div');
            status.className = 'item-status';
            status.textContent = item.annotated
                ? (item.boxes ? `${item.boxes} 个框` : '已标注（无目标）')
                : '待标注';

            info.appendChild(name);
            info.appendChild(status);

            const remove = document.createElement('button');
            remove.className = 'delete-btn';
            remove.textContent = '×';
            remove.title = '删除这张图片';
            remove.addEventListener('click', async (event) => {
                event.stopPropagation();
                await deleteImage(item.filename);
            });

            row.appendChild(thumb);
            row.appendChild(info);
            row.appendChild(remove);
            row.addEventListener('click', () => selectImage(item.filename));

            fragment.appendChild(row);
        });

        host.appendChild(fragment);
    }

    function updateProgress() {
        const total = state.images.length;
        const done = state.images.filter((item) => item.annotated).length;
        const label = $('progressLabel');
        if (!total) {
            label.textContent = '未加载图片';
            label.classList.remove('ok');
        } else if (done === total) {
            label.textContent = `全部完成 ${done}/${total}`;
            label.classList.add('ok');
        } else {
            label.textContent = `已标注 ${done}/${total}`;
            label.classList.remove('ok');
        }
    }

    function selectImage(filename) {
        state.current = filename;
        $('fileName').textContent = filename;
        document.querySelectorAll('.image-item').forEach((row) => {
            row.classList.toggle('active', row.dataset.name === filename);
        });

        const token = (state.loadToken = (state.loadToken || 0) + 1);
        const image = new Image();
        image.onload = async () => {
            if (token !== state.loadToken) {
                return; // 用户已经切到别的图了，丢弃这次结果
            }
            state.image = image;
            $('emptyHint').classList.add('hidden');

            const naturalW = image.naturalWidth || 0;
            const naturalH = image.naturalHeight || 0;

            try {
                const payload = await api(`/api/annotation/${encodeURIComponent(filename)}`);
                if (token !== state.loadToken) {
                    return;
                }
                state.annotation = normaliseAnnotation(payload, naturalW, naturalH);
            } catch (error) {
                if (token !== state.loadToken) {
                    return;
                }
                state.annotation = { boxes: [], img_w: naturalW, img_h: naturalH };
                toast(`读取标注失败：${error.message}`, 'error');
            }

            state.selected = -1;
            state.history = [];
            state.future = [];
            state.draft = null;
            state.hoverPoint = null;
            fitView();
            refreshBoxPanel();
            setSaveState('—', '');
        };
        image.onerror = () => {
            if (token === state.loadToken) {
                toast(`图片加载失败：${filename}`, 'error');
            }
        };
        image.src = `/uploads/${encodeURIComponent(filename)}`;
    }

    /**
     * 把服务端返回的标注对齐到图片的真实像素网格。
     *
     * 旧标注、EXIF 旋转图都可能出现「存的尺寸 ≠ 图片实际尺寸」，
     * 这时按比例把坐标缩放过去，保证画面不被拉伸、框也不跑位。
     */
    function normaliseAnnotation(payload, naturalW, naturalH) {
        const boxes = Array.isArray(payload.boxes) ? payload.boxes : [];
        const annW = Number(payload.img_w) || naturalW;
        const annH = Number(payload.img_h) || naturalH;

        if (!naturalW || !naturalH || !annW || !annH) {
            return { boxes, img_w: naturalW || annW, img_h: naturalH || annH };
        }

        if (Math.abs(annW - naturalW) < 0.5 && Math.abs(annH - naturalH) < 0.5) {
            return { boxes, img_w: naturalW, img_h: naturalH };
        }

        const sx = naturalW / annW;
        const sy = naturalH / annH;
        return {
            boxes: boxes.map((box) => rescaleShape(box, sx, sy)),
            img_w: naturalW,
            img_h: naturalH,
        };
    }

    function rescaleShape(box, sx, sy) {
        const scaled = {
            ...box,
            x1: box.x1 * sx,
            y1: box.y1 * sy,
            x2: box.x2 * sx,
            y2: box.y2 * sy,
        };
        if (isPolygon(box)) {
            scaled.points = box.points.map(([x, y]) => [x * sx, y * sy]);
        }
        return scaled;
    }

    async function reloadAnnotation() {
        if (!state.current) {
            return;
        }
        try {
            const payload = await api(`/api/annotation/${encodeURIComponent(state.current)}`);
            state.annotation = normaliseAnnotation(payload, imageWidth(), imageHeight());
            draw();
            refreshBoxPanel();
        } catch (error) {
            /* 刷新失败不影响主流程 */
        }
    }

    function clearViewer() {
        state.current = null;
        state.image = null;
        state.annotation = { boxes: [], img_w: 0, img_h: 0 };
        state.selected = -1;
        state.loadToken = (state.loadToken || 0) + 1;
        $('fileName').textContent = '未选择图片';
        $('emptyHint').classList.remove('hidden');
        $('boxCount').textContent = '0 个框';
        refreshBoxPanel();
        draw();
    }

    async function deleteImage(filename) {
        try {
            await api('/api/delete_images', jsonRequest({ filenames: [filename] }));
            state.images = state.images.filter((item) => item.filename !== filename);
            if (state.current === filename) {
                clearViewer();
            }
            renderImageList();
            updateProgress();
            toast(`已删除 ${filename}`, 'info');
            if (!state.current && state.images.length) {
                selectImage(state.images[0].filename);
            }
        } catch (error) {
            toast(`删除失败：${error.message}`, 'error');
        }
    }

    // ---------------------------------------------------------------- 上传

    async function uploadFiles(fileList) {
        const files = Array.from(fileList || []).filter((file) =>
            /\.(jpe?g|png|bmp|webp)$/i.test(file.name)
        );
        if (!files.length) {
            toast('没有可上传的图片（支持 jpg / png / bmp / webp）', 'info');
            return;
        }

        const form = new FormData();
        files.forEach((file) => form.append('files', file));

        toast(`正在上传 ${files.length} 个文件…`, 'info', 2000);
        try {
            const payload = await api('/api/upload', { method: 'POST', body: form });
            toast(`上传成功 ${payload.count} 张`, 'success');
            if (payload.skipped && payload.skipped.length) {
                toast(
                    `跳过 ${payload.skipped.length} 个文件：${payload.skipped[0].reason}`,
                    'info',
                    5200
                );
            }
            await refreshImages();
            const last = payload.uploaded && payload.uploaded[payload.uploaded.length - 1];
            if (last) {
                selectImage(last.filename);
            }
        } catch (error) {
            toast(`上传失败：${error.message}`, 'error', 5200);
        }
    }

    // ---------------------------------------------------------------- 视频抽帧

    async function importVideo() {
        const input = $('videoFileInput');
        const file = input.files && input.files[0];
        if (!file) {
            toast('先选择一个视频文件', 'info');
            return;
        }

        const button = $('confirmVideo');
        const original = button.textContent;
        button.disabled = true;
        button.textContent = '抽帧中…';
        toast('正在抽帧，视频较长时需要一会儿…', 'info', 3000);

        const form = new FormData();
        form.append('video', file);
        form.append('interval', $('videoInterval').value);
        form.append('max_frames', $('videoMaxFrames').value);

        try {
            const payload = await api('/api/upload_video', { method: 'POST', body: form });
            const meta = payload.video || {};
            const extra = payload.duplicates ? `，跳过 ${payload.duplicates} 张重复帧` : '';
            const duration = meta.duration ? `（视频时长 ${meta.duration}s）` : '';
            toast(`抽到 ${payload.count} 帧${extra}${duration}`, 'success', 5000);

            $('videoModal').classList.add('hidden');
            input.value = '';
            await refreshImages();
            if (payload.frames && payload.frames.length) {
                selectImage(payload.frames[0]);
            }
        } catch (error) {
            toast(`抽帧失败：${error.message}`, 'error', 6000);
        } finally {
            button.disabled = false;
            button.textContent = original;
        }
    }

    // ---------------------------------------------------------------- 模型

    async function refreshModels() {
        const payload = await api('/api/models');
        const select = $('modelSelect');
        select.innerHTML = '';

        const pretrained = document.createElement('optgroup');
        pretrained.label = '预训练模型';
        (payload.pretrained || []).forEach((model) => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.label;
            pretrained.appendChild(option);
        });
        select.appendChild(pretrained);

        if (payload.custom && payload.custom.length) {
            const custom = document.createElement('optgroup');
            custom.label = '自定义模型';
            payload.custom.forEach((model) => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.label;
                custom.appendChild(option);
            });
            select.appendChild(custom);
        }

        const upload = document.createElement('option');
        upload.value = '__upload__';
        upload.textContent = '📤 上传自定义模型…';
        select.appendChild(upload);

        if (payload.current) {
            select.value = payload.current;
            state.modelLoaded = true;
            state.modelName = payload.current;
            setModelBadge(`模型：${payload.current}`, true);
        }
    }

    function setModelBadge(text, ok) {
        const badge = $('modelBadge');
        badge.textContent = text;
        badge.classList.toggle('ok', Boolean(ok));
    }

    async function loadModel() {
        const name = $('modelSelect').value;
        if (!name || name === '__upload__') {
            toast('请先选择一个模型', 'info');
            return;
        }

        const button = $('loadModelBtn');
        button.disabled = true;
        button.textContent = '加载中…';
        toast('正在加载模型，首次使用需要下载权重，请稍候…', 'info', 4000);

        try {
            const payload = await api('/api/load_model', jsonRequest({ model_name: name }));
            state.modelLoaded = true;
            state.modelName = payload.model_name;
            state.classes = payload.classes || {};
            setModelBadge(`模型：${payload.model_name}`, true);
            renderClassList();
            toast(
                `模型已加载：${payload.model_name}（${Object.keys(state.classes).length} 个类别）`,
                'success'
            );
        } catch (error) {
            state.modelLoaded = false;
            setModelBadge('未加载模型', false);
            toast(`加载失败：${error.message}`, 'error', 6000);
        } finally {
            button.disabled = false;
            button.textContent = '加载模型';
        }
    }

    // ---------------------------------------------------------------- 类别

    function renderClassList() {
        const host = $('classList');
        host.innerHTML = '';
        const ids = Object.keys(state.classes).map(Number).sort((a, b) => a - b);

        $('classCount').textContent = `${ids.length} 类`;

        if (!ids.length) {
            const empty = document.createElement('div');
            empty.className = 'empty-list';
            empty.textContent = '加载模型后，这里会列出模型的全部类别';
            host.appendChild(empty);
            return;
        }

        const fragment = document.createDocumentFragment();
        ids.forEach((classId) => {
            const row = document.createElement('label');
            row.className = 'class-row';
            if (state.classFilter.has(classId)) {
                row.classList.add('selected');
            }

            const check = document.createElement('input');
            check.type = 'checkbox';
            check.checked = state.classFilter.has(classId);
            check.addEventListener('change', () => {
                if (check.checked) {
                    state.classFilter.add(classId);
                    row.classList.add('selected');
                } else {
                    state.classFilter.delete(classId);
                    row.classList.remove('selected');
                }
            });

            const swatch = document.createElement('span');
            swatch.className = 'swatch';
            swatch.style.background = colorFor(classId);

            const name = document.createElement('span');
            name.className = 'class-name';
            name.textContent = state.classes[String(classId)];

            const idNode = document.createElement('span');
            idNode.className = 'class-id';
            idNode.textContent = `#${classId}`;

            row.appendChild(check);
            row.appendChild(swatch);
            row.appendChild(name);
            row.appendChild(idNode);
            fragment.appendChild(row);
        });
        host.appendChild(fragment);
    }

    function activeClassFilter() {
        return Array.from(state.classFilter);
    }

    async function refreshClasses() {
        try {
            const payload = await api('/api/classes');
            if (payload && payload.model) {
                state.modelLoaded = true;
                state.modelName = payload.model;
                state.classes = payload.classes || {};
                renderClassList();
            }
        } catch (error) {
            /* 类别拉取失败不影响标注本身 */
        }
    }

    // ---------------------------------------------------------------- 推理

    function requireModel() {
        if (!state.modelLoaded) {
            toast('请先加载模型', 'info');
            return false;
        }
        return true;
    }

    async function annotateCurrent() {
        if (!requireModel()) {
            return;
        }
        if (!state.current) {
            toast('先选一张图片', 'info');
            return;
        }
        const button = $('annotateCurrentBtn');
        button.disabled = true;
        button.textContent = '推理中…';
        try {
            const payload = await api('/api/annotate', jsonRequest({
                filename: state.current,
                confidence: Number($('confidenceSlider').value),
                iou: Number($('iouSlider').value),
                classes: activeClassFilter(),
            }));
            state.annotation = {
                boxes: payload.boxes || [],
                img_w: payload.img_w || imageWidth(),
                img_h: payload.img_h || imageHeight(),
            };
            state.selected = -1;
            state.history = [];
            state.future = [];
            draw();
            refreshBoxPanel();
            setSaveState('已保存', 'saved');
            markAnnotated(state.current, state.annotation.boxes.length);
            toast(`检测到 ${state.annotation.boxes.length} 个目标`, 'success');
        } catch (error) {
            toast(`标注失败：${error.message}`, 'error', 5200);
        } finally {
            button.disabled = false;
            button.textContent = '标注当前图';
        }
    }

    // ---------------------------------------------------------------- 批量

    async function startBatch() {
        if (!requireModel()) {
            return;
        }
        const body = {
            confidence: Number($('confidenceSlider').value),
            iou: Number($('iouSlider').value),
            classes: activeClassFilter(),
            include_annotated: $('reannotate').checked,
        };

        try {
            const job = await api('/api/batch', jsonRequest(body));
            state.batchId = job.id;
            $('batchOverlay').classList.remove('hidden');
            $('batchDone').textContent = '0';
            $('batchTotal').textContent = String(job.total || 0);
            $('batchFailed').textContent = '';
            $('batchCurrent').textContent = '';
            $('batchBar').style.width = '0%';
            pollBatch();
        } catch (error) {
            toast(`无法开始批量标注：${error.message}`, 'error', 5200);
        }
    }

    async function pollBatch() {
        if (!state.batchId) {
            return;
        }
        try {
            const job = await api(`/api/batch/${state.batchId}`);
            const total = job.total || 0;
            const done = job.done || 0;
            const percent = total ? Math.round((done / total) * 100) : 0;
            $('batchBar').style.width = `${percent}%`;
            $('batchDone').textContent = String(done);
            $('batchFailed').textContent = job.failed ? `（失败 ${job.failed}）` : '';
            $('batchCurrent').textContent = job.current ? `正在处理：${job.current}` : '';

            if (job.status === 'running') {
                state.batchTimer = window.setTimeout(pollBatch, 400);
                return;
            }
            finishBatch(job);
        } catch (error) {
            $('batchOverlay').classList.add('hidden');
            toast(`批量标注中断：${error.message}`, 'error', 5200);
            state.batchId = null;
        }
    }

    async function finishBatch(job) {
        state.batchId = null;
        window.clearTimeout(state.batchTimer);
        $('batchOverlay').classList.add('hidden');

        await refreshImages();
        await reloadAnnotation();

        if (job.status === 'cancelled') {
            toast(`已取消，完成 ${job.done}/${job.total} 张`, 'info');
        } else if (job.failed) {
            const first = (job.errors && job.errors[0]) || null;
            toast(
                `批量完成：成功 ${job.done} 张，失败 ${job.failed} 张` +
                (first ? `（${first.error}）` : ''),
                'error',
                6000
            );
        } else {
            toast(`批量标注完成，共 ${job.done} 张`, 'success');
        }
    }

    async function cancelBatch() {
        if (!state.batchId) {
            return;
        }
        try {
            await api(`/api/batch/${state.batchId}/cancel`, { method: 'POST' });
            $('batchCurrent').textContent = '正在取消…';
        } catch (error) {
            toast(`取消失败：${error.message}`, 'error');
        }
    }

    // ---------------------------------------------------------------- 导出

    function renderExportFormats() {
        const host = $('exportFormats');
        host.innerHTML = '';
        FORMATS.forEach((format) => {
            const button = document.createElement('button');
            button.className = `format${format.value === state.exportFormat ? ' active' : ''}`;
            button.dataset.format = format.value;
            button.innerHTML = `${format.label}<small>${format.desc}</small>`;
            button.addEventListener('click', () => {
                state.exportFormat = format.value;
                host.querySelectorAll('.format').forEach((node) => {
                    node.classList.toggle('active', node.dataset.format === format.value);
                });
            });
            host.appendChild(button);
        });
    }

    function filenameFromDisposition(header) {
        if (!header) {
            return null;
        }
        const utf8 = /filename\*=UTF-8''([^;]+)/i.exec(header);
        if (utf8) {
            try {
                return decodeURIComponent(utf8[1]);
            } catch (error) {
                return utf8[1];
            }
        }
        const plain = /filename="?([^";]+)"?/i.exec(header);
        return plain ? plain[1] : null;
    }

    async function confirmExport() {
        const button = $('confirmExport');
        button.disabled = true;
        button.textContent = '打包中…';
        try {
            const response = await fetch('/api/export', jsonRequest({
                format: state.exportFormat,
                include_images: $('includeImages').checked,
                only_used_classes: $('onlyUsedClasses').checked,
                val_ratio: $('splitVal').checked ? Number($('valRatio').value) : 0,
            }));

            if (!response.ok) {
                const payload = await response.json().catch(() => null);
                throw new Error((payload && payload.error) || `导出失败（HTTP ${response.status}）`);
            }

            const blob = await response.blob();
            const filename = filenameFromDisposition(response.headers.get('Content-Disposition')) ||
                `yolo_export_${Date.now()}.zip`;

            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.setTimeout(() => URL.revokeObjectURL(url), 5000);

            $('exportModal').classList.add('hidden');
            toast(`已下载 ${filename}（${(blob.size / 1024).toFixed(0)} KB）`, 'success', 4500);
        } catch (error) {
            toast(error.message, 'error', 6000);
        } finally {
            button.disabled = false;
            button.textContent = '下载 zip';
        }
    }

    // ---------------------------------------------------------------- 键盘

    function isTyping(target) {
        if (!target) {
            return false;
        }
        const tag = target.tagName;
        return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || target.isContentEditable;
    }

    function onKeyDown(event) {
        if (event.key === ' ' && !isTyping(event.target)) {
            state.space = true;
            event.preventDefault();
            return;
        }
        if (isTyping(event.target)) {
            return;
        }

        const meta = event.ctrlKey || event.metaKey;

        if (meta && event.key.toLowerCase() === 'z') {
            event.preventDefault();
            if (event.shiftKey) {
                redo();
            } else {
                undo();
            }
            return;
        }
        if (meta && event.key.toLowerCase() === 'y') {
            event.preventDefault();
            redo();
            return;
        }
        if (meta && event.key.toLowerCase() === 's') {
            event.preventDefault();
            saveNow();
            return;
        }

        switch (event.key) {
            case 'Delete':
            case 'Backspace':
                if (state.selected >= 0) {
                    event.preventDefault();
                    removeBox(state.selected);
                }
                break;
            case 'Escape':
                cancelDraft();
                state.selected = -1;
                draw();
                refreshBoxPanel();
                $('exportModal').classList.add('hidden');
                $('helpModal').classList.add('hidden');
                $('videoModal').classList.add('hidden');
                break;
            case 'Enter':
                if (state.draft) {
                    event.preventDefault();
                    closeDraft();
                }
                break;
            case 'b':
            case 'B':
                setTool('box');
                break;
            case 'p':
            case 'P':
                setTool('polygon');
                break;
            case 's':
            case 'S':
                setTool('sam');
                break;
            case 'ArrowLeft':
            case 'ArrowRight':
            case 'ArrowUp':
            case 'ArrowDown': {
                if (state.selected < 0) {
                    break;
                }
                event.preventDefault();
                const step = event.shiftKey ? 10 : 1;
                if (event.key === 'ArrowLeft') {
                    nudgeSelected(-step, 0);
                } else if (event.key === 'ArrowRight') {
                    nudgeSelected(step, 0);
                } else if (event.key === 'ArrowUp') {
                    nudgeSelected(0, -step);
                } else {
                    nudgeSelected(0, step);
                }
                break;
            }
            case 'PageUp':
                event.preventDefault();
                selectRelativeImage(-1);
                break;
            case 'PageDown':
                event.preventDefault();
                selectRelativeImage(1);
                break;
            case 'f':
            case 'F':
                fitView();
                break;
            case 'h':
            case 'H':
                $('helpModal').classList.remove('hidden');
                break;
            case '+':
            case '=':
                zoomBy(1.2);
                break;
            case '-':
                zoomBy(1 / 1.2);
                break;
            default:
                break;
        }

        if (/^[1-9]$/.test(event.key) && state.selected >= 0) {
            const ids = Object.keys(state.classes).map(Number).sort((a, b) => a - b);
            const classId = ids[Number(event.key) - 1];
            if (classId !== undefined) {
                const before = snapshot();
                state.annotation.boxes[state.selected].class_id = classId;
                state.annotation.boxes[state.selected].class_name = classNameOf(classId);
                commitChange(before);
                draw();
                refreshBoxPanel();
            }
        }
    }

    function onKeyUp(event) {
        if (event.key === ' ') {
            state.space = false;
        }
    }

    function zoomBy(factor) {
        if (!state.image) {
            return;
        }
        const centerX = state.width / 2;
        const centerY = state.height / 2;
        const before = screenToImg(centerX, centerY);
        state.view.scale = clamp(state.view.scale * factor, 0.02, 40);
        const after = imgToScreen(before.x, before.y);
        state.view.tx += centerX - after.x;
        state.view.ty += centerY - after.y;
        state.userZoomed = true;
        draw();
        updateZoomLabel();
    }

    function selectRelativeImage(delta) {
        if (!state.images.length) {
            return;
        }
        const visible = state.onlyPending
            ? state.images.filter((item) => !item.annotated)
            : state.images;
        if (!visible.length) {
            return;
        }
        const index = visible.findIndex((item) => item.filename === state.current);
        const next = visible[(index + delta + visible.length) % visible.length];
        if (next) {
            selectImage(next.filename);
        }
    }

    // ---------------------------------------------------------------- 绑定

    function bindEvents() {
        $('uploadBtn').addEventListener('click', () => $('fileInput').click());
        $('fileInput').addEventListener('change', (event) => {
            uploadFiles(event.target.files);
            event.target.value = '';
        });

        $('videoBtn').addEventListener('click', () => $('videoModal').classList.remove('hidden'));
        $('cancelVideo').addEventListener('click', () => $('videoModal').classList.add('hidden'));
        $('confirmVideo').addEventListener('click', importVideo);

        $('modelSelect').addEventListener('change', (event) => {
            if (event.target.value === '__upload__') {
                $('modelFileInput').click();
                event.target.value = state.modelName || '';
            }
        });
        $('modelFileInput').addEventListener('change', async (event) => {
            const file = event.target.files && event.target.files[0];
            event.target.value = '';
            if (!file) {
                return;
            }
            const form = new FormData();
            form.append('model', file);
            try {
                const payload = await api('/api/upload_model', { method: 'POST', body: form });
                toast(`模型已上传：${payload.name}`, 'success');
                await refreshModels();
                $('modelSelect').value = payload.name;
            } catch (error) {
                toast(`模型上传失败：${error.message}`, 'error');
            }
        });

        $('loadModelBtn').addEventListener('click', loadModel);
        $('annotateCurrentBtn').addEventListener('click', annotateCurrent);
        $('batchBtn').addEventListener('click', startBatch);
        $('batchCancel').addEventListener('click', cancelBatch);

        $('confidenceSlider').addEventListener('input', (event) => {
            $('confidenceVal').textContent = Number(event.target.value).toFixed(2);
            localStorage.setItem('yolo.confidence', event.target.value);
        });
        $('iouSlider').addEventListener('input', (event) => {
            $('iouVal').textContent = Number(event.target.value).toFixed(2);
            localStorage.setItem('yolo.iou', event.target.value);
        });

        $('onlyPending').addEventListener('change', (event) => {
            state.onlyPending = event.target.checked;
            renderImageList();
        });

        $('clearAllBtn').addEventListener('click', async () => {
            if (!state.images.length) {
                toast('没有图片可清空', 'info');
                return;
            }
            if (!window.confirm(`确定清空全部 ${state.images.length} 张图片和标注吗？该操作不可恢复。`)) {
                return;
            }
            try {
                await api('/api/clear_all', { method: 'POST' });
                state.images = [];
                clearViewer();
                renderImageList();
                updateProgress();
                toast('已清空', 'info');
            } catch (error) {
                toast(`清空失败：${error.message}`, 'error');
            }
        });

        $('selectAllClasses').addEventListener('click', () => {
            Object.keys(state.classes).forEach((id) => state.classFilter.add(Number(id)));
            renderClassList();
        });
        $('clearClasses').addEventListener('click', () => {
            state.classFilter.clear();
            renderClassList();
        });

        document.querySelectorAll('.tab').forEach((tab) => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach((node) => node.classList.remove('active'));
                document.querySelectorAll('.panel').forEach((node) => node.classList.remove('active'));
                tab.classList.add('active');
                $(`panel-${tab.dataset.panel}`).classList.add('active');
            });
        });

        $('exportBtn').addEventListener('click', () => {
            if (!state.images.some((item) => item.annotated)) {
                toast('还没有标注数据，先标注几张图片', 'info');
                return;
            }
            $('exportModal').classList.remove('hidden');
        });
        $('cancelExport').addEventListener('click', () => $('exportModal').classList.add('hidden'));
        $('confirmExport').addEventListener('click', confirmExport);

        $('helpBtn').addEventListener('click', () => $('helpModal').classList.remove('hidden'));
        $('closeHelp').addEventListener('click', () => $('helpModal').classList.add('hidden'));

        $('shutdownBtn').addEventListener('click', async () => {
            if (!window.confirm('确定关闭本地服务吗？关闭后需要重新运行 start.bat。')) {
                return;
            }
            try {
                await api('/api/shutdown', { method: 'POST' });
            } catch (error) {
                /* 服务关掉后 fetch 失败是预期行为 */
            }
            toast('服务已关闭，可以关掉这个页面了', 'success');
            document.querySelectorAll('button').forEach((button) => {
                button.disabled = true;
            });
        });

        canvas.addEventListener('pointerdown', onPointerDown);
        canvas.addEventListener('pointermove', onPointerMove);
        canvas.addEventListener('pointerup', onPointerUp);
        canvas.addEventListener('pointercancel', onPointerUp);
        canvas.addEventListener('wheel', onWheel, { passive: false });
        canvas.addEventListener('contextmenu', (event) => event.preventDefault());
        canvas.addEventListener('dblclick', (event) => {
            if (state.tool === 'polygon' && state.draft) {
                event.preventDefault();
                closeDraft();
            }
        });

        document.querySelectorAll('.tool-btn').forEach((button) => {
            button.addEventListener('click', () => setTool(button.dataset.tool));
        });

        document.addEventListener('keydown', onKeyDown);
        document.addEventListener('keyup', onKeyUp);

        // 拖拽上传
        let dragDepth = 0;
        const hasFiles = (event) =>
            Array.from(event.dataTransfer ? event.dataTransfer.types : []).includes('Files');

        viewport.addEventListener('dragenter', (event) => {
            if (!hasFiles(event)) {
                return;
            }
            event.preventDefault();
            dragDepth += 1;
            viewport.classList.add('dragging');
        });
        viewport.addEventListener('dragover', (event) => {
            if (hasFiles(event)) {
                event.preventDefault();
            }
        });
        viewport.addEventListener('dragleave', (event) => {
            if (!hasFiles(event)) {
                return;
            }
            dragDepth = Math.max(0, dragDepth - 1);
            if (!dragDepth) {
                viewport.classList.remove('dragging');
            }
        });
        viewport.addEventListener('drop', (event) => {
            if (!hasFiles(event)) {
                return;
            }
            event.preventDefault();
            dragDepth = 0;
            viewport.classList.remove('dragging');
            uploadFiles(event.dataTransfer.files);
        });

        if (window.ResizeObserver) {
            new ResizeObserver(() => {
                resizeCanvas();
                if (!state.userZoomed) {
                    fitView();
                }
            }).observe(viewport);
        } else {
            window.addEventListener('resize', () => {
                resizeCanvas();
                fitView();
            });
        }
    }

    // ---------------------------------------------------------------- 启动

    async function init() {
        viewport = $('viewport');
        canvas = $('canvas');
        ctx = canvas.getContext('2d');

        const savedConfidence = localStorage.getItem('yolo.confidence');
        if (savedConfidence) {
            $('confidenceSlider').value = savedConfidence;
            $('confidenceVal').textContent = Number(savedConfidence).toFixed(2);
        }
        const savedIou = localStorage.getItem('yolo.iou');
        if (savedIou) {
            $('iouSlider').value = savedIou;
            $('iouVal').textContent = Number(savedIou).toFixed(2);
        }

        bindEvents();
        renderExportFormats();
        setTool('box');
        resizeCanvas();
        refreshBoxPanel();

        try {
            const samStatus = await api('/api/sam/status');
            state.samReady = Boolean(samStatus && samStatus.loaded);
        } catch (error) {
            state.samReady = false;
        }

        try {
            await refreshModels();
            if (state.modelLoaded) {
                await refreshClasses();
            }
            await refreshImages();
            if (Object.keys(state.classes).length) {
                renderClassList();
            }
        } catch (error) {
            toast(`初始化失败：${error.message}`, 'error', 6000);
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
