@echo off
chcp 65001 >nul
cd /d "%~dp0"
title YOLO 智能标注工具

where python >nul 2>&1
if errorlevel 1 (
    echo [错误] 没有找到 python，请先安装 Python 3.9+ 并加入 PATH
    pause
    exit /b 1
)

echo 检查依赖...
python -c "import flask, ultralytics, PIL" >nul 2>&1
if errorlevel 1 (
    echo 首次运行，正在安装依赖，可能需要几分钟...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络或手动执行 pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo 清理占用 5000 端口的旧进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo 启动服务，浏览器会自动打开 http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py

pause
