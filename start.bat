@echo off
cd /d "%~dp0"
echo 正在清理旧进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000.*LISTEN"') do (
    echo  杀掉 PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)
echo 启动 YOLO 智能标注工具...
python app.py
pause
