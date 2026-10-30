@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHON=<MAIN_ENV>\Scripts\python.exe
if not exist "%PYTHON%" (
  echo [错误] 找不到 Python: %PYTHON%
  pause
  exit /b 1
)
echo [启动] 手动字幕擦除  (首次加载 lama 模型约 10-50 秒)
"%PYTHON%" run.py
if errorlevel 1 pause
