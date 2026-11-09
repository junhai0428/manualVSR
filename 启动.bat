@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 解释器来源优先级：环境变量 SUBERASER_PYTHON > local_config.json 的 python 字段 > PATH 中的 python
if defined SUBERASER_PYTHON set "PYTHON=%SUBERASER_PYTHON%"
if not defined PYTHON (
  if exist "local_config.json" (
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "try{(Get-Content -Raw -Encoding UTF8 'local_config.json' | ConvertFrom-Json).python}catch{}"`) do set "PYTHON=%%i"
  )
)
if not defined PYTHON set "PYTHON=python"

if not exist "%PYTHON%" (
  echo [错误] 找不到 Python: %PYTHON%
  echo         请在 local_config.json 里填 "python": "解释器绝对路径"
  echo         或设置环境变量 SUBERASER_PYTHON
  pause
  exit /b 1
)
echo [启动] 手动字幕擦除  (首次加载 lama 模型约 10-50 秒)
"%PYTHON%" run.py
if errorlevel 1 pause
