# -*- coding: utf-8 -*-
"""本机私有配置：解释器路径、模型目录、测试素材一律不写死在代码里。

优先级：环境变量 > 项目根 `local_config.json`（不入版本库）> 通用默认值。

`local_config.json` 示例（按需只写其中几项）::

    {
      "python":        "C:/envs/iopaint/Scripts/python.exe",
      "detect_python": "C:/envs/ocr/Scripts/python.exe",
      "model_dir":     "D:/models",
      "test_video":    "D:/media/sample.mp4"
    }

对应环境变量：SUBERASER_PYTHON / SUBERASER_DETECT_PY /
SUBERASER_MODEL_DIR / SE_TEST_VIDEO
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "local_config.json")

_ENV_KEYS = {
    "python": "SUBERASER_PYTHON",
    "detect_python": "SUBERASER_DETECT_PY",
    "model_dir": "SUBERASER_MODEL_DIR",
    "test_video": "SE_TEST_VIDEO",
}

_cache: Dict[str, Any] = {}


def _load() -> Dict[str, Any]:
    if not _cache:
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _cache.update({k: v for k, v in data.items() if v})
        except (OSError, ValueError):
            pass
    return _cache


def get(key: str, default: str = "") -> str:
    """取配置：环境变量优先，其次 local_config.json，最后默认值。"""
    env_key = _ENV_KEYS.get(key)
    if env_key:
        val = os.environ.get(env_key)
        if val:
            return val
    val = _load().get(key)
    return str(val) if val else default


def model_dir() -> str:
    """lama 权重缓存根目录（torch.hub / iopaint 都指向它）。"""
    return get("model_dir", os.path.join(os.path.expanduser("~"), ".suberaser", "models"))


def detect_python() -> str:
    """跑 OCR 检测子进程的解释器；空串表示未配置。"""
    return get("detect_python", "")


def test_video() -> str:
    """端到端测试素材；默认取 tests/douyin_012.mp4（仓库不含素材，缺省即跳过测试）。"""
    return get("test_video", os.path.join(ROOT, "tests", "douyin_012.mp4"))
