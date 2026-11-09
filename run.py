# -*- coding: utf-8 -*-
r"""入口：先设置缓存/线程环境变量（必须在 import torch 之前），再启动 GUI。

用法:  <你的 Python 解释器> run.py（需装 PySide6 + iopaint）
       推荐直接双击 启动.bat —— 它会从 local_config.json / SUBERASER_PYTHON 取解释器
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from tools.prepare_model import set_cache_env, ensure_model  # noqa: E402

set_cache_env()
try:
    ensure_model(verbose=False)
except FileNotFoundError as e:
    print(f"[warn] {e}")


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
