# -*- coding: utf-8 -*-
"""GUI 冒烟测试（offscreen）：打开视频 → 拖框 → 键盘逐帧 → 设出点 → 一键擦除。

运行:  <MAIN_ENV>\\Scripts\\python.exe tests/test_gui.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.prepare_model import set_cache_env  # noqa: E402

set_cache_env()

from PySide6.QtCore import Qt, QPoint, QEventLoop, QTimer  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from src.ui.main_window import MainWindow  # noqa: E402
from src.project import default_project_path  # noqa: E402
from src.video_io import probe  # noqa: E402
from tools.make_test_video import gen  # noqa: E402

VIDEO = r"E:\tmp\suberase_test\test_sub.mp4"
OUT = r"E:\tmp\suberase_test\gui_out.mp4"

app = QApplication.instance() or QApplication(sys.argv)
ok_all = True
_finished = {}


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))


def wait_until(pred, timeout_ms=120000, step=100):
    loop = QEventLoop()
    t = QTimer()
    t.setInterval(step)

    def tick():
        if pred():
            t.stop()
            loop.quit()

    t.timeout.connect(tick)
    t.start()
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return pred()


def main() -> int:
    if not os.path.exists(VIDEO):
        gen(VIDEO, seconds=4)
    # 清掉上一轮遗留的工程文件，保证从干净状态开始
    pj = default_project_path(VIDEO)
    if os.path.exists(pj):
        try:
            os.remove(pj)
        except OSError:
            pass

    w = MainWindow()
    w.show()
    app.processEvents()

    check("窗口构造", w.meta is None and w.view is not None)

    # 等模型加载
    w.loader.wait(180000)
    app.processEvents()
    check("模型后台加载", w.engine is not None,
          w.engine.summary() if w.engine else "engine=None")

    # 打开视频
    w.open_video(VIDEO)
    app.processEvents()
    check("打开视频", w.meta is not None and w.reader is not None,
          f"{w.meta.width}x{w.meta.height} {w.meta.nb_frames}帧" if w.meta else "")
    check("首帧已显示", w.view._img is not None)

    # 模拟拖框（画面下部字幕带）
    r = w.view.view_rect
    check("显示区域有效", r.width() > 10 and r.height() > 10, f"{r.width()}x{r.height()}")
    p1 = QPoint(r.x() + int(0.15 * r.width()), r.y() + int(0.80 * r.height()))
    p2 = QPoint(r.x() + int(0.85 * r.width()), r.y() + int(0.95 * r.height()))
    QTest.mousePress(w.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
    QTest.mouseMove(w.view, p2)
    QTest.mouseRelease(w.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p2)
    app.processEvents()
    check("拖框新建", len(w.project.boxes) == 1,
          str([(b.x, b.y, b.w, b.h, b.start, b.end) for b in w.project.boxes]))
    b = w.project.boxes[0] if w.project.boxes else None
    if b:
        check("框位于画面下部", b.y > w.meta.height * 0.6, f"y={b.y} h={b.h}")
        check("新建框时段=当前帧", b.start == 0 and b.end == 0, f"{b.start}-{b.end}")

    # 键盘逐帧
    f0 = w.frame_idx
    QTest.keyClick(w, Qt.Key.Key_Right)
    app.processEvents()
    QTest.keyClick(w, Qt.Key.Key_Right)
    app.processEvents()
    check("键盘 → 逐帧前进", w.frame_idx == f0 + 2, f"{f0} -> {w.frame_idx}")
    QTest.keyClick(w, Qt.Key.Key_Left)
    app.processEvents()
    check("键盘 ← 逐帧后退", w.frame_idx == f0 + 1, f"-> {w.frame_idx}")
    QTest.keyClick(w, Qt.Key.Key_End)
    app.processEvents()
    check("End 跳末帧", w.frame_idx == w.meta.nb_frames - 1, f"{w.frame_idx}")

    # 设出点（O）
    QTest.keyClick(w, Qt.Key.Key_O)
    app.processEvents()
    check("O 设出点", b is not None and b.end == w.meta.nb_frames - 1,
          f"end={b.end if b else None}")

    # 时间轴拖动
    w.seek(30)
    app.processEvents()
    check("seek 后时段内仍活跃", b is not None and b.covers(30), f"frame={w.frame_idx}")

    # 回归：过期框不应再占位，且框内可以新建框
    b.start, b.end = 0, 20
    w.project.boxes = [b]
    w.panel.project = w.project
    w.panel.refresh()
    w._refresh_boxes()
    w.seek(60)
    app.processEvents()
    check("过期框不参与交互", b not in w.view._active_boxes(), f"frame={w.frame_idx}")
    check("过期框默认不绘制", not w.view.show_inactive)
    QTest.mousePress(w.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
    QTest.mouseMove(w.view, p2)
    QTest.mouseRelease(w.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p2)
    app.processEvents()
    check("过期框内可新建框", len(w.project.boxes) == 2,
          str([(x.start, x.end) for x in w.project.boxes]))
    check("原框未被拖动", (b.start, b.end) == (0, 20), f"{b.start}-{b.end}")
    check("新框时段=当前帧", w.project.boxes[1].start == 60 and w.project.boxes[1].end == 60)
    # Esc 清除选中
    QTest.keyClick(w, Qt.Key.Key_Escape)
    app.processEvents()
    check("Esc 取消选中", w.view.selected_id is None)
    w.project.boxes = [b]   # 还原为单框继续跑擦除
    w._refresh_boxes()
    b.start, b.end = 0, w.meta.nb_frames - 1

    # 一键擦除（屏蔽对话框）
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (OUT, ""))
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    if os.path.exists(OUT):
        try:
            os.remove(OUT)
        except OSError:
            pass
    w.start_erase()
    done = wait_until(lambda: _finished.get("done") or _finished.get("err"), timeout_ms=300000)
    if not done:
        # 直接等线程
        if w.worker:
            w.worker.wait(300000)
    check("擦除任务结束", bool(_finished), str(_finished))
    check("输出文件存在", os.path.exists(OUT) and os.path.getsize(OUT) > 1024,
          f"{os.path.getsize(OUT)/1048576:.2f} MB" if os.path.exists(OUT) else "无")

    if os.path.exists(OUT):
        om = probe(OUT)
        check("输出帧数一致", abs(om.nb_frames - w.meta.nb_frames) <= 1,
              f"{om.nb_frames} vs {w.meta.nb_frames}")

    # 工程文件落盘 + 重新打开自动载入
    check("工程已保存", os.path.exists(pj), os.path.basename(pj))
    if os.path.exists(pj):
        n0 = len(w.project.boxes)
        w.open_video(VIDEO)
        app.processEvents()
        check("重新打开自动载入框", len(w.project.boxes) == n0, f"{len(w.project.boxes)} vs {n0}")

    print("\n" + ("=== ALL PASS ===" if ok_all else "=== 存在 FAIL ==="))
    return 0 if ok_all else 1


# 打补丁：把 worker 结束回调接进来
_orig_main = MainWindow


class _Patched(_orig_main):
    def _on_finished(self, stats):
        _finished["done"] = stats
        super()._on_finished(stats)

    def _on_failed(self, err):
        _finished["err"] = err
        super()._on_failed(err)


if __name__ == "__main__":
    MainWindow = _Patched  # noqa: F811
    sys.exit(main())
