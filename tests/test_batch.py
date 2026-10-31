# -*- coding: utf-8 -*-
"""批量擦除自测：多工程串行 + 断点续跑（跳过已存在输出）。

运行:  <MAIN_ENV>\\Scripts\\python.exe tests/test_batch.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.prepare_model import set_cache_env  # noqa: E402

set_cache_env()

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from src.project import Project, Box, Settings, default_project_path  # noqa: E402
from src.inpaint_engine import InpaintEngine  # noqa: E402
from src.ui.batch_dialog import BatchDialog  # noqa: E402
from src.video_io import probe  # noqa: E402
from tools.make_test_video import gen  # noqa: E402

DIR = r"E:\tmp\suberase_batch"
V1 = os.path.join(DIR, "clip_a.mp4")
V2 = os.path.join(DIR, "clip_b.mp4")
O1 = os.path.join(DIR, "clip_a_clean.mp4")
O2 = os.path.join(DIR, "clip_b_clean.mp4")

app = QApplication.instance() or QApplication(sys.argv)
ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))


def wait_signal(sig, timeout_ms=600000):
    loop = QEventLoop()
    res = {}

    def on_done(*args):
        res["args"] = args
        loop.quit()

    sig.connect(on_done)
    t = QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return res.get("args")


def make_project(video: str, w: int, h: int, sec: int, out: str) -> Project:
    if not os.path.exists(video):
        gen(video, w=w, h=h, seconds=sec, audio=True)
    meta = probe(video)
    p = Project(video=video, meta={"fps": meta.fps}, output=out)
    p.boxes = [Box(x=int(w * 0.15), y=int(h * 0.75), w=int(w * 0.7), h=int(h * 0.16),
                   start=0, end=meta.nb_frames - 1)]
    p.settings = Settings()
    return p


def main() -> int:
    os.makedirs(DIR, exist_ok=True)
    for f in (O1, O2, default_project_path(V1), default_project_path(V2)):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    # 两个不同分辨率/时长的片子，验证 meta 各自 probe
    p1 = make_project(V1, 1280, 720, 3, O1)
    p2 = make_project(V2, 960, 540, 2, O2)
    j1 = p1.save()
    j2 = p2.save()
    check("生成测试工程", os.path.exists(j1) and os.path.exists(j2),
          f"{os.path.basename(j1)}, {os.path.basename(j2)}")

    t0 = time.time()
    engine = InpaintEngine()
    check("引擎加载", engine.torch_model is not None, f"{engine.summary()} {time.time()-t0:.1f}s")

    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

    dlg = BatchDialog(engine)
    added = dlg.add_files([j1, j2])
    check("添加工程", added == 2 and len(dlg.items) == 2, f"added={added}")

    dlg.start()
    args = wait_signal(dlg.worker.finished_all)
    summary = args[0] if args else {}
    check("批量完成", summary.get("ok") == 2 and summary.get("fail") == 0, str(summary))
    check("输出 A", os.path.exists(O1) and os.path.getsize(O1) > 1024,
          f"{os.path.getsize(O1)/1024:.0f} KB" if os.path.exists(O1) else "无")
    check("输出 B", os.path.exists(O2) and os.path.getsize(O2) > 1024,
          f"{os.path.getsize(O2)/1024:.0f} KB" if os.path.exists(O2) else "无")

    for src, out in ((V1, O1), (V2, O2)):
        if os.path.exists(out):
            m, om = probe(src), probe(out)
            check(f"帧数一致 {os.path.basename(out)}",
                  abs(om.nb_frames - m.nb_frames) <= 1, f"{om.nb_frames} vs {m.nb_frames}")

    # 断点续跑：再跑一次应全部跳过且不重算
    mt1, mt2 = os.path.getmtime(O1), os.path.getmtime(O2)
    dlg2 = BatchDialog(engine)
    dlg2.add_files([j1, j2])
    dlg2.start()
    args2 = wait_signal(dlg2.worker.finished_all, timeout_ms=120000)
    s2 = args2[0] if args2 else {}
    check("二次运行全部跳过", s2.get("skipped") == 2 and s2.get("ok") == 0, str(s2))
    check("未覆盖已有产物", os.path.getmtime(O1) == mt1 and os.path.getmtime(O2) == mt2)

    print("\n" + ("=== ALL PASS ===" if ok_all else "=== 存在 FAIL ==="))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
