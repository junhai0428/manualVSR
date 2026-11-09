# -*- coding: utf-8 -*-
"""批量擦除自测：同一真实素材 + 两个不同时段的工程，串行跑 + 断点续跑。

运行:  <你的 Python 解释器> tests/test_batch.py
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

from src.project import Project, Box, Settings  # noqa: E402
from src.inpaint_engine import InpaintEngine  # noqa: E402
from src.ui.batch_dialog import BatchDialog  # noqa: E402
from src.video_io import probe  # noqa: E402
from src.env import test_video  # noqa: E402

# 真实素材：SE_TEST_VIDEO / local_config.json 的 test_video / tests/douyin_012.mp4
VIDEO = test_video()
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")
J1 = os.path.join(OUTDIR, "batch_a.suberase.json")
J2 = os.path.join(OUTDIR, "batch_b.suberase.json")
O1 = os.path.join(OUTDIR, "batch_a_clean.mp4")
O2 = os.path.join(OUTDIR, "batch_b_clean.mp4")

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
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return res.get("args")


def make_project(path: str, y0: float, y1: float, t0: int, t1: int, out: str) -> str:
    """同一素材不同字幕带/时段 -> 不同工程。y0/y1 为高度比例。"""
    meta = probe(VIDEO)
    p = Project(video=VIDEO, meta={"fps": meta.fps}, output=out)
    p.boxes = [Box(x=int(meta.width * 0.05), y=int(meta.height * y0),
                   w=int(meta.width * 0.9), h=int(meta.height * (y1 - y0)),
                   start=t0, end=t1)]
    p.settings = Settings()
    return p.save(path)


def main() -> int:
    if not os.path.exists(VIDEO):
        print(f"[SKIP] 未找到测试素材: {VIDEO}\n"
              f"       用环境变量 SE_TEST_VIDEO 或 local_config.json 的 test_video 指定")
        return 0
    os.makedirs(OUTDIR, exist_ok=True)
    for f in (O1, O2, J1, J2):
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
    check("素材存在", True, VIDEO)
    meta = probe(VIDEO)
    check("素材信息", meta.nb_frames > 100, f"{meta.width}x{meta.height} {meta.nb_frames}帧")

    # 两个工程：不同字幕带 + 不同时段
    j1 = make_project(J1, 0.47, 0.62, 40, 100, O1)
    j2 = make_project(J2, 0.47, 0.62, 150, 200, O2)
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
    check("输出路径取自工程", dlg.items[0].out == O1 and dlg.items[1].out == O2,
          f"{dlg.items[0].out}, {dlg.items[1].out}")

    dlg.start()
    args = wait_signal(dlg.worker.finished_all)
    summary = args[0] if args else {}
    check("批量完成", summary.get("ok") == 2 and summary.get("fail") == 0, str(summary))
    check("只擦时段内帧", summary.get("inpainted") == (100 - 40 + 1) + (200 - 150 + 1),
          f"inpainted={summary.get('inpainted')} 期望=112")
    check("输出 A", os.path.exists(O1) and os.path.getsize(O1) > 1024,
          f"{os.path.getsize(O1)/1024:.0f} KB" if os.path.exists(O1) else "无")
    check("输出 B", os.path.exists(O2) and os.path.getsize(O2) > 1024,
          f"{os.path.getsize(O2)/1024:.0f} KB" if os.path.exists(O2) else "无")

    for out in (O1, O2):
        if os.path.exists(out):
            om = probe(out)
            check(f"帧数一致 {os.path.basename(out)}",
                  abs(om.nb_frames - meta.nb_frames) <= 1,
                  f"{om.nb_frames} vs {meta.nb_frames}")

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
