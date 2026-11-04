# -*- coding: utf-8 -*-
r"""自测：时间轴分轨道 + 文字自动检测（合并逻辑/工程格式/真实素材集成）。

运行:  <MAIN_ENV>\Scripts\python.exe tests/test_detect.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))


def test_track_assignment() -> None:
    """分轨道：时间上重叠的框分到不同轨道，不重叠的共用一轨。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from src.ui.timeline import TimelineWidget, MAX_TRACKS, TRACK_H

    app = QApplication.instance() or QApplication([])

    def mkbox(start, end):
        from src.project import Box
        return Box(x=0, y=0, w=10, h=10, start=start, end=end)

    t = TimelineWidget()
    t.set_meta(100)

    # 不重叠 → 同一轨道
    t.set_boxes([mkbox(0, 10), mkbox(20, 30), mkbox(40, 50)])
    check("不重叠框同轨", len(t._tracks) == 1 and len(t._tracks[0]) == 3,
          f"tracks={len(t._tracks)}")

    # 完全重叠 → 两轨
    t.set_boxes([mkbox(0, 50), mkbox(10, 20)])
    check("重叠框分轨", len(t._tracks) == 2,
          f"tracks={[(i, lo, hi) for tr in t._tracks for i, lo, hi in tr]}")

    # 半开放框（start=0, end=末帧）与任何框重叠 → 各占一轨
    t.set_boxes([mkbox(0, 99), mkbox(0, 99), mkbox(30, 40)])
    check("半开放框占独立轨道", len(t._tracks) == 3, f"tracks={len(t._tracks)}")

    # 超上限 → 溢出列表，最后一轨叠画
    t.set_boxes([mkbox(i * 5, i * 5 + 60) for i in range(MAX_TRACKS + 2)])
    check("轨道上限溢出", len(t._tracks) == MAX_TRACKS
          and len(t._overflow) == 2,
          f"tracks={len(t._tracks)} overflow={len(t._overflow)}")

    # 高度随轨道数变化
    t.set_boxes([mkbox(0, 10), mkbox(20, 30)])
    h1 = t.height()
    t.set_boxes([mkbox(0, 10), mkbox(0, 10)])
    h2 = t.height()
    check("高度随轨道数自适应", h2 > h1, f"{h1} -> {h2}px")
    check("轨道高度公式", h1 == 8 + 1 * TRACK_H + 8, f"单轨高={h1}")
    app.processEvents()


def test_selection_ui() -> None:
    """框变多后的选中体验：时间轴点色块选中 + 面板筛选。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication
    from src.project import Project, Box
    from src.ui.timeline import TimelineWidget
    from src.ui.box_panel import BoxPanel

    app = QApplication.instance() or QApplication([])

    # 时间轴：点击轨道色块命中对应框，空白处不命中
    t = TimelineWidget()
    t.set_meta(100)
    t.resize(400, 60)
    boxes = [Box(x=0, y=0, w=10, h=10, start=0, end=40),
             Box(x=0, y=0, w=10, h=10, start=0, end=40),
             Box(x=0, y=0, w=10, h=10, start=50, end=60)]
    t.set_boxes(boxes)
    t._assign_tracks()
    check("重叠框分两轨", len(t._tracks) == 2)
    # 第 0 轨第 1 个框（start=0）中点
    r0 = t._box_rect(*t._tracks[0][0][1:], 0)
    hit0 = t._box_at(r0.center())
    check("时间轴命中轨道0框", hit0 == boxes[0].id, f"{hit0}")
    r1 = t._box_rect(*t._tracks[1][0][1:], 1)
    hit1 = t._box_at(r1.center())
    check("时间轴命中轨道1框", hit1 == boxes[1].id, f"{hit1}")
    check("时间轴空白不命中", t._box_at(QPoint(t.width() - 5, t.height() - 2)) is None)

    # 面板筛选：过滤后行→id 映射仍正确，选中被隐藏的框会清筛选
    panel = BoxPanel()
    p = Project(video="v.mp4")
    p.boxes = [Box(x=0, y=0, w=10, h=10, start=0, end=90, label=""),
               Box(x=0, y=0, w=10, h=10, start=10, end=20, label="自动"),
               Box(x=0, y=0, w=10, h=10, start=30, end=40, label="自动")]
    panel.set_project(p, 100)
    check("面板未筛选行数", panel.table.rowCount() == 3)
    panel.ed_filter.setText("自动")
    check("筛选后行数", panel.table.rowCount() == 2,
          f"rows={panel.table.rowCount()}")
    panel.table.setCurrentCell(0, 0)
    check("筛选后选中映射", panel.selected_box_id() == p.boxes[1].id)
    panel.ed_filter.clear()
    panel.ed_filter.setText("50")
    panel.select_box(p.boxes[0].id)          # 被筛选隐藏 → 应清筛选并选中
    check("选中隐藏框清筛选", panel.ed_filter.text() == ""
          and panel.selected_box_id() == p.boxes[0].id)
    panel.set_frame_idx(35)
    panel.ck_active.setChecked(True)
    check("仅当前帧生效过滤", panel.table.rowCount() == 2,
          f"rows={panel.table.rowCount()}")   # 35 帧: 框0(0-90)、框2(30-40) 生效
    app.processEvents()


def test_merge_logic() -> None:
    """检测框合并：位置相近且时间连续的并入同簇。"""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from auto_detect import iou, merge_detections, clusters_to_boxes, rect_of

    check("IoU 相同矩形=1", iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0)
    check("IoU 不相交=0", iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0)

    dets = [
        {"frame": 0, "rect": (100, 500, 300, 40)},     # 同一字幕连续出现
        {"frame": 15, "rect": (102, 501, 298, 42)},
        {"frame": 30, "rect": (101, 500, 300, 41)},
        {"frame": 200, "rect": (100, 500, 300, 40)},   # 时间断裂 → 新簇
        {"frame": 45, "rect": (100, 100, 200, 30)},    # 位置不同 → 新簇
    ]
    clusters = merge_detections(dets, iou_thr=0.3, gap_frames=30)
    check("合并成 3 簇", len(clusters) == 3,
          str([(c["first"], c["last"], c["hits"]) for c in clusters]))
    c0 = clusters[0]
    check("同簇时间范围", c0["first"] == 0 and c0["last"] == 30 and c0["hits"] == 3,
          f"{c0['first']}-{c0['last']} hits={c0['hits']}")

    boxes = clusters_to_boxes(clusters, step=15, margin=8,
                              nb_frames=439, width=720, height=1280)
    # 结果按 (y, x) 排序；字幕簇 y=500 是 boxes 里 y 最大的一个
    b0 = next(b for b in boxes if b["y"] == 492)
    check("簇→框 外扩与钳制", b0["x"] == 92 and b0["w"] == 317 and b0["h"] == 59,
          str(b0))
    check("簇→框 时间扩展", b0["start"] == 0 and b0["end"] == 45, str(b0))
    all_in = all(0 <= b["x"] and b["x"] + b["w"] <= 720
                 and 0 <= b["y"] and b["y"] + b["h"] <= 1280 for b in boxes)
    check("框都在画面内", all_in)
    check("rect_of 四点取包围", rect_of([[10, 20], [50, 20], [50, 40], [10, 40]])
          == (10, 20, 40, 20))


def test_project_format() -> None:
    """检测结果 → 工程 json：格式与手动工程完全一致。"""
    from src.workers import build_project_from_detect
    from src.project import Project

    result = {
        "video": "X:/nonexist/demo.mp4", "fps": 30.0, "nb_frames": 100,
        "nb_frames_exact": True, "width": 720, "height": 1280,
        "step_frames": 15, "sampled": 7, "providers": ["CUDAExecutionProvider"],
        "det_seconds": 1.0,
        "boxes": [{"x": 100, "y": 500, "w": 300, "h": 40, "start": 0, "end": 45},
                  {"x": -50, "y": 1200, "w": 400, "h": 200, "start": 50, "end": 200}],
    }
    p = build_project_from_detect("X:/nonexist/demo.mp4", result)
    b1 = p.boxes[1]
    check("越界框被钳制", b1.x >= 0 and b1.y + b1.h <= 1280 and b1.end <= 99,
          f"({b1.x},{b1.y},{b1.w},{b1.h}) {b1.start}-{b1.end}")
    check("检测框带「自动」标记", all(b.label == "自动" for b in p.boxes))

    d = p.to_dict()
    check("工程字段齐全", set(d.keys()) == {"version", "video", "output", "meta",
                                            "boxes", "settings"},
          str(sorted(d.keys())))
    check("meta 结构不变", set(d["meta"].keys()) >= {"fps", "width", "height",
                                                     "duration", "nb_frames"},
          str(sorted(d["meta"].keys())))
    # 存盘 → 回读一致（标准 load 通道）
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "suberase_test_gen.json")
    p.save(tmp)
    p2 = Project.load(tmp)
    check("回读框一致", [(b.x, b.y, b.w, b.h, b.start, b.end)
                         for b in p2.boxes] ==
          [(b.x, b.y, b.w, b.h, b.start, b.end) for b in p.boxes],
          f"{len(p2.boxes)} boxes")
    check("回读 meta 一致", p2.meta.get("nb_frames") == 100
          and p2.meta.get("width") == 720)
    os.remove(tmp)


def test_real_material() -> None:
    """真实素材集成：调 auto_detect 子进程（<OCR_ENV>），核对结果与工程落盘。"""
    from src.workers import run_detect_subprocess, build_project_from_detect

    video = os.environ.get("SE_TEST_VIDEO") or \
        r"<PROJECT_DIR>\tests\douyin_012.mp4"
    if not os.path.exists(video):
        check("真实素材存在", False, video)
        return
    check("真实素材存在", True, video)

    frames = []
    result = run_detect_subprocess(video, 0.5, 0.3, 8,
                                   on_progress=frames.append)
    check("子进程检测完成", len(result.get("boxes", [])) > 0,
          f"{result.get('sampled')} 采样帧 / {result.get('det_seconds')}s / "
          f"{len(result.get('boxes', []))} 框 / providers={result.get('providers')}")
    check("检测跑在 GPU", "CUDAExecutionProvider" in result.get("providers", []),
          str(result.get("providers")))
    check("阈值参数随结果返回", result.get("thresh") == 0.3
          and result.get("box_thresh") == 0.5,
          f"thresh={result.get('thresh')} box_thresh={result.get('box_thresh')}")
    check("进度回调有输出", len(frames) > 0, f"{len(frames)} 次进度")

    nb = int(result.get("nb_frames") or 0)
    good = all(0 <= b["start"] <= b["end"] <= nb - 1
               and 0 <= b["x"] and b["x"] + b["w"] <= result["width"]
               and 0 <= b["y"] and b["y"] + b["h"] <= result["height"]
               for b in result["boxes"])
    check("框与时间均在范围内", good, f"nb_frames={nb}")

    # 生成工程并校验格式
    import tempfile
    from src.project import Project, default_project_path
    p = build_project_from_detect(video, result)
    tmp = os.path.join(tempfile.gettempdir(), "suberase_test_real.suberase.json")
    p.save(tmp)
    p2 = Project.load(tmp)
    check("真实工程回读一致", len(p2.boxes) == len(p.boxes)
          and all(b.label == "自动" for b in p2.boxes),
          f"{os.path.basename(tmp)}")
    os.remove(tmp)


def test_frame_click_select() -> None:
    """回归：画面上点击框必须真的选中（selection_changed 发 id 字符串）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from PySide6.QtCore import QPointF, QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    from src.project import Box
    from src.ui.frame_view import FrameView

    app = QApplication.instance() or QApplication([])
    fv = FrameView()
    fv.resize(400, 300)
    fv.set_frame(np.zeros((100, 200, 3), np.uint8))   # 视频 200x100
    fv.update_view_rect()

    b1 = Box(x=20, y=20, w=60, h=30, start=0, end=99)
    b2 = Box(x=25, y=22, w=60, h=30, start=0, end=99)   # 与 b1 重叠
    fv.set_boxes([b1, b2])

    got = []
    fv.selection_changed.connect(got.append)

    def click(x, y):
        p = QPointF(x, y)
        fv.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, p,
                                       Qt.MouseButton.LeftButton,
                                       Qt.MouseButton.LeftButton,
                                       Qt.KeyboardModifier.NoModifier))
        fv.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, p,
                                         Qt.MouseButton.LeftButton,
                                         Qt.MouseButton.NoButton,
                                         Qt.KeyboardModifier.NoModifier))

    wp = fv.to_widget(30, 25)     # 两个框重叠的区域内
    click(wp.x(), wp.y())
    check("点击框选中(发id字符串)", got and got[-1] == b2.id
          and fv.selected_id == b2.id, f"got={got[-1] if got else None}")
    click(wp.x(), wp.y())         # 原地再点 → 轮选到 b1
    check("原地再点轮选", got[-1] == b1.id and fv.selected_id == b1.id,
          f"got={got[-1]}")
    # 点击空白 → 不选中（进入新建模式，不产生新框）
    n_before = len(fv.boxes)
    corner = fv.to_widget(190, 90)
    click(corner.x(), corner.y())
    check("点击框外不误选", fv.selected_id == b1.id or fv.selected_id is None,
          f"sel={fv.selected_id}")
    check("点击空白不建框", len(fv.boxes) == n_before)
    app.processEvents()


def main() -> int:
    test_track_assignment()
    test_selection_ui()
    test_frame_click_select()
    test_merge_logic()
    test_project_format()
    test_real_material()
    print("\n" + ("=== ALL PASS ===" if ok_all else "=== 存在 FAIL ==="))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
