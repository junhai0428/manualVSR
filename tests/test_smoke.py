# -*- coding: utf-8 -*-
"""自测：探测 → seek 精度 → 引擎 batch 推理 → 端到端擦除。

运行:  <MAIN_ENV>\Scripts\python.exe tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.video_io import probe, FrameReader, count_frames_exact, nvenc_available  # noqa: E402
from src.project import Project, Box, Settings, default_output_path  # noqa: E402
from src.inpaint_engine import InpaintEngine  # noqa: E402
from src.eraser import EraseTask, plan_segments, merged_crop  # noqa: E402
from tools.make_test_video import gen, decode_index_block  # noqa: E402

VIDEO = r"E:\tmp\suberase_test\test_sub.mp4"
OUT = r"E:\tmp\suberase_test\test_sub_clean.mp4"

ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))


def test_probe() -> tuple:
    if not os.path.exists(VIDEO):
        gen(VIDEO, seconds=4)
    m = probe(VIDEO)
    check("probe 元信息", m.width == 1280 and m.height == 720 and abs(m.fps - 30) < 0.01,
          f"{m.width}x{m.height} fps={m.fps} frames={m.nb_frames} audio={m.has_audio}")
    check("总帧数", m.nb_frames in (119, 120, 121), f"nb_frames={m.nb_frames} exact={m.nb_frames_exact}")
    return m,


def test_seek(m) -> None:
    r = FrameReader(VIDEO, m)
    bad = []
    for idx in [0, 1, 5, 17, 33, 60, 61, 90, 118, 119]:
        f = r.read(idx)
        if f is None:
            bad.append(f"{idx}:None")
            continue
        got = decode_index_block(f)
        if got != idx:
            bad.append(f"{idx}->{got}")
    # 顺序回退再读
    f = r.read(40)
    got40 = decode_index_block(f)
    if got40 != 40:
        bad.append(f"back40->{got40}")
    r.release()
    check("seek 帧精度", not bad, f"偏差: {bad if bad else '无'}")


def test_engine() -> InpaintEngine:
    t0 = time.time()
    eng = InpaintEngine()
    check("引擎加载", eng.torch_model is not None,
          f"{eng.summary()} 加载耗时 {time.time()-t0:.1f}s")

    n, hh, ww = 4, 160, 640
    img = (np.random.rand(n, hh, ww, 3) * 60 + 180).astype(np.uint8)
    msk = np.zeros((n, hh, ww), np.uint8)
    msk[:, 60:110, 100:500] = 255
    t1 = time.time()
    res = eng.inpaint_batch(img, msk)
    dt = time.time() - t1
    check("batch 输出形状", res.shape == img.shape, str(res.shape))
    check("输出非空", int(res.mean()) > 10, f"mean={res.mean():.1f}")
    check("GPU 推理", "cuda" in eng.stats["device"],
          f"{n}帧耗时 {dt*1000:.0f}ms ({dt*1000/n:.0f}ms/帧) device={eng.stats['device']} "
          f"vram_free={eng.stats['vram_free_gb']}GB")
    # 遮蔽区域应被填充（不再是原值）
    diff = np.abs(res[:, 60:110, 100:500].astype(int) - img[:, 60:110, 100:500].astype(int)).mean()
    check("遮蔽区被修复", diff > 3, f"mask 区平均差 {diff:.1f}")
    return eng


def test_plan() -> None:
    boxes = [Box(x=0, y=0, w=10, h=10, start=10, end=20),
             Box(x=0, y=0, w=10, h=10, start=15, end=30)]
    segs = plan_segments(40, boxes)
    check("区间切分", segs[0] == (0, 10, []) and any(s[2] and s[0] == 10 for s in segs),
          str([(a, b, len(c)) for a, b, c in segs]))
    crop = merged_crop(boxes, 48, 1280, 720)
    check("crop 区域", crop[2] - crop[0] >= 256 and crop[3] - crop[1] >= 160, str(crop))


def test_e2e(m, eng) -> None:
    p = Project(video=VIDEO, meta={"fps": m.fps})
    p.boxes = [Box(x=200, y=560, w=880, h=120, start=0, end=m.nb_frames - 1)]
    p.settings = Settings(margin=48, dilate=3, encoder="auto", keep_audio=True)
    p.output = default_output_path(VIDEO)

    if os.path.exists(OUT):
        try:
            os.remove(OUT)
        except OSError:
            pass
    task = EraseTask(p, m, OUT, eng)
    t0 = time.time()
    stats = task.run(on_progress=lambda d, t, msg: None)
    check("端到端完成", stats["frames"] >= m.nb_frames - 1 and not stats["error"],
          f"{stats['frames']}帧 擦除{stats['inpainted']}帧 用时{stats['seconds']}s "
          f"编码器={stats['encoder']} err={stats['error']}")

    if not os.path.exists(OUT):
        check("输出文件", False, OUT)
        return
    size_mb = os.path.getsize(OUT) / 1048576
    # 测试片画面极简（渐变+纯色块），码率天然很低，只做下限校验
    check("输出文件", size_mb > 0.01, f"{size_mb:.2f} MB")

    om = probe(OUT)
    check("输出帧数/时长", abs(om.duration - m.duration) < 0.5 and om.nb_frames >= m.nb_frames - 2,
          f"src={m.duration:.2f}s/{m.nb_frames}  out={om.duration:.2f}s/{om.nb_frames}")
    check("音频保留", om.has_audio, f"has_audio={om.has_audio}")

    rs, ro = FrameReader(VIDEO, m), FrameReader(OUT, om)
    fs, fo = rs.read(30), ro.read(30)
    rs.release(); ro.release()
    if fs is None or fo is None:
        check("抽帧对比", False, "读帧失败")
        return
    sub_diff = np.abs(fs[560:680].astype(int) - fo[560:680].astype(int)).mean()
    top_diff = np.abs(fs[0:400].astype(int) - fo[0:400].astype(int)).mean()
    check("字幕区被擦除", sub_diff > 15, f"字幕区平均差 {sub_diff:.1f}")
    check("非字幕区保留", top_diff < 12, f"上部平均差 {top_diff:.1f}")


def main() -> int:
    print(f"ffmpeg/自测视频: {VIDEO}")
    (m,) = test_probe()
    test_seek(m)
    eng = test_engine()
    test_plan()
    test_e2e(m, eng)
    print("\n" + ("=== ALL PASS ===" if ok_all else "=== 存在 FAIL ==="))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
