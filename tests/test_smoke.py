# -*- coding: utf-8 -*-
r"""自测：探测 → seek 精度 → 引擎 batch 推理 → 端到端擦除。

运行:  <MAIN_ENV>\Scripts\python.exe tests/test_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.video_io import probe, FrameReader, count_frames_exact, nvenc_available  # noqa: E402
from src.project import Project, Box, Settings, default_output_path  # noqa: E402
from src.inpaint_engine import InpaintEngine  # noqa: E402
from src.eraser import EraseTask, plan_segments, merged_crop  # noqa: E402

# 真实素材（可用环境变量 SE_TEST_VIDEO 覆盖），不自造测试视频
VIDEO = os.environ.get("SE_TEST_VIDEO") or r"<PROJECT_DIR>\tests\douyin_012.mp4"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")
OUT = os.path.join(OUTDIR, "douyin_012_clean.mp4")

ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))


def test_probe() -> tuple:
    check("素材存在", os.path.exists(VIDEO), VIDEO)
    m = probe(VIDEO)
    check("probe 元信息", m.width > 0 and m.height > 0 and m.fps > 0,
          f"{m.width}x{m.height} fps={m.fps} frames={m.nb_frames} "
          f"audio={m.has_audio} exact={m.nb_frames_exact}")
    return m,


def test_seek(m) -> None:
    """真实素材没有帧号水印，用「跳转读 vs 顺序读」逐像素一致性校验 seek 精度。"""
    idxs = sorted({0, 1, 5, 17, 33, 60, 61, 100, 200, 300, m.nb_frames - 2, m.nb_frames - 1})
    idxs = [i for i in idxs if 0 <= i < m.nb_frames]

    # 基准：一次顺序读到底（不触发任何跳转）
    base_reader = FrameReader(VIDEO, m)
    base = {}
    for i in range(0, max(idxs) + 1):
        f = base_reader.read(i)
        if f is None:
            break
        if i in idxs:
            base[i] = f.copy()
    base_reader.release()

    r = FrameReader(VIDEO, m)
    bad = []
    for i in idxs:
        f = r.read(i)                      # 随机跳转（必要时重启 -ss 管道）
        if f is None:
            bad.append(f"{i}:None")
            continue
        if i not in base:
            bad.append(f"{i}:no-base")
            continue
        d = float(np.abs(f.astype(int) - base[i].astype(int)).mean())
        if d > 0.01:
            bad.append(f"{i}:diff={d:.2f}")
    # 回退读
    back = r.read(idxs[0])
    d = float(np.abs(back.astype(int) - base[idxs[0]].astype(int)).mean())
    if d > 0.01:
        bad.append(f"back{d}")
    r.release()
    check("seek 帧精度(与顺序读一致)", not bad, f"偏差: {bad if bad else f'无 ({len(idxs)}个采样点)'}")


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

    # 半开放框：出点未定 -> 生效到末尾；入点未定 -> 从开头生效
    ob = Box(x=0, y=0, w=10, h=10, start=40, end=40, open_end=True)
    segs = plan_segments(60, [ob])
    check("出点未定切分", segs == [(0, 40, []), (40, 60, [ob])],
          str([(a, b, len(c)) for a, b, c in segs]))
    check("出点未定 covers", ob.covers(0) is False and ob.covers(59) is True,
          ob.span_text())
    hb = Box(x=0, y=0, w=10, h=10, start=40, end=40, open_start=True)
    check("入点未定 covers", hb.covers(0) is True and hb.covers(59) is False, hb.span_text())
    check("span 展开", ob.span(60) == (40, 59) and hb.span(60) == (0, 40),
          f"{ob.span(60)} {hb.span(60)}")


def test_e2e(m, eng) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    t0, t1 = 40, 100          # 字幕段，只擦这一段，其余帧直通
    bw = int(m.width * 0.9)
    bh = max(60, int(m.height * 0.10))
    bx = (m.width - bw) // 2
    # 自动定位字幕行：取时段中心帧，找白色文字像素最密集的水平带（不靠目测）
    fr = FrameReader(VIDEO, m)
    f_mid = fr.read((t0 + t1) // 2)
    fr.release()
    hsv = cv2.cvtColor(f_mid, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 210)).astype(np.uint8)
    rowsum = white.sum(axis=1)
    by = int(np.convolve(rowsum, np.ones(bh, int), mode="valid").argmax())
    check("自动定位字幕行", int(rowsum[by:by + bh].sum()) > 500,
          f"y={by}-{by + bh} 白像素={int(rowsum[by:by + bh].sum())}")
    p = Project(video=VIDEO, meta={"fps": m.fps})
    p.boxes = [Box(x=bx, y=by, w=bw, h=bh, start=t0, end=t1)]
    p.settings = Settings(margin=48, dilate=3, encoder="auto", keep_audio=True)
    p.output = OUT
    print(f"[e2e] 框: x={bx} y={by} w={bw} h={bh} 时段{t0}-{t1} ({m.width}x{m.height})")

    if os.path.exists(OUT):
        try:
            os.remove(OUT)
        except OSError:
            pass
    task = EraseTask(p, m, OUT, eng)
    stats = task.run(on_progress=lambda d, t, msg: None)
    check("端到端完成", stats["frames"] >= m.nb_frames - 1 and not stats["error"],
          f"{stats['frames']}帧 擦除{stats['inpainted']}帧 用时{stats['seconds']}s "
          f"编码器={stats['encoder']} err={stats['error']}")
    check("只擦时段内帧", stats["inpainted"] == t1 - t0 + 1,
          f"inpainted={stats['inpainted']} 期望={t1 - t0 + 1}")

    if not os.path.exists(OUT):
        check("输出文件", False, OUT)
        return
    size_mb = os.path.getsize(OUT) / 1048576
    check("输出文件", size_mb > 0.1, f"{size_mb:.2f} MB")

    om = probe(OUT)
    check("输出帧数/时长", abs(om.duration - m.duration) < 0.5 and om.nb_frames >= m.nb_frames - 2,
          f"src={m.duration:.2f}s/{m.nb_frames}  out={om.duration:.2f}s/{om.nb_frames}")
    if m.has_audio:
        check("音频保留", om.has_audio, f"has_audio={om.has_audio}")
    else:
        check("源无音轨(跳过音频断言)", True, "源无音轨")

    # 抽帧对比：时段内帧的字幕应被擦掉；时段外帧应原样直通
    rs, ro = FrameReader(VIDEO, m), FrameReader(OUT, om)
    fs_in, fo_in = rs.read(60), ro.read(60)         # 时段内（有字幕）
    fs_out, fo_out = rs.read(300), ro.read(300)     # 时段外（直通）
    rs.release(); ro.release()
    if fs_in is None or fo_in is None or fs_out is None or fo_out is None:
        check("抽帧对比", False, "读帧失败")
        return

    def changed_ratio(a, b, thr=40):
        d = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
        return float((d > thr).mean())

    box_in = changed_ratio(fs_in[by:by + bh], fo_in[by:by + bh])
    top_h = max(1, int(m.height * 0.10))
    top_ratio = changed_ratio(fs_in[0:top_h], fo_in[0:top_h])
    check("时段内字幕被擦除", box_in > max(0.02, top_ratio * 3),
          f"框内显著变化像素 {box_in:.2%} vs 框外 {top_ratio:.2%}")

    # 最直接的产品判据：框内的白色文字像素应当消失
    def white_count(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return int(((hsv[..., 1] < 60) & (hsv[..., 2] > 210)).sum())

    w_src, w_out = white_count(fs_in[by:by + bh]), white_count(fo_in[by:by + bh])
    check("白色文字像素消失", w_out < max(50, w_src * 0.15),
          f"src={w_src} out={w_out}")
    top_diff = float(np.abs(fs_in[0:top_h].astype(int) - fo_in[0:top_h].astype(int)).mean())
    check("框外基本不变", top_diff < 12, f"上部平均差 {top_diff:.1f}")
    # 时段外直通：仅有编码噪声
    pass_diff = float(np.abs(fs_out[by:by + bh].astype(int) - fo_out[by:by + bh].astype(int)).mean())
    check("时段外帧直通", pass_diff < 8, f"框内平均差 {pass_diff:.1f}（应只剩编码噪声）")


def main() -> int:
    print(f"测试素材: {VIDEO}")
    (m,) = test_probe()
    test_seek(m)
    eng = test_engine()
    test_plan()
    test_e2e(m, eng)
    print("\n" + ("=== ALL PASS ===" if ok_all else "=== 存在 FAIL ==="))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
