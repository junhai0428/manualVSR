# -*- coding: utf-8 -*-
r"""自测：时域稳定 —— 向内羽化 / 全局运动估计 / 补偿 EMA / 真实背景优选 / 旧工程兼容。

运行:  <你的解释器> tests/test_temporal.py
纯逻辑测试，不加载模型、不需要视频素材。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.eraser import (feather_alpha, estimate_shift, warp_shift,  # noqa: E402
                        bg_usable, composite, find_clean_ref, fade_blend,
                        BG_FADE_FRAMES, MIN_PHASE_RESP)
from src.project import Project, Settings  # noqa: E402

ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and cond
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))


def make_mask(h=80, w=120, pad=20):
    m = np.zeros((h, w), np.uint8)
    m[pad:h - pad, pad:w - pad] = 255
    return m


def test_feather() -> None:
    m = make_mask()
    a = feather_alpha(m, 3)[:, :, 0]
    check("羽化：mask 外恒 0", float(a[m == 0].max()) == 0.0)
    check("羽化：mask 中心恒 1", float(a[40, 60]) == 1.0)
    edge = float(a[20, 60])       # 边界内侧第一行
    check("羽化：边缘渐变更柔和", 0.0 <= edge < 1.0, f"edge={edge:.3f}")
    hard = feather_alpha(m, 0)[:, :, 0]
    check("羽化=0 等价硬替换", float(hard[m > 0].min()) == 1.0
          and float(hard[m == 0].max()) == 0.0)


def test_motion() -> None:
    rng = np.random.default_rng(7)
    prev = (rng.random((240, 320)) * 255).astype(np.uint8)
    cur = warp_shift(prev, 5.0, 3.0)
    dx, dy, resp = estimate_shift(prev, cur)
    check("相位相关估计出位移", abs(dx - 5.0) < 0.5 and abs(dy - 3.0) < 0.5,
          f"({dx:.2f},{dy:.2f}) resp={resp:.3f}")
    check("相位相关峰值可信", resp > 0.3, f"resp={resp:.3f}")
    dx2, dy2, _ = estimate_shift(prev, cur, (90, 150))
    check("排除字幕带仍能估计", abs(dx2 - 5.0) < 0.5 and abs(dy2 - 3.0) < 0.5,
          f"({dx2:.2f},{dy2:.2f})")
    check("warp 方向与估计同向",
          float(np.abs(warp_shift(prev, dx, dy).astype(np.int16)
                       - cur.astype(np.int16)).mean()) < 3.0)

    other = (rng.random((240, 320)) * 255).astype(np.uint8)
    _dx, _dy, r2 = estimate_shift(prev, other)
    check("无关帧运动估计失败", r2 < MIN_PHASE_RESP or (_dx == 0 and _dy == 0),
          f"resp={r2:.3f} shift=({_dx},{_dy})")


def test_composite() -> None:
    m = make_mask()
    orig = np.full((80, 120, 3), 10, np.uint8)
    fill = np.full((80, 120, 3), 200, np.uint8)
    hard = composite(orig, fill, feather_alpha(m, 0))
    check("合成：mask 外保持原图", bool((hard[m == 0] == 10).all()))
    check("合成：mask 内用填充", bool((hard[m > 0] == 200).all()))
    soft = composite(orig, fill, feather_alpha(m, 4))
    edge_val = int(soft[20, 60, 0])
    check("合成：羽化边缘介于两者之间", 10 < edge_val < 200, f"edge={edge_val}")


def test_bg_usable() -> None:
    m = make_mask()
    cur = np.full((80, 120, 3), 60, np.uint8)
    check("真实背景：对齐一致时可用", bg_usable(cur.copy(), cur, m))
    noisy = cur.copy()
    noisy[m == 0] = 200          # mask 外完全不对齐
    check("真实背景：对齐差时拒绝", not bg_usable(noisy, cur, m))
    with_text = cur.copy()
    with_text[30:50, 30:90] = (255, 255, 255)   # mask 内残留白字
    check("真实背景：残留字幕时拒绝", not bg_usable(with_text, cur, m))


def test_compensated_ema() -> None:
    """复现摇移场景：补偿 EMA 应同时降低抖动、且不抬高修复误差。"""
    rng = np.random.default_rng(11)
    N, DX, DY, ALPHA = 8, 2, 1, 0.6
    base = (rng.random((80, 120, 3)) * 180 + 30).astype(np.float32)
    m = make_mask() > 0
    gt = np.stack([np.roll(base, (DY * t, DX * t), axis=(0, 1))
                   for t in range(N)])
    # 模拟逐帧独立修复：真实背景 + 每帧不同的生成噪声
    res = np.stack([np.clip(gt[t] + rng.normal(0, 9, gt[t].shape), 0, 255)
                    for t in range(N)])

    def align(a, t):
        return np.roll(a, (-DY * t, -DX * t), axis=(0, 1))

    def jitter(arr):
        return float(np.mean([np.abs(align(arr[t], t) - align(arr[t - 1], t - 1))[m].mean()
                              for t in range(1, N)]))

    def error(arr):
        return float(np.mean([np.abs(arr[t] - gt[t])[m].mean() for t in range(N)]))

    ema, mc = [res[0]], [res[0]]
    for t in range(1, N):
        ema.append((1 - ALPHA) * res[t] + ALPHA * ema[t - 1])
        prev_aligned = np.roll(mc[t - 1], (DY, DX), axis=(0, 1))
        mc.append((1 - ALPHA) * res[t] + ALPHA * prev_aligned)
    ema, mc = np.stack(ema), np.stack(mc)

    j_raw, j_ema, j_mc = jitter(res), jitter(ema), jitter(mc)
    e_raw, e_ema, e_mc = error(res), error(ema), error(mc)
    check("补偿 EMA 抖动低于逐帧", j_mc < j_raw * 0.75,
          f"{j_raw:.2f} -> {j_mc:.2f}")
    check("补偿 EMA 抖动低于屏幕系 EMA", j_mc < j_ema,
          f"屏幕系 {j_ema:.2f} vs 补偿 {j_mc:.2f}")
    check("屏幕系 EMA 抬高误差（拖影）", e_ema > e_raw,
          f"{e_raw:.2f} -> {e_ema:.2f}")
    check("补偿 EMA 不抬高误差", e_mc <= e_raw * 1.02,
          f"{e_raw:.2f} -> {e_mc:.2f}")


def test_fade_blend() -> None:
    b = 0.0
    for _ in range(BG_FADE_FRAMES):
        b = fade_blend(b, 1.0)
    check("填充源渐变需多帧才切满", abs(b - 1.0) < 1e-6, f"blend={b:.3f}")
    b = fade_blend(0.0, 1.0)
    check("单帧校验抖动不会立即切换", 0.0 < b < 1.0, f"blend={b:.3f}")
    b2 = fade_blend(b, 0.0)
    check("撤销时同样渐变", abs(b2 - (b - 1.0 / BG_FADE_FRAMES)) < 1e-6,
          f"{b:.3f} -> {b2:.3f}")


def test_clean_ref() -> None:
    check("参考帧落在字幕段外", find_clean_ref(102, [(0, 100)]) == 100)
    check("前 6 帧全有字幕则不选参考帧",
          find_clean_ref(101, [(0, 100)]) is None)
    check("段从 0 开始无参考帧", find_clean_ref(0, []) is None)
    check("跨多个字幕段查找", find_clean_ref(50, [(0, 10), (20, 30)]) == 48)


def test_settings_compat() -> None:
    old = {"version": 1, "video": "v.mp4", "output": "", "meta": {},
           "boxes": [], "settings": {"margin": 20, "dilate": 2,
                                     "batch_size": 0, "encoder": "auto",
                                     "crf": 18, "keep_audio": True, "model": "lama"}}
    p = Project.from_dict(old)
    s = p.settings
    check("旧工程缺字段用默认", s.temporal_alpha == 0.6 and s.motion_comp is True
          and s.feather_px == 3 and s.real_bg is True,
          f"alpha={s.temporal_alpha} feather={s.feather_px}")
    s.temporal_alpha = 0.3
    s.real_bg = False
    d = p.to_dict()
    check("新字段写入工程", d["settings"]["temporal_alpha"] == 0.3
          and d["settings"]["real_bg"] is False)
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "suberase_temporal_test.json")
    p.save(tmp)
    p2 = Project.load(tmp)
    check("新字段回读一致", p2.settings.temporal_alpha == 0.3
          and p2.settings.real_bg is False)
    os.remove(tmp)


def main() -> int:
    test_feather()
    test_motion()
    test_composite()
    test_bg_usable()
    test_compensated_ema()
    test_fade_blend()
    test_clean_ref()
    test_settings_compat()
    print("\n" + ("=== ALL PASS ===" if ok_all else "=== 存在 FAIL ==="))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
