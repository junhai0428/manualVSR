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
                        adaptive_alpha, temporal_update, resample_carry,
                        carry_weight,
                        BG_FADE_FRAMES, MIN_PHASE_RESP, ALPHA_MIN_SHIFT,
                        ALPHA_FULL_SHIFT, CARRY_FADE, ANCHOR_MAX_PX)
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


def test_adaptive_alpha() -> None:
    check("自适应 α：静止镜头归零", adaptive_alpha(0.6, 0.0, 0.0) == 0.0)
    check("自适应 α：亚像素微动归零", adaptive_alpha(0.6, 0.3, 0.3) == 0.0,
          f"hypot={np.hypot(0.3, 0.3):.2f} < {ALPHA_MIN_SHIFT}")
    mid = adaptive_alpha(0.6, (ALPHA_MIN_SHIFT + ALPHA_FULL_SHIFT) / 2.0, 0.0)
    check("自适应 α：中段线性缩放", abs(mid - 0.3) < 1e-6, f"mid={mid:.3f}")
    check("自适应 α：位移足够用满额",
          abs(adaptive_alpha(0.6, 0.0, 5.0) - 0.6) < 1e-6)
    check("自适应 α：α=0 时恒为 0", adaptive_alpha(0.0, 9.0, 9.0) == 0.0)


def test_anchor_ema() -> None:
    """锚点系 EMA：亚像素插值不逐帧累积，长镜头不越擦越糊。"""
    rng = np.random.default_rng(23)
    N, DX, DY, ALPHA = 24, 1.3, 0.9, 0.6
    # 用平滑底图：真实画面相邻像素高度相关，不像白噪声那样一次亚像素插值就失真
    base = cv2.GaussianBlur((rng.random((90, 140, 3)) * 200 + 20).astype(np.float32),
                            (0, 0), 4.0)
    gt, res = [], []
    for t in range(N):
        M = np.float32([[1, 0, DX * t], [0, 1, DY * t]])
        gt.append(cv2.warpAffine(base, M, (140, 90), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE))
        res.append(gt[-1] + rng.normal(0, 1.0, base.shape).astype(np.float32))

    # 旧实现：对屏幕系已插值的结果连续 warp
    old = res[0].copy()
    for t in range(1, N):
        old = (1 - ALPHA) * res[t] + ALPHA * warp_shift(old, DX, DY)

    # 新实现：状态钉在锚点坐标系，每帧进出各一次
    state, acc_x, acc_y = res[0].copy(), 0.0, 0.0
    fill = res[0]
    for t in range(1, N):
        acc_x += DX
        acc_y += DY
        state, fill, reset = temporal_update(state, res[t], ALPHA, acc_x, acc_y)
        if reset:
            acc_x = acc_y = 0.0

    # 只比中心区：边缘是 BORDER_REPLICATE 造出来的假内容，真实使用中框外有 margin
    def err(img, ref):
        return float(np.abs(img - ref)[20:70, 20:120].mean())

    e_old, e_new = err(old, gt[-1]), err(fill, gt[-1])
    check("锚点 EMA 比累积式更接近真值", e_new < e_old * 0.85,
          f"锚点 {e_new:.2f} vs 累积 {e_old:.2f}")

    # 误差不应随帧数单调上涨（累积式的典型症状）
    state2, f_early = res[0].copy(), res[0]
    for t in range(1, 5):
        state2, f_early, _ = temporal_update(state2, res[t], ALPHA, DX * t, DY * t)
    e_early = err(f_early, gt[4])
    check("锚点 EMA 误差不随帧数增长", e_new <= e_early * 1.5 + 0.5,
          f"第4帧 {e_early:.2f} → 第{N}帧 {e_new:.2f}")

    s2, f2, reset = temporal_update(state, res[-1], ALPHA, ANCHOR_MAX_PX + 5.0, 0.0)
    check("锚点漂移过远自动重新锚定", bool(reset) and bool(np.allclose(f2, s2)),
          f"{ANCHOR_MAX_PX}+5px")


def test_carry_resample() -> None:
    prev_crop = (100, 100, 300, 260)                       # 200x160
    prev_fill = np.full((160, 200, 3), 42, np.float32)
    prev_fill[:, :, 1] = 84
    crop = (150, 80, 320, 240)                             # 170x160，仅部分重叠
    got = resample_carry(prev_fill, prev_crop, crop)
    check("跨段重采样：有重叠就返回", got is not None)
    img, valid = got
    check("跨段重采样：尺寸跟随本段 crop",
          img.shape == (160, 170, 3) and valid.shape == (160, 170),
          f"{img.shape}")
    dx1, dy1 = 150 - crop[0], 100 - crop[1]
    zone = img[dy1:dy1 + 5, dx1:dx1 + 5]
    check("跨段重采样：重叠区内容原样搬运",
          bool((zone[:, :, 0] == 42).all()) and bool((zone[:, :, 1] == 84).all())
          and bool(valid[dy1:dy1 + 5, dx1:dx1 + 5].all()))
    check("跨段重采样：非重叠区标记无效",
          (not valid[0, 0]) and float(img[0, 0].sum()) == 0.0)
    check("跨段重采样：完全无重叠返回 None",
          resample_carry(prev_fill, prev_crop, (500, 500, 600, 600)) is None)
    check("跨段重采样：尺寸对不上返回 None",
          resample_carry(prev_fill[:10], prev_crop, crop) is None)
    check("跨段渐变权重单调降到 0",
          carry_weight(CARRY_FADE) > carry_weight(1) > 0.0
          and carry_weight(0) == 0.0
          and abs(carry_weight(CARRY_FADE) - CARRY_FADE / (CARRY_FADE + 1)) < 1e-9,
          f"w={carry_weight(CARRY_FADE):.2f} → {carry_weight(0):.2f}")


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
    test_adaptive_alpha()
    test_anchor_ema()
    test_carry_resample()
    test_fade_blend()
    test_clean_ref()
    test_settings_compat()
    print("\n" + ("=== ALL PASS ===" if ok_all else "=== 存在 FAIL ==="))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
