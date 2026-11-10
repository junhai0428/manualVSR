# -*- coding: utf-8 -*-
"""擦除流水线：顺序读帧 → 按活跃框分组 → crop + 真 batch 推理 → 贴回 → 管道编码。

全程内存，不落中间文件。未被任何框覆盖的帧原样直通（不推理）。
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .project import Box, Project, Settings
from .video_io import FrameReader, VideoMeta, VideoWriter

MIN_CROP_H = 160
MIN_CROP_W = 256


def plan_segments(total_frames: int, boxes: List[Box]) -> List[Tuple[int, int, List[Box]]]:
    """按活跃框集合切分帧区间，同段内 crop 区域一致，便于组 batch。"""
    active = [b for b in boxes if b.enabled]
    if not active:
        return [(0, total_frames, [])]
    bounds = {0, total_frames}
    for b in active:
        lo, hi = b.span(total_frames)      # 半开放框要展开成真实区间
        bounds.add(max(0, min(lo, total_frames)))
        bounds.add(max(0, min(hi + 1, total_frames)))
    pts = sorted(p for p in bounds if 0 <= p <= total_frames)
    segs = []
    for a, b_end in zip(pts[:-1], pts[1:]):
        if b_end <= a:
            continue
        mid = a  # 段内任意帧活跃框相同
        bs = [b for b in active if b.covers(mid)]
        segs.append((a, b_end, bs))
    return segs


def merged_crop(boxes: List[Box], margin: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = min(b.x for b in boxes) - margin
    y1 = min(b.y for b in boxes) - margin
    x2 = max(b.x + b.w for b in boxes) + margin
    y2 = max(b.y + b.h for b in boxes) + margin
    # 保证最小上下文
    if y2 - y1 < MIN_CROP_H:
        cy = (y1 + y2) // 2
        y1 = cy - MIN_CROP_H // 2
        y2 = y1 + MIN_CROP_H
    if x2 - x1 < MIN_CROP_W:
        cx = (x1 + x2) // 2
        x1 = cx - MIN_CROP_W // 2
        x2 = x1 + MIN_CROP_W
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = min(w, max(x2, x1 + MIN_CROP_W))
    y2 = min(h, max(y2, y1 + MIN_CROP_H))
    x2 = min(x2, w)
    y2 = min(y2, h)
    return x1, y1, x2, y2


def build_mask(boxes: List[Box], crop: Tuple[int, int, int, int],
               cw: int, ch: int, dilate: int) -> np.ndarray:
    x1, y1, x2, y2 = crop
    mask = np.zeros((ch, cw), np.uint8)
    for b in boxes:
        rx1 = max(0, b.x - x1)
        ry1 = max(0, b.y - y1)
        rx2 = min(cw, b.x + b.w - x1)
        ry2 = min(ch, b.y + b.h - y1)
        if rx2 > rx1 and ry2 > ry1:
            mask[ry1:ry2, rx1:rx2] = 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, k)
    return mask


# ---------------- 时域稳定 ----------------
MIN_PHASE_RESP = 0.05        # 相位相关峰值低于它视为估计失败
MAX_SHIFT_PX = 40            # 单帧位移超过它视为失准（场景切换等）
BG_TOL = 12.0                # 真实背景与当前帧 mask 外的平均差上限
BG_TEXT_RATIO = 0.06         # 候选背景 mask 内白字占比上限
BG_FADE_FRAMES = 3           # 真实背景与平滑结果切换的渐变帧数
BG_MAX_SHIFT = 96.0          # 参考背景累计位移超它就不再采信
ALPHA_MIN_SHIFT = 0.5        # 帧间位移低于它（镜头基本静止）→ α 归零
ALPHA_FULL_SHIFT = 2.0       # 帧间位移达到它 → 用满 α
ANCHOR_MAX_PX = 32.0         # 锚点累计位移超它就把锚点前移到当前帧
CARRY_FADE = 3               # 跨段延续平滑的渐变帧数


def feather_alpha(mask: np.ndarray, px: int) -> np.ndarray:
    """向内羽化：mask 内距边界 <px 渐变到 0，mask 外恒 0。返回 [H,W,1] float32。"""
    if px <= 0:
        return (mask > 0).astype(np.float32)[:, :, None]
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip(dist / float(px), 0.0, 1.0).astype(np.float32)[:, :, None]


def estimate_shift(prev_gray: np.ndarray, cur_gray: np.ndarray,
                   exclude_y: Optional[Tuple[int, int]] = None
                   ) -> Tuple[float, float, float]:
    """估计 cur 相对 prev 的全局平移 (dx, dy, response)。

    只用字幕带之外的水平条带做相位相关，避免字幕内容主导相关峰；
    两条带取中值抗异常。摇移是全局运动，任一条带都能估出来。
    """
    h = prev_gray.shape[0]
    bands: List[Tuple[int, int]] = []
    if exclude_y:
        y1, y2 = exclude_y
        if y1 >= 48:
            bands.append((0, y1 - 8))
        if h - y2 >= 48:
            bands.append((y2 + 8, h))
    if not bands:
        bands.append((0, h))
    found = []
    for a, b in bands:
        if b - a < 48:
            continue
        p = prev_gray[a:b].astype(np.float32)
        c = cur_gray[a:b].astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(p, c)
        if (resp >= MIN_PHASE_RESP and abs(dx) <= MAX_SHIFT_PX
                and abs(dy) <= MAX_SHIFT_PX):
            found.append((float(dx), float(dy), float(resp)))
    if not found:
        return 0.0, 0.0, 0.0
    return (float(np.median([v[0] for v in found])),
            float(np.median([v[1] for v in found])),
            float(np.median([v[2] for v in found])))


def warp_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """把图像内容按 (dx,dy) 平移，方向与 phaseCorrelate 返回值一致。"""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return img
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def bg_usable(bg: np.ndarray, cur_crop: np.ndarray, mask: np.ndarray,
              tol: float = BG_TOL) -> bool:
    """真实背景候选是否可用：mask 外对齐误差小，且 mask 内没有残留字幕。"""
    outside = mask == 0
    if not outside.any():
        return False
    diff = float(np.abs(bg.astype(np.int16)
                        - cur_crop.astype(np.int16))[outside].mean())
    if diff > tol:
        return False
    inside = mask > 0
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    ratio = float(((s < 60) & (v > 210) & inside).sum()) / float(max(1, inside.sum()))
    return ratio <= BG_TEXT_RATIO


def composite(orig: np.ndarray, fill: np.ndarray,
              a_map: np.ndarray) -> np.ndarray:
    """只在 mask 内（带向内羽化）用 fill 替换 orig，mask 外保持原样。"""
    return np.clip(orig.astype(np.float32) * (1.0 - a_map)
                   + fill.astype(np.float32) * a_map, 0, 255).astype(np.uint8)


def fade_blend(current: float, target: float,
               frames: int = BG_FADE_FRAMES) -> float:
    """填充源占比渐变一帧：单帧校验抖动不会让真实背景来回开关。"""
    return float(current + np.clip(target - current,
                                   -1.0 / max(1, frames),
                                   1.0 / max(1, frames)))


def adaptive_alpha(base: float, dx: float, dy: float,
                   min_shift: float = ALPHA_MIN_SHIFT,
                   full_shift: float = ALPHA_FULL_SHIFT) -> float:
    """按帧间位移幅度缩放时域平滑强度。

    镜头基本静止（位移 < min_shift）时不做 EMA：这时对齐全靠亚像素插值，
    平滑没有收益，反而把每帧微小的修复差异抹成贴片/拖影。
    位移越大，逐帧修复的随机纹理差异越明显，平滑收益越大，线性升到满额。
    """
    if base <= 0.0:
        return 0.0
    s = float(np.hypot(dx, dy))
    if s <= min_shift:
        return 0.0
    t = (s - min_shift) / max(1e-6, full_shift - min_shift)
    return float(base * min(1.0, t))


def temporal_update(smooth: np.ndarray, cur: np.ndarray, alpha: float,
                    dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, bool]:
    """锚点系 EMA：状态钉在锚定帧坐标系，每帧只做「进一次、出一次」两次重采样。

    直接对屏幕系结果连续 warp（`s = warp((1-α)·r + α·s)`）会让亚像素插值的
    模糊逐帧累积，慢摇移素材上越擦越糊。这里把状态存回锚点坐标系，
    模糊不随帧数增长。累计位移超过 ANCHOR_MAX_PX 就把锚点前移到当前帧。

    返回 (新状态, 当前帧系填充, 是否需要重置累计位移)。
    """
    r_anchor = warp_shift(cur, -dx, -dy)          # 当前结果搬进锚点系
    state = ((1.0 - alpha) * r_anchor + alpha * smooth).astype(np.float32)
    fill = warp_shift(state, dx, dy)              # 一次搬回当前帧
    if abs(dx) > ANCHOR_MAX_PX or abs(dy) > ANCHOR_MAX_PX:
        # 位移太远，锚点处的旧内容已被 warparound 拉花，就地重新锚定
        return fill.copy(), fill, True
    return state, fill, False


def carry_weight(left: int, frames: int = CARRY_FADE) -> float:
    """跨段延续的混合权重：left 帧内从 frames/(frames+1) 线性降到 0。"""
    return max(0.0, float(left)) / float(max(1, frames) + 1)


def resample_carry(prev_fill: np.ndarray, prev_crop: Tuple[int, int, int, int],
                   crop: Tuple[int, int, int, int]
                   ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """把上一段 crop 坐标系下的填充图搬到新 crop 坐标系。

    段切换时 crop 可能整体平移/缩放，两段只有部分重叠。这里只做整数平移
    （不缩放、不插值，避免给延续内容引入额外模糊），重叠之外的部分交给
    valid 掩码，由调用方回退到本段自己的修复结果。

    返回 (img, valid)；两段完全无重叠时返回 None。
    """
    if prev_fill is None:
        return None
    px1, py1, px2, py2 = prev_crop
    x1, y1, x2, y2 = crop
    ch, cw = y2 - y1, x2 - x1
    if ch <= 0 or cw <= 0:
        return None
    ox1, oy1 = max(px1, x1), max(py1, y1)
    ox2, oy2 = min(px2, x2), min(py2, y2)
    if ox2 <= ox1 or oy2 <= oy1:
        return None
    ph, pw = prev_fill.shape[0], prev_fill.shape[1]
    if (py2 - py1, px2 - px1) != (ph, pw):
        return None
    img = np.zeros((ch, cw, prev_fill.shape[2]), np.float32)
    valid = np.zeros((ch, cw), bool)
    sx1, sy1 = ox1 - px1, oy1 - py1
    dx1, dy1 = ox1 - x1, oy1 - y1
    w, h = ox2 - ox1, oy2 - oy1
    img[dy1:dy1 + h, dx1:dx1 + w] = prev_fill[sy1:sy1 + h, sx1:sx1 + w]
    valid[dy1:dy1 + h, dx1:dx1 + w] = True
    return img, valid


def find_clean_ref(seg_a: int, active_segments: List[Tuple[int, int]],
                   max_back: int = 6) -> Optional[int]:
    """往前找一帧不带任何擦除框的真实背景，避免把上一段字幕贴回来。"""
    for k in range(2, max_back + 2):
        idx = seg_a - k
        if idx < 0:
            break
        if any(a <= idx < b for a, b in active_segments):
            continue
        return idx
    return None


class EraseTask:
    def __init__(self, project: Project, meta: VideoMeta, out_path: str, engine=None):
        self.project = project
        self.meta = meta
        self.out_path = out_path
        self.engine = engine
        self.cancel_flag = False
        self.stats = {"frames": 0, "inpainted": 0, "seconds": 0.0, "encoder": "",
                      "canceled": False, "error": "", "bg_frames": 0.0,
                      "carry_frames": 0}

    def cancel(self) -> None:
        self.cancel_flag = True

    def run(self, on_progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
        st = self.settings = self.project.settings
        t0 = time.time()
        total = self.meta.nb_frames
        reader = FrameReader(self.project.video, self.meta)
        writer = VideoWriter(
            self.out_path, self.meta, self.project.video,
            encoder=st.encoder, keep_audio=st.keep_audio, crf=st.crf,
        )
        self.stats["encoder"] = writer.encoder

        done = 0
        inpainted = 0
        carry_img = None          # 上一段末帧的填充（按上一段 crop 坐标系存）
        carry_valid = None
        carry_crop = None
        carry_left = 0
        segments = plan_segments(total, self.project.boxes)
        active_segments = [(a, b) for a, b, bs in segments if bs]
        try:
            for seg_a, seg_b, seg_boxes in segments:
                if self.cancel_flag:
                    self.stats["canceled"] = True
                    break
                if not seg_boxes:
                    # 中间隔着无字幕帧，延续已无意义
                    carry_img = carry_valid = carry_crop = None
                    carry_left = 0
                    for f in range(seg_a, seg_b):
                        frame = reader.read(f)
                        if frame is None:
                            done = total
                            break
                        writer.write(frame)
                        done += 1
                        if on_progress and done % 50 == 0:
                            on_progress(done, total, "直通")
                else:
                    crop = merged_crop(seg_boxes, st.margin, self.meta.width, self.meta.height)
                    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
                    mask = build_mask(seg_boxes, crop, cw, ch, st.dilate)
                    a_map = feather_alpha(mask, st.feather_px)
                    bs = self.engine.auto_batch(ch, cw, st.batch_size) if self.engine else 4

                    # 段内时域状态：crop 变了，能迁移的迁移，不能迁移的才重置
                    smooth = None                 # 锚点系稳定填充图 float32 [ch,cw,3]
                    s_dx, s_dy = 0.0, 0.0         # 锚点 -> 当前帧累计位移
                    prev_gray = None              # 上一帧灰度（运动估计用）
                    bg_dx, bg_dy = 0.0, 0.0       # 参考背景帧 -> 当前帧累计位移
                    ref_bg = None                 # 段前无字幕帧的真实背景
                    bg_blend = 0.0               # 真实背景占比（渐变切换）
                    if st.real_bg and seg_a > 0:
                        ref_idx = find_clean_ref(seg_a, active_segments)
                        ref_frame = reader.read(ref_idx) if ref_idx is not None else None
                        if ref_frame is not None:
                            ref_bg = ref_frame[crop[1]:crop[3], crop[0]:crop[2]].copy()
                            prev_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                    alpha = max(0.0, min(0.95, float(st.temporal_alpha)))

                    # 跨段延续：把上一段末尾的填充按整数平移搬进本段 crop，3 帧渐变交接
                    if carry_img is not None and carry_crop is not None:
                        got = resample_carry(carry_img, carry_crop, crop)
                        if got is not None:
                            carry_img, carry_valid = got
                            carry_left = CARRY_FADE
                        else:
                            carry_img = carry_valid = None
                            carry_left = 0
                    else:
                        carry_left = 0

                    f = seg_a
                    while f < seg_b:
                        if self.cancel_flag:
                            self.stats["canceled"] = True
                            break
                        end = min(f + bs, seg_b)
                        idxs = list(range(f, end))
                        frames, crops, masks = [], [], []
                        for i in idxs:
                            frame = reader.read(i)
                            if frame is None:
                                break
                            frames.append(frame)
                            sub = frame[crop[1]:crop[3], crop[0]:crop[2]]
                            crops.append(sub[:, :, ::-1])  # BGR -> RGB
                            masks.append(mask)
                        if not frames:
                            break
                        img_b = np.stack(crops).astype(np.uint8)
                        msk_b = np.stack(masks).astype(np.uint8)
                        res = self.engine.inpaint_batch(img_b, msk_b)
                        last_fill = None
                        for k, frame in enumerate(frames):
                            orig = frame[crop[1]:crop[3], crop[0]:crop[2]]
                            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            dx = dy = 0.0
                            if st.motion_comp and prev_gray is not None:
                                dx, dy, _ = estimate_shift(
                                    prev_gray, cur_gray, (crop[1], crop[3]))
                            bg_dx += dx
                            bg_dy += dy

                            r = res[k].astype(np.float32)
                            # 平滑强度随帧间位移缩放：静止/微动不做 EMA，避免贴片拖影
                            a = adaptive_alpha(alpha, dx, dy) if st.motion_comp else alpha
                            if smooth is None or a <= 0.0:
                                # 锚点前移到当前帧：状态就是本帧结果，不留历史残留
                                smooth = r.copy()
                                s_dx = s_dy = 0.0
                                fill = r
                            else:
                                s_dx += dx
                                s_dy += dy
                                smooth, fill, reset = temporal_update(
                                    smooth, r, a, s_dx, s_dy)
                                if reset:
                                    s_dx = s_dy = 0.0

                            # 跨段交接：本段前几帧与上一段的填充渐变混合，避免硬切
                            if carry_left > 0 and carry_img is not None:
                                w = carry_weight(carry_left)
                                fill = np.where(
                                    carry_valid[:, :, None],
                                    (1.0 - w) * fill + w * carry_img, fill)
                                carry_left -= 1
                                self.stats["carry_frames"] += 1

                            if ref_bg is not None:
                                far = (abs(bg_dx) > BG_MAX_SHIFT
                                       or abs(bg_dy) > BG_MAX_SHIFT)
                                bg = warp_shift(ref_bg, bg_dx, bg_dy)
                                if far:
                                    target = 0.0      # 对齐走太远，参考背景已不可信
                                else:
                                    target = 1.0 if bg_usable(bg, orig, mask) else 0.0
                                # 填充源渐变切换：单帧校验结果抖动不会造成来回跳
                                bg_blend = fade_blend(bg_blend, target)
                                if bg_blend > 0:
                                    fill = ((1.0 - bg_blend) * fill
                                            + bg_blend * bg.astype(np.float32))
                                    self.stats["bg_frames"] += bg_blend

                            frame[crop[1]:crop[3], crop[0]:crop[2]] = composite(
                                orig, fill, a_map)
                            last_fill = fill
                            writer.write(frame)
                            prev_gray = cur_gray
                            done += 1
                        if last_fill is not None:
                            carry_img = np.ascontiguousarray(last_fill, dtype=np.float32)
                            carry_crop = crop
                        inpainted += len(frames)
                        f = end
                        if on_progress:
                            speed = done / max(1e-6, time.time() - t0)
                            on_progress(done, total, f"擦除 {inpainted} 帧 · {speed:.1f} fps")
        except Exception as e:
            self.stats["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            try:
                writer.close()
            except Exception as e:
                if not self.stats["error"]:
                    self.stats["error"] = str(e)
            reader.release()

        self.stats["frames"] = done
        self.stats["inpainted"] = inpainted
        self.stats["bg_frames"] = round(self.stats["bg_frames"], 1)
        self.stats["seconds"] = round(time.time() - t0, 2)
        if on_progress:
            on_progress(done, total, "完成")
        return self.stats
