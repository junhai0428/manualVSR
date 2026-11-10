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
                      "canceled": False, "error": "", "bg_frames": 0.0}

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
        segments = plan_segments(total, self.project.boxes)
        active_segments = [(a, b) for a, b, bs in segments if bs]
        try:
            for seg_a, seg_b, seg_boxes in segments:
                if self.cancel_flag:
                    self.stats["canceled"] = True
                    break
                if not seg_boxes:
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

                    # 段内时域状态：换段必须重置（crop 与字幕位置都变了）
                    smooth = None                 # 稳定填充图 float32 [ch,cw,3]
                    prev_gray = None              # 上一帧灰度（运动估计用）
                    acc_dx, acc_dy = 0.0, 0.0    # 相对参考帧的累计位移
                    ref_bg = None                 # 段前无字幕帧的真实背景
                    bg_blend = 0.0               # 真实背景占比（渐变切换）
                    if st.real_bg and seg_a > 0:
                        ref_idx = find_clean_ref(seg_a, active_segments)
                        ref_frame = reader.read(ref_idx) if ref_idx is not None else None
                        if ref_frame is not None:
                            ref_bg = ref_frame[crop[1]:crop[3], crop[0]:crop[2]].copy()
                            prev_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                    alpha = max(0.0, min(0.95, float(st.temporal_alpha)))

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
                        for k, frame in enumerate(frames):
                            orig = frame[crop[1]:crop[3], crop[0]:crop[2]]
                            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            dx = dy = 0.0
                            if st.motion_comp and prev_gray is not None:
                                dx, dy, _ = estimate_shift(
                                    prev_gray, cur_gray, (crop[1], crop[3]))
                            acc_dx += dx
                            acc_dy += dy

                            r = res[k].astype(np.float32)
                            if smooth is None:
                                smooth = r.copy()
                            elif alpha > 0:
                                prev_aligned = warp_shift(smooth, dx, dy)
                                smooth = (1.0 - alpha) * r + alpha * prev_aligned
                            else:
                                smooth = r

                            fill = smooth
                            if ref_bg is not None:
                                bg = warp_shift(ref_bg, acc_dx, acc_dy)
                                target = 1.0 if bg_usable(bg, orig, mask) else 0.0
                                # 填充源渐变切换：单帧校验结果抖动不会造成来回跳
                                bg_blend = fade_blend(bg_blend, target)
                                if bg_blend > 0:
                                    fill = ((1.0 - bg_blend) * smooth
                                            + bg_blend * bg.astype(np.float32))
                                    self.stats["bg_frames"] += bg_blend

                            frame[crop[1]:crop[3], crop[0]:crop[2]] = composite(
                                orig, fill, a_map)
                            writer.write(frame)
                            prev_gray = cur_gray
                            done += 1
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
