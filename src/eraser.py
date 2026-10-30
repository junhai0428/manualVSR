# -*- coding: utf-8 -*-
"""擦除流水线：顺序读帧 → 按活跃框分组 → crop + 真 batch 推理 → 贴回 → 管道编码。

全程内存，不落中间文件。未被任何框覆盖的帧原样直通（不推理）。
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple

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
        bounds.add(max(0, min(b.start, total_frames)))
        bounds.add(max(0, min(b.end + 1, total_frames)))
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
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask, k)
    return mask


class EraseTask:
    def __init__(self, project: Project, meta: VideoMeta, out_path: str, engine=None):
        self.project = project
        self.meta = meta
        self.out_path = out_path
        self.engine = engine
        self.cancel_flag = False
        self.stats = {"frames": 0, "inpainted": 0, "seconds": 0.0, "encoder": "",
                      "canceled": False, "error": ""}

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
        try:
            for seg_a, seg_b, seg_boxes in plan_segments(total, self.project.boxes):
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
                    bs = self.engine.auto_batch(ch, cw, st.batch_size) if self.engine else 4
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
                            masks.append(build_mask(seg_boxes, crop, cw, ch, st.dilate))
                        if not frames:
                            break
                        img_b = np.stack(crops).astype(np.uint8)
                        msk_b = np.stack(masks).astype(np.uint8)
                        res = self.engine.inpaint_batch(img_b, msk_b)
                        for k, frame in enumerate(frames):
                            frame[crop[1]:crop[3], crop[0]:crop[2]] = res[k]
                            writer.write(frame)
                            done += 1
                        inpainted += len(frames)
                        f = end
                        if on_progress:
                            speed = done / max(1e-6, time.time() - t0)
                            on_progress(done, total, f"擦除 {inpainted} 帧 · {speed:.1f} fps")
                            if self.cancel_flag:
                                pass
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
        self.stats["seconds"] = round(time.time() - t0, 2)
        if on_progress:
            on_progress(done, total, "完成")
        return self.stats
