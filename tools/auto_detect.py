# -*- coding: utf-8 -*-
"""文字区域自动检测（RapidOCR det-only）→ 擦除框 JSON。

独立子进程脚本，运行在 <OCR_ENV> 环境（rapidocr-onnxruntime +
onnxruntime-gpu），不依赖本项目 venv。输入视频，输出 JSON：
{
  "video", "fps", "nb_frames", "width", "height", "step_frames",
  "providers", "det_seconds",
  "boxes": [ {"x","y","w","h","start","end"}, ... ]
}
坐标为源视频像素，start/end 为闭区间帧号（与工程 Box 同语义）。

用法：
  python tools/auto_detect.py --video <path> [--step-sec 0.5]
      [--iou 0.3] [--margin 8] [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

Rect = Tuple[int, int, int, int]    # x, y, w, h


# ---------------- ffprobe / ffmpeg ----------------
def ffprobe_meta(video: str) -> Dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=avg_frame_rate,r_frame_rate,width,height,nb_frames",
           "-show_entries", "format=duration",
           "-of", "json", video]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {out.stderr.strip()[:300]}")
    info = json.loads(out.stdout)
    st = info["streams"][0]
    def _fps(s: str) -> float:
        if not s or s == "0/0":
            return 0.0
        num, _, den = s.partition("/")
        try:
            return float(num) / float(den or 1)
        except ZeroDivisionError:
            return 0.0
    fps = _fps(st.get("avg_frame_rate")) or _fps(st.get("r_frame_rate")) or 25.0
    w, h = int(st["width"]), int(st["height"])
    nb = st.get("nb_frames")
    if nb:
        nb_frames = int(nb)
        exact = True
    else:
        dur = float((info.get("format") or {}).get("duration") or 0)
        nb_frames = max(1, int(round(dur * fps)))
        exact = False
    return {"fps": fps, "width": w, "height": h, "nb_frames": nb_frames,
            "nb_frames_exact": exact}


def iter_sampled_frames(video: str, step: int, width: int, height: int,
                        on_progress=None, total_frames: int = 0):
    """逐帧产出 (frame_no, bgr_ndarray)。frame_no = k * step。

    用 ffmpeg 输出 rawvideo (bgr24) 到管道，内存托管，不落盘。
    """
    import numpy as np

    vf = f"select=not(mod(n\\,{step}))"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", video,
           "-vf", vf, "-vsync", "0", "-an", "-sn", "-dn",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    k = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            yield k * step, np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            k += 1
            if on_progress and total_frames and k % 10 == 0:
                on_progress(min(total_frames - 1, k * step))
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.wait(timeout=30)
        if proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg 抽帧失败: {err.strip()[:300]}")
    finally:
        if proc.poll() is None:
            proc.kill()


# ---------------- 几何：IoU / 合并 ----------------
def rect_of(points) -> Rect:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (int(round(x0)), int(round(y0)),
            max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0))))


def iou(a: Rect, b: Rect) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / float(aw * ah + bw * bh - inter)


def merge_detections(dets: List[Dict], iou_thr: float, gap_frames: int) -> List[Dict]:
    """把多个采样帧上的检测框合并成时间簇。

    dets: [{"frame": int, "rect": Rect}]（按 frame 升序）
    返回: [{"rect": 合并矩形, "first": 首帧, "last": 末帧, "hits": 次数}]
    规则：与簇 last 帧间隔 <= gap_frames 且与簇矩形 IoU >= iou_thr 的检测并入
    （取 IoU 最大的簇）；否则新建簇。
    """
    clusters: List[Dict] = []
    for d in sorted(dets, key=lambda x: x["frame"]):
        best, best_iou = None, 0.0
        for c in clusters:
            if d["frame"] - c["last"] > gap_frames:
                continue
            v = iou(d["rect"], c["rect"])
            if v >= iou_thr and v > best_iou:
                best, best_iou = c, v
        if best is None:
            clusters.append({"rect": d["rect"], "first": d["frame"],
                             "last": d["frame"], "hits": 1})
        else:
            best["rect"] = union(best["rect"], d["rect"])
            best["last"] = d["frame"]
            best["hits"] += 1
    return clusters


def union(a: Rect, b: Rect) -> Rect:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    x0, y0 = min(ax0, bx0), min(ay0, by0)
    x1 = max(ax0 + aw, bx0 + bw)
    y1 = max(ay0 + ah, by0 + bh)
    return (x0, y0, x1 - x0, y1 - y0)


def clusters_to_boxes(clusters: List[Dict], step: int, margin: int,
                      nb_frames: int, width: int, height: int) -> List[Dict]:
    """簇 → 擦除框：时间上下各扩一个采样间隔，像素外扩 margin。"""
    last_frame = max(0, nb_frames - 1)
    out = []
    for c in clusters:
        x, y, w, h = c["rect"]
        x, y = x - margin, y - margin
        w, h = w + 2 * margin, h + 2 * margin
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        start = max(0, c["first"] - step)
        end = min(last_frame, c["last"] + step)
        out.append({"x": x, "y": y, "w": w, "h": h, "start": start, "end": end})
    out.sort(key=lambda b: (b["y"], b["x"]))
    return out


# ---------------- 主流程 ----------------
def detect(video: str, step_sec: float = 0.5, iou_thr: float = 0.3,
           margin: int = 8, out_path: Optional[str] = None,
           on_progress=None) -> Dict:
    from rapidocr_onnxruntime import RapidOCR

    meta = ffprobe_meta(video)
    fps, w, h = meta["fps"], meta["width"], meta["height"]
    nb_frames = meta["nb_frames"]
    step = max(1, int(round(fps * step_sec)))

    engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
    providers = list(engine.text_det.infer.session.get_providers())

    dets: List[Dict] = []
    t0 = time.time()
    n_sampled = 0
    for frame_no, img in iter_sampled_frames(
            video, step, w, h, on_progress, nb_frames):
        res, _ = engine(img, use_det=True, use_cls=False, use_rec=False)
        n_sampled += 1
        if not res:
            continue
        for points in res:
            r = rect_of(points)
            if r[2] < 4 or r[3] < 4:        # 过滤碎块
                continue
            dets.append({"frame": frame_no, "rect": r})
    det_seconds = round(time.time() - t0, 2)

    gap = step * 2
    clusters = merge_detections(dets, iou_thr, gap)
    boxes = clusters_to_boxes(clusters, step, margin, nb_frames, w, h)

    result = {"video": os.path.abspath(video), "fps": fps,
              "nb_frames": nb_frames, "nb_frames_exact": meta["nb_frames_exact"],
              "width": w, "height": h, "step_frames": step,
              "sampled": n_sampled, "providers": providers,
              "det_seconds": det_seconds, "boxes": boxes}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="文字区域自动检测 → 擦除框 JSON")
    ap.add_argument("--video", required=True)
    ap.add_argument("--step-sec", type=float, default=0.5, help="抽帧间隔秒")
    ap.add_argument("--iou", type=float, default=0.3, help="合并 IoU 阈值")
    ap.add_argument("--margin", type=int, default=8, help="框外扩像素")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--progress", action="store_true", help="stderr 输出进度")
    args = ap.parse_args()

    def prog(frame_no: int) -> None:
        print(f"PROGRESS {frame_no}", file=sys.stderr, flush=True)

    r = detect(args.video, args.step_sec, args.iou, args.margin, args.out,
               on_progress=prog if args.progress else None)
    print(json.dumps({
        "providers": r["providers"], "det_seconds": r["det_seconds"],
        "sampled": r["sampled"], "boxes": len(r["boxes"]),
        "nb_frames": r["nb_frames"], "fps": round(r["fps"], 3),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
