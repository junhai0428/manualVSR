# -*- coding: utf-8 -*-
"""文字区域自动检测（RapidOCR det+rec）→ 擦除框 JSON。

独立子进程脚本，运行在单独的 OCR 环境（rapidocr-onnxruntime + onnxruntime-gpu），
不依赖本项目 venv（该环境解释器由环境变量 SUBERASER_DETECT_PY 或
local_config.json 的 detect_python 指定）。RapidOCR 原生管线出框，rec 识别过滤误检，
跨采样帧的框合成时间簇得到擦除框。输入视频，输出 JSON：
{
  "video", "fps", "nb_frames", "width", "height", "step_frames",
  "providers", "det_seconds",
  "boxes": [ {"x","y","w","h","start","end"}, ... ]
}
坐标为源视频像素，start/end 为闭区间帧号（与工程 Box 同语义）。

用法：
  python tools/auto_detect.py --video <path> [--step-sec 0.1]
      [--iou 0.3] [--margin 3] [--out result.json]
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


def quad_angle(points) -> float:
    """文字框倾斜角度（度），取文本框上边缘方向，范围 [-90, 90]。

    水平字幕≈0°，斜排的包装/背景文字角度明显。
    """
    import math
    if not points or len(points) < 4:
        return 0.0
    dx = float(points[1][0]) - float(points[0][0])
    dy = float(points[1][1]) - float(points[0][1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    a = math.degrees(math.atan2(dy, dx))
    while a > 90:
        a -= 180
    while a < -90:
        a += 180
    return a


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


def _same_line(a: Dict, b: Dict, ov_ratio: float = 0.5) -> bool:
    """同一行：垂直中心接近且水平有足够重叠（上下两行字幕不会被判成同行）。"""
    ac = a["y"] + a["h"] / 2
    bc = b["y"] + b["h"] / 2
    if abs(ac - bc) > 0.5 * min(a["h"], b["h"]):
        return False
    ow = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    return ow > 0 and ow / min(a["w"], b["w"]) >= ov_ratio


def _size_similar(a: Dict, b: Dict, sim: float = 0.8) -> bool:
    """宽高比例都接近（比值 >= sim），差值大说明是不同长度的两句字幕。"""
    return (min(a["w"], b["w"]) / max(a["w"], b["w"]) >= sim
            and min(a["h"], b["h"]) / max(a["h"], b["h"]) >= sim)


def bridge_gaps(boxes: List[Dict], max_gap: int, sim: float = 0.8) -> List[Dict]:
    """修复同一行字幕被切成多段导致的空洞，同时避免框尺寸膨胀。

    - 尺寸接近的相邻段（宽高比 >= sim）：合并成一个框（时间/矩形取并集）
    - 尺寸差异大的相邻段（换句、长短变化）：不合并矩形，只把前一段的
      end 延到后一段开头，填掉时间空洞；两段各自保留自己的矩形
    """
    if not boxes:
        return boxes
    out: List[Dict] = []
    for b in sorted(boxes, key=lambda x: (x["y"], x["x"], x["start"])):
        b = dict(b)
        target = None
        for m in reversed(out):
            if _same_line(m, b) and b["start"] - m["end"] - 1 <= max_gap:
                target = m
                break
        if target is not None:
            if _size_similar(target, b, sim):
                x, y, w, h = union((target["x"], target["y"],
                                    target["w"], target["h"]),
                                   (b["x"], b["y"], b["w"], b["h"]))
                target.update(x=x, y=y, w=w, h=h,
                              start=min(target["start"], b["start"]),
                              end=max(target["end"], b["end"]))
                continue
            if b["start"] - 1 > target["end"]:      # 只补时间洞
                target["end"] = b["start"] - 1
        out.append(b)
    out.sort(key=lambda b: (b["y"], b["x"]))
    return out


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
def detect(video: str, step_sec: float = 0.1, iou_thr: float = 0.3,
           margin: int = 3, out_path: Optional[str] = None,
           on_progress=None, thresh: float = 0.6, box_thresh: float = 0.8,
           rec_score: float = 0.8, min_h: int = 20,
           bridge_sec: float = 0.5, max_angle: float = 10.0,
           max_h: int = 0) -> Dict:
    from rapidocr_onnxruntime import RapidOCR

    meta = ffprobe_meta(video)
    fps, w, h = meta["fps"], meta["width"], meta["height"]
    nb_frames = meta["nb_frames"]
    step = max(1, int(round(fps * step_sec)))

    engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
    # DBNet 后处理阈值（不支持 per-call，这里按实例覆盖）
    engine.text_det.postprocess_op.thresh = thresh
    engine.text_det.postprocess_op.box_thresh = box_thresh
    providers = list(engine.text_det.infer.session.get_providers())

    dets: List[Dict] = []
    t0 = time.time()
    n_sampled = 0
    for frame_no, img in iter_sampled_frames(
            video, step, w, h, on_progress, nb_frames):
        # det + rec：每个候选框跑文字识别，认不出字/置信度低的丢弃（杀误检）
        res, _ = engine(img, use_det=True, use_cls=False, use_rec=True,
                        text_score=rec_score)
        n_sampled += 1
        if not res:
            continue
        for entry in res:
            points, text, score = entry[0], entry[1], float(entry[2])
            r = rect_of(points)
            if r[2] < 4 or r[3] < 4:        # 过滤碎块
                continue
            if r[3] < min_h:                # 过滤小字（商品包装/衣服上的小文字）
                continue
            if max_h > 0 and r[3] > max_h:  # 过滤超大字（背景装饰大字）
                continue
            if max_angle > 0 and abs(quad_angle(points)) > max_angle:
                continue                    # 过滤倾斜文字（斜排的包装/背景字）
            dets.append({"frame": frame_no, "rect": r,
                         "text": text, "score": score})
    det_seconds = round(time.time() - t0, 2)

    gap = step * 2
    clusters = merge_detections(dets, iou_thr, gap)
    boxes = clusters_to_boxes(clusters, step, margin, nb_frames, w, h)
    # 桥接被切段的同位置字幕，消除中间无框的空洞
    bridge_frames = max(0, int(round(fps * bridge_sec)))
    if bridge_frames > 0:
        boxes = bridge_gaps(boxes, bridge_frames)

    result = {"video": os.path.abspath(video), "fps": fps,
              "nb_frames": nb_frames, "nb_frames_exact": meta["nb_frames_exact"],
              "width": w, "height": h, "step_frames": step,
              "sampled": n_sampled, "providers": providers,
              "thresh": thresh, "box_thresh": box_thresh, "rec_score": rec_score,
              "min_h": min_h, "bridge_sec": bridge_sec,
              "max_angle": max_angle, "max_h": max_h,
              "det_seconds": det_seconds, "boxes": boxes}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="文字区域自动检测 → 擦除框 JSON")
    ap.add_argument("--video", required=True)
    ap.add_argument("--step-sec", type=float, default=0.1, help="抽帧间隔秒")
    ap.add_argument("--iou", type=float, default=0.3, help="合并 IoU 阈值")
    ap.add_argument("--margin", type=int, default=3, help="框外扩像素")
    ap.add_argument("--thresh", type=float, default=0.6,
                    help="DBNet 概率图二值化阈值（小=框更松更全）")
    ap.add_argument("--box-thresh", dest="box_thresh", type=float, default=0.8,
                    help="候选框整体置信度门槛（小=保留更多弱候选）")
    ap.add_argument("--rec-score", type=float, default=0.8,
                    help="OCR 识别过滤分数：识别置信度低于它的一律丢弃（杀误检主手段）")
    ap.add_argument("--min-h", dest="min_h", type=int, default=20,
                    help="最小文字高度(px)：检测框高度低于它的一律丢弃（过滤小字）")
    ap.add_argument("--bridge-sec", type=float, default=0.5,
                    help="断点桥接：同一位置字幕两段框间隙小于该秒数则合并（0=关闭）")
    ap.add_argument("--max-angle", dest="max_angle", type=float, default=10.0,
                    help="最大倾斜角度(度)：超过它的倾斜文字丢弃（0=不过滤）")
    ap.add_argument("--max-h", dest="max_h", type=int, default=0,
                    help="最大文字高度(px)：超过它的超大字丢弃（0=不过滤）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    ap.add_argument("--progress", action="store_true", help="stderr 输出进度")
    args = ap.parse_args()

    def prog(frame_no: int) -> None:
        print(f"PROGRESS {frame_no}", file=sys.stderr, flush=True)

    r = detect(args.video, args.step_sec, args.iou, args.margin, args.out,
               on_progress=prog if args.progress else None,
               thresh=args.thresh, box_thresh=args.box_thresh,
               rec_score=args.rec_score, min_h=args.min_h,
               bridge_sec=args.bridge_sec, max_angle=args.max_angle,
               max_h=args.max_h)
    print(json.dumps({
        "providers": r["providers"], "det_seconds": r["det_seconds"],
        "sampled": r["sampled"], "boxes": len(r["boxes"]),
        "nb_frames": r["nb_frames"], "fps": round(r["fps"], 3),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
