# -*- coding: utf-8 -*-
"""临时诊断：校验 ffmpeg -ss 抽帧精度与视频时间戳。"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.make_test_video import decode_index_block  # noqa: E402

FFMPEG = "ffmpeg"
VIDEO = r"E:\tmp\suberase_test\test_sub.mp4"
W, H, FPS = 1280, 720, 30.0


def grab_single(t: float):
    p = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{t:.6f}", "-i", VIDEO,
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True)
    if len(p.stdout) < W * H * 3:
        return None, len(p.stdout)
    a = np.frombuffer(p.stdout, np.uint8).reshape(H, W, 3)
    return decode_index_block(a), len(p.stdout)


def grab_pipe(start_idx: int, count: int):
    ss = start_idx / FPS
    p = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{ss:.6f}", "-i", VIDEO,
         "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-vsync", "0", "-"],
        capture_output=True)
    got = []
    n = len(p.stdout) // (W * H * 3)
    for i in range(min(count, n)):
        a = np.frombuffer(p.stdout, np.uint8, count=W * H * 3,
                          offset=i * W * H * 3).reshape(H, W, 3)
        got.append(decode_index_block(a))
    return got, n


print("== 单帧 -ss 抽帧 ==")
for idx in [0, 1, 5, 17, 33, 60, 61, 90, 118, 119]:
    got, ln = grab_single(idx / FPS)
    print(f"  目标 {idx:3d} -> {got}   bytes={ln}")

print("== 管道从 0 开始读前 8 帧 ==")
got, n = grab_pipe(0, 8)
print(f"  {got}  总帧数={n}")

print("== 管道从 58 开始读 6 帧 ==")
got, n = grab_pipe(58, 6)
print(f"  {got}  总帧数={n}")

print("== 前 5 帧 PTS/时间戳 (ffprobe) ==")
p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "frame=pts,pts_time,pkt_dts,best_effort_timestamp",
                    "-of", "csv=p=0", "-read_intervals", "%+#5", VIDEO],
                   capture_output=True, encoding="utf-8")
print(p.stdout.strip()[:500])
