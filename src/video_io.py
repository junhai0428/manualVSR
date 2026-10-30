# -*- coding: utf-8 -*-
"""视频元信息探测、帧读取（随机 seek + 顺序浏览）、NVENC 写出。

帧号与时间换算：以 ffprobe 的 r_frame_rate 为准（fps 换算），
avg_frame_rate 与 r_frame_rate 差异过大时给出 VFR 警告（ffprobe 兜底校验）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
PROBE_TIMEOUT = 60
_NVENC_CACHE: Optional[bool] = None


class VideoIOError(RuntimeError):
    pass


@dataclass
class VideoMeta:
    path: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fps_num: int = 0
    fps_den: int = 1
    duration: float = 0.0
    nb_frames: int = 0
    nb_frames_exact: bool = False
    has_audio: bool = False
    vfr: bool = False
    note: str = ""

    @property
    def fps_frac(self) -> str:
        return f"{self.fps_num}/{self.fps_den}" if self.fps_num else f"{self.fps:.3f}"

    def frame_to_time(self, idx: int) -> float:
        return idx / self.fps if self.fps else 0.0

    def time_to_frame(self, t: float) -> int:
        return int(round(t * self.fps)) if self.fps else 0

    def fmt_time(self, idx: int) -> str:
        t = self.frame_to_time(idx)
        m, s = divmod(t, 60)
        return f"{int(m):02d}:{s:05.2f}"


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, timeout=timeout, encoding="utf-8", errors="replace"
    )


def _parse_rate(v: str) -> Tuple[int, int]:
    if not v or v in ("0/0", "N/A"):
        return 0, 0
    if "/" in v:
        a, b = v.split("/", 1)
        try:
            return int(float(a)), int(float(b))
        except ValueError:
            return 0, 0
    try:
        return int(round(float(v) * 1000)), 1000
    except ValueError:
        return 0, 0


def probe(path: str) -> VideoMeta:
    if not os.path.exists(path):
        raise VideoIOError(f"文件不存在: {path}")
    p = _run([FFPROBE, "-v", "error", "-print_format", "json",
              "-show_streams", "-show_format", path], PROBE_TIMEOUT)
    if p.returncode != 0:
        raise VideoIOError(f"ffprobe 失败: {p.stderr.strip()[:300]}")
    data = json.loads(p.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise VideoIOError("没有视频流")
    a = any(s.get("codec_type") == "audio" for s in streams)

    rn, rd = _parse_rate(v.get("r_frame_rate", ""))
    an, ad = _parse_rate(v.get("avg_frame_rate", ""))
    fps = (rn / rd) if rd else ((an / ad) if ad else 0.0)

    dur = 0.0
    for key in ("duration", "tags.DURATION"):
        pass
    try:
        dur = float(v.get("duration") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    if dur <= 0:
        try:
            dur = float(data.get("format", {}).get("duration") or 0)
        except (TypeError, ValueError):
            dur = 0.0

    nb = 0
    exact = False
    try:
        nb = int(v.get("nb_frames") or 0)
    except (TypeError, ValueError):
        nb = 0
    if nb <= 0:
        try:  # mp4/mov 常把帧数写在 tags 里
            nb = int(v.get("tags", {}).get("NUMBER_OF_FRAMES", 0))
        except (TypeError, ValueError):
            nb = 0
    if nb > 0:
        exact = True
    elif dur > 0 and fps > 0:
        nb = int(round(dur * fps))
        exact = False

    vfr = False
    if rd and ad and rn and an:
        r_fps, a_fps = rn / rd, an / ad
        if r_fps > 0 and abs(r_fps - a_fps) / r_fps > 0.02:
            vfr = True

    meta = VideoMeta(
        path=path,
        width=int(v.get("width") or 0),
        height=int(v.get("height") or 0),
        fps=fps or 25.0,
        fps_num=rn or int(round((fps or 25) * 1000)),
        fps_den=rd or 1000,
        duration=dur,
        nb_frames=max(nb, 1),
        nb_frames_exact=exact,
        has_audio=a,
        vfr=vfr,
    )
    if vfr:
        meta.note = "检测到可变帧率(VFR)，帧号按平均帧率换算，可能有 1 帧偏差"
    elif not exact:
        meta.note = "总帧数由 时长×帧率 估算"
    return meta


def count_frames_exact(path: str, timeout: int = 600) -> Optional[int]:
    """精确统计帧数（慢，用于兜底校验）。"""
    p = _run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
              "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path], timeout)
    if p.returncode != 0:
        return None
    try:
        return int((p.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def nvenc_available(timeout: int = 25) -> bool:
    global _NVENC_CACHE
    if _NVENC_CACHE is not None:
        return _NVENC_CACHE
    try:
        p = _run([FFMPEG, "-hide_banner", "-loglevel", "error",
                  "-f", "lavfi", "-i", "nullsrc=s=320x240:d=0.1:r=25",
                  "-c:v", "h264_nvenc", "-f", "null", "-"], timeout)
        _NVENC_CACHE = p.returncode == 0
    except Exception:
        _NVENC_CACHE = False
    return _NVENC_CACHE


class FrameReader:
    """精确逐帧读取：常驻 ffmpeg 子进程输出 rawvideo，顺序解码走管道（极快），
    跳转时用 `-ss`（输入 seek，按时间戳精确定位）重启管道。

    不走 cv2.CAP_PROP_POS_FRAMES：该接口在 ffmpeg 后端存在 1~2 帧偏移。
    """

    SEEK_BACK = 2       # 跳转时回退帧数，抵消 -ss 的取整误差
    READ_AHEAD = 96     # 连续读窗口，超出则重新 seek

    def __init__(self, path: str, meta: "VideoMeta"):
        self.path = path
        self.meta = meta
        self.w, self.h = meta.width, meta.height
        self.frame_bytes = self.w * self.h * 3
        self.proc = None
        self._next: Optional[int] = None
        self._buf = b""

    # ---- 进程管理 ----
    def _close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
        except Exception:
            pass
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        self.proc = None

    def _open(self, start_frame: int) -> None:
        self._close()
        ss = max(0.0, start_frame / self.meta.fps) if self.meta.fps else 0.0
        cmd = [FFMPEG, "-v", "error", "-ss", f"{ss:.6f}", "-i", self.path,
               "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-vsync", "0", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     stdin=subprocess.DEVNULL)
        self._next = start_frame
        self._buf = b""

    # ---- 读帧 ----
    def _read_exact(self, n: int) -> Optional[bytes]:
        buf = self._buf
        while len(buf) < n:
            chunk = self.proc.stdout.read(max(65536, n - len(buf)))
            if not chunk:
                return None
            buf += chunk
        self._buf = buf[n:]
        return buf[:n]

    def _raw(self) -> Optional[np.ndarray]:
        data = self._read_exact(self.frame_bytes)
        if data is None:
            return None
        # copy：frombuffer 得到的是只读视图，后续要贴回擦除结果
        return np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3).copy()

    def read(self, idx: int) -> Optional[np.ndarray]:
        if self.proc is None or idx < self._next or idx > self._next + self.READ_AHEAD:
            self._open(max(0, idx - self.SEEK_BACK))
        frame = None
        while self._next <= idx:
            frame = self._raw()
            if frame is None:
                return None
            self._next += 1
        return frame

    def release(self) -> None:
        self._close()


class VideoWriter:
    """把 BGR 帧通过管道交给 ffmpeg 编码，音频从源片 copy。全内存，不落中间文件。"""

    def __init__(self, out_path: str, meta: VideoMeta, src_path: str,
                 encoder: str = "auto", keep_audio: bool = True, crf: int = 18):
        self.out_path = out_path
        self.src_path = src_path
        self.written = 0
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        enc = encoder
        if enc == "auto":
            enc = "nvenc" if nvenc_available() else "x264"
        if enc == "nvenc" and not nvenc_available():
            enc = "x264"
        self.encoder = enc

        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{meta.width}x{meta.height}", "-r", meta.fps_frac, "-i", "pipe:0"]
        if keep_audio and meta.has_audio:
            # 用 -t 把音频裁到与视频同长；不用 -shortest（会反过来截断视频尾帧）
            cmd += ["-ss", "0", "-t", f"{max(0.001, meta.duration):.3f}", "-i", src_path,
                    "-map", "0:v:0", "-map", "1:a?"]
        else:
            cmd += ["-map", "0:v:0"]
        if enc == "nvenc":
            cmd += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                    "-cq", str(crf), "-b:v", "0"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
        cmd += ["-pix_fmt", "yuv420p", "-c:a", "copy",
                "-movflags", "+faststart", out_path]

        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())
            self.written += 1
        except BrokenPipeError as e:  # pragma: no cover
            err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
            raise VideoIOError(f"ffmpeg 写入中断: {err[-500:]}") from e

    def close(self, timeout: int = 600) -> int:
        if self._closed:
            return self.proc.returncode or 0
        self._closed = True
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            err = self.proc.stderr.read().decode("utf-8", "replace")
        except Exception:
            err = ""
        rc = self.proc.wait(timeout=timeout)
        if rc != 0:
            raise VideoIOError(f"ffmpeg 退出码 {rc}: {err[-500:]}")
        return rc
