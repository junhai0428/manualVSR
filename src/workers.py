# -*- coding: utf-8 -*-
"""后台线程：模型加载、单任务擦除、批量擦除、自动检测生成框。GUI 主线程只做显示与进度更新。"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

from .inpaint_engine import InpaintEngine
from .eraser import EraseTask
from .project import Project, Box, Settings, default_project_path, default_output_path
from .video_io import VideoMeta, probe

# 检测子进程环境（rapidocr-onnxruntime + onnxruntime-gpu）
DETECT_PYTHON = os.environ.get(
    "SUBERASER_DETECT_PY", r"<OCR_ENV>\Scripts\python.exe")
AUTO_DETECT_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools", "auto_detect.py")


def run_detect_subprocess(video: str, step_sec: float, iou_thr: float,
                          margin: int, on_progress=None,
                          thresh: float = 0.6, box_thresh: float = 0.8,
                          rec_score: float = 0.8, min_h: int = 20,
                          bridge_sec: float = 0.5) -> dict:
    """调用 tools/auto_detect.py 子进程跑检测，返回结果 JSON(dict)。

    on_progress(frame_no) 由子进程 stderr 的 PROGRESS 行驱动。
    """
    out_json = os.path.join(
        tempfile.gettempdir(), f"suberase_detect_{os.getpid()}.json")
    cmd = [DETECT_PYTHON, AUTO_DETECT_SCRIPT,
           "--video", video, "--step-sec", str(step_sec),
           "--iou", str(iou_thr), "--margin", str(margin),
           "--thresh", str(thresh), "--box-thresh", str(box_thresh),
           "--rec-score", str(rec_score), "--min-h", str(min_h),
           "--bridge-sec", str(bridge_sec),
           "--out", out_json, "--progress"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    for line in proc.stderr:
        line = line.strip()
        if line.startswith("PROGRESS ") and on_progress:
            try:
                on_progress(int(line.split()[1]))
            except ValueError:
                pass
    out, err = proc.communicate(timeout=600)
    if proc.returncode != 0:
        tail = (err or "").strip().splitlines()[-5:]
        msg = " | ".join(tail) if tail else f"exit {proc.returncode}"
        raise RuntimeError("自动检测失败: " + msg)
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)


def result_to_boxes(result: dict) -> List[Box]:
    """检测结果 JSON → Box 列表（label 标记为「自动」）。"""
    boxes = []
    for b in result.get("boxes", []):
        boxes.append(Box(x=int(b["x"]), y=int(b["y"]), w=int(b["w"]), h=int(b["h"]),
                         start=int(b["start"]), end=int(b["end"]), label="自动"))
    return boxes


def build_project_from_detect(video: str, result: dict) -> Project:
    """检测结果 → 标准工程（格式与手动创建的完全一致）。"""
    fps = float(result.get("fps") or 25)
    nb = int(result.get("nb_frames") or 0)
    p = Project(video=video, output=default_output_path(video), meta={
        "fps": fps, "width": int(result.get("width") or 0),
        "height": int(result.get("height") or 0),
        "duration": nb / fps if fps else 0, "nb_frames": nb})
    p.boxes = result_to_boxes(result)
    for b in p.boxes:
        b.clamp(p.meta["width"], p.meta["height"], nb)
    p.clamp_frames(nb)
    return p


@dataclass
class BatchItem:
    """一条待处理工程。"""
    project_path: str
    project: Project
    video: str
    out: str
    status: str = "待处理"
    frames: int = 0
    seconds: float = 0.0


class EngineLoader(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, model: str = "lama", device: str = "auto", parent=None):
        super().__init__(parent)
        self.model = model
        self.device = device
        self.engine: Optional[InpaintEngine] = None

    def run(self) -> None:
        try:
            self.engine = InpaintEngine(self.model, self.device)
            self.ready.emit(self.engine)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class EraseWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, project: Project, meta: VideoMeta, out_path: str,
                 engine: InpaintEngine, parent=None):
        super().__init__(parent)
        self.project = project
        self.meta = meta
        self.out_path = out_path
        self.engine = engine
        self.task: Optional[EraseTask] = None

    def run(self) -> None:
        try:
            self.task = EraseTask(self.project, self.meta, self.out_path, self.engine)
            stats = self.task.run(
                on_progress=lambda d, t, m: self.progress.emit(d, t, m)
            )
            if stats.get("error"):
                self.failed.emit(stats["error"])
            else:
                self.finished_ok.emit(stats)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def cancel(self) -> None:
        if self.task:
            self.task.cancel()


class BatchWorker(QThread):
    """串行处理多个工程：共用一个已加载的模型，逐个 EraseTask。

    支持断点续跑（输出已存在且非空则跳过）与中途取消（跑完当前任务即停）。
    """
    task_started = Signal(int, str)                 # idx, 输出文件名
    task_progress = Signal(int, int, int, str)      # idx, done, total, msg
    task_done = Signal(int, dict)                   # idx, stats
    task_failed = Signal(int, str)                  # idx, error
    finished_all = Signal(dict)

    def __init__(self, items: List[BatchItem], engine: InpaintEngine,
                 skip_existing: bool = True, parent=None):
        super().__init__(parent)
        self.items = items
        self.engine = engine
        self.skip_existing = skip_existing
        self._cancel = False
        self._current: Optional[EraseTask] = None

    def cancel(self) -> None:
        self._cancel = True
        if self._current:
            self._current.cancel()

    def run(self) -> None:
        summary = {"ok": 0, "fail": 0, "skipped": 0,
                   "frames": 0, "inpainted": 0, "seconds": 0.0, "canceled": False}
        for i, it in enumerate(self.items):
            if self._cancel:
                summary["canceled"] = True
                break
            name = os.path.basename(it.out)
            self.task_started.emit(i, name)
            if self.skip_existing and os.path.exists(it.out) and os.path.getsize(it.out) > 1024:
                it.status = "跳过(已存在)"
                summary["skipped"] += 1
                self.task_done.emit(i, {"skipped": True, "frames": 0, "inpainted": 0,
                                        "seconds": 0.0, "encoder": "-", "error": ""})
                continue
            try:
                meta = probe(it.video)
                it.project.clamp_frames(meta.nb_frames)   # 旧工程展开的哨兵 -> 末帧号
                self._current = EraseTask(it.project, meta, it.out, self.engine)
                stats = self._current.run(
                    on_progress=lambda d, t, m: self.task_progress.emit(i, d, t, m)
                )
                self._current = None
                if stats.get("error"):
                    it.status = "失败"
                    summary["fail"] += 1
                    self.task_failed.emit(i, stats["error"])
                else:
                    it.status = "完成"
                    summary["ok"] += 1
                    summary["frames"] += stats.get("frames", 0)
                    summary["inpainted"] += stats.get("inpainted", 0)
                    summary["seconds"] += stats.get("seconds", 0.0)
                    it.frames = stats.get("frames", 0)
                    it.seconds = stats.get("seconds", 0.0)
                    self.task_done.emit(i, stats)
            except Exception as e:
                it.status = "失败"
                summary["fail"] += 1
                self.task_failed.emit(i, f"{type(e).__name__}: {e}")
        summary["seconds"] = round(summary["seconds"], 2)
        self.finished_all.emit(summary)


class DetectWorker(QThread):
    """单视频自动检测：子进程跑 OCR det，产出 Box 列表。"""
    progress = Signal(int)              # 已处理到的帧号
    finished_ok = Signal(list, str)     # boxes, 摘要
    failed = Signal(str)

    def __init__(self, video: str, step_sec: float = 0.1, iou_thr: float = 0.3,
                 margin: int = 3, thresh: float = 0.6, box_thresh: float = 0.8,
                 rec_score: float = 0.8, min_h: int = 20,
                 bridge_sec: float = 0.5, parent=None):
        super().__init__(parent)
        self.video = video
        self.step_sec = step_sec
        self.iou_thr = iou_thr
        self.margin = margin
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.rec_score = rec_score
        self.min_h = min_h
        self.bridge_sec = bridge_sec

    def run(self) -> None:
        try:
            result = run_detect_subprocess(
                self.video, self.step_sec, self.iou_thr, self.margin,
                on_progress=lambda f: self.progress.emit(f),
                thresh=self.thresh, box_thresh=self.box_thresh,
                rec_score=self.rec_score, min_h=self.min_h,
                bridge_sec=self.bridge_sec)
            boxes = result_to_boxes(result)
            summary = (f"{result.get('sampled', 0)} 采样帧 / "
                       f"{result.get('det_seconds', '?')}s / "
                       f"{'+'.join(result.get('providers', [])[:1])}")
            self.finished_ok.emit(boxes, summary)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class GenerateWorker(QThread):
    """批量自动生成工程：视频 → 检测 → 保存 .suberase.json（格式不变）。

    断点续跑：同目录同名工程文件已存在则跳过。
    """
    item_started = Signal(int, str)             # idx, 视频名
    item_done = Signal(int, str, int)           # idx, 工程路径, 框数
    item_skipped = Signal(int, str)             # idx, 原因
    item_failed = Signal(int, str)              # idx, err
    finished_all = Signal(dict)

    def __init__(self, videos: List[str], step_sec: float = 0.1,
                 iou_thr: float = 0.3, margin: int = 3,
                 thresh: float = 0.6, box_thresh: float = 0.8,
                 rec_score: float = 0.8, min_h: int = 20,
                 bridge_sec: float = 0.5, parent=None):
        super().__init__(parent)
        self.videos = videos
        self.step_sec = step_sec
        self.iou_thr = iou_thr
        self.margin = margin
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.rec_score = rec_score
        self.min_h = min_h
        self.bridge_sec = bridge_sec
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        summary = {"ok": 0, "skip": 0, "fail": 0, "boxes": 0, "canceled": False}
        for i, video in enumerate(self.videos):
            if self._cancel:
                summary["canceled"] = True
                break
            name = os.path.basename(video)
            self.item_started.emit(i, name)
            pj_path = default_project_path(video)
            if os.path.exists(pj_path):
                summary["skip"] += 1
                self.item_skipped.emit(i, f"工程已存在: {os.path.basename(pj_path)}")
                continue
            try:
                result = run_detect_subprocess(
                    video, self.step_sec, self.iou_thr, self.margin,
                    on_progress=lambda f: None,
                    thresh=self.thresh, box_thresh=self.box_thresh,
                    rec_score=self.rec_score, min_h=self.min_h,
                    bridge_sec=self.bridge_sec)
                p = build_project_from_detect(video, result)
                saved = p.save(pj_path)
                summary["ok"] += 1
                summary["boxes"] += len(p.boxes)
                self.item_done.emit(i, saved, len(p.boxes))
            except Exception as e:
                summary["fail"] += 1
                self.item_failed.emit(i, f"{type(e).__name__}: {e}")
        self.finished_all.emit(summary)
