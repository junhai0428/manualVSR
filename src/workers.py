# -*- coding: utf-8 -*-
"""后台线程：模型加载、单任务擦除、批量擦除。GUI 主线程只做显示与进度更新。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

from .inpaint_engine import InpaintEngine
from .eraser import EraseTask
from .project import Project
from .video_io import VideoMeta, probe


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
