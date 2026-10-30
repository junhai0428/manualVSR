# -*- coding: utf-8 -*-
"""后台线程：模型加载、擦除任务。GUI 主线程只做显示与进度更新。"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QThread, Signal

from .inpaint_engine import InpaintEngine
from .eraser import EraseTask
from .project import Project
from .video_io import VideoMeta


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
