# -*- coding: utf-8 -*-
"""工程模型：框（像素坐标 + 帧区间）、工程文件读写。"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

PROJECT_VERSION = 1
PROJECT_EXT = ".suberase.json"


@dataclass
class Box:
    """坐标系：源视频原始像素。start/end 为闭区间帧号。

    open_end=True  : 出点未定，时段从 start 延伸到末尾（新建框的默认状态）
    open_start=True: 入点未定，时段从开头延伸到 end
    两个都为 True 即全片生效。
    """
    x: int
    y: int
    w: int
    h: int
    start: int
    end: int
    enabled: bool = True
    label: str = ""
    open_end: bool = False
    open_start: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def rect(self) -> tuple:
        return (self.x, self.y, self.w, self.h)

    def covers(self, frame: int) -> bool:
        if not self.enabled:
            return False
        lo = 0 if self.open_start else self.start
        hi = 10 ** 9 if self.open_end else self.end
        return lo <= frame <= hi

    def clamp(self, width: int, height: int, nb_frames: int) -> None:
        self.x = max(0, min(self.x, width - 1))
        self.y = max(0, min(self.y, height - 1))
        self.w = max(1, min(self.w, width - self.x))
        self.h = max(1, min(self.h, height - self.y))
        self.start = max(0, min(self.start, nb_frames - 1))
        self.end = max(self.start, min(self.end, nb_frames - 1))

    # ---- 显示/查询辅助 ----
    def is_open(self) -> bool:
        """任一端未定 = 半开放框（虚线显示）。"""
        return self.open_start or self.open_end

    def span(self, nb_frames: int) -> tuple:
        """实际生效的帧区间（含未定端的展开）。"""
        lo = 0 if self.open_start else self.start
        hi = max(0, nb_frames - 1) if self.open_end else self.end
        return lo, hi

    def span_text(self) -> str:
        if self.open_start and self.open_end:
            return "全片"
        if self.open_start:
            return f"开头 - {self.end}"
        if self.open_end:
            return f"{self.start} - 末尾"
        return f"{self.start} - {self.end}"


@dataclass
class Settings:
    margin: int = 48          # crop 区域在框外扩展的像素
    dilate: int = 3           # mask 膨胀像素
    batch_size: int = 0       # 0 = 按显存预算自动
    encoder: str = "auto"     # auto / nvenc / x264
    crf: int = 18
    keep_audio: bool = True
    model: str = "lama"


@dataclass
class Project:
    version: int = PROJECT_VERSION
    video: str = ""
    output: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    boxes: List[Box] = field(default_factory=list)
    settings: Settings = field(default_factory=Settings)

    def add_box(self, box: Box) -> Box:
        self.boxes.append(box)
        return box

    def remove_box(self, box_id: str) -> bool:
        for i, b in enumerate(self.boxes):
            if b.id == box_id:
                self.boxes.pop(i)
                return True
        return False

    def active_boxes(self, frame: int) -> List[Box]:
        return [b for b in self.boxes if b.covers(frame)]

    def need_process(self) -> bool:
        return any(b.enabled for b in self.boxes)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "video": self.video,
            "output": self.output,
            "meta": self.meta,
            "boxes": [asdict(b) for b in self.boxes],
            "settings": asdict(self.settings),
        }

    @staticmethod
    def from_dict(d: dict) -> "Project":
        p = Project()
        p.version = int(d.get("version", 1))
        p.video = d.get("video", "")
        p.output = d.get("output", "")
        p.meta = d.get("meta", {}) or {}
        p.boxes = []
        for b in d.get("boxes", []) or []:
            try:
                box = Box(
                    x=int(b["x"]), y=int(b["y"]), w=int(b["w"]), h=int(b["h"]),
                    start=int(b["start"]), end=int(b["end"]),
                    enabled=bool(b.get("enabled", True)),
                    label=str(b.get("label", "")),
                    open_end=bool(b.get("open_end", False)),
                    open_start=bool(b.get("open_start", False)),
                    id=str(b.get("id") or uuid.uuid4().hex[:8]),
                )
                p.boxes.append(box)
            except (KeyError, ValueError, TypeError):
                continue
        s = d.get("settings", {}) or {}
        p.settings = Settings(**{k: v for k, v in s.items() if k in Settings.__dataclass_fields__})
        return p

    def save(self, path: Optional[str] = None) -> str:
        path = path or default_project_path(self.video)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as f:
            return Project.from_dict(json.load(f))


def default_project_path(video: str) -> str:
    base, _ = os.path.splitext(video)
    return base + PROJECT_EXT


def default_output_path(video: str) -> str:
    base, ext = os.path.splitext(video)
    return f"{base}_clean{ext or '.mp4'}"
