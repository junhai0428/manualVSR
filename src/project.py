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


END_OPEN = 10 ** 9      # 哨兵：出点未定，加载工程后由 clamp 夹成真实末帧号


@dataclass
class Box:
    """坐标系：源视频原始像素。start/end 为闭区间帧号（存盘就是具体数字）。

    半开放语义直接用边界值表达，不额外存标志位：
      start == 0        -> 从开头生效（列表显示「开头」）
      end  == 末帧号    -> 一直生效到末尾（列表显示「末尾」）
    新建框默认 end = 末帧号，即出点未定；按 O 后写成当前帧号。
    """
    x: int
    y: int
    w: int
    h: int
    start: int
    end: int
    enabled: bool = True
    label: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def rect(self) -> tuple:
        return (self.x, self.y, self.w, self.h)

    def covers(self, frame: int) -> bool:
        return self.enabled and self.start <= frame <= self.end

    def clamp(self, width: int, height: int, nb_frames: int) -> None:
        self.x = max(0, min(self.x, width - 1))
        self.y = max(0, min(self.y, height - 1))
        self.w = max(1, min(self.w, width - self.x))
        self.h = max(1, min(self.h, height - self.y))
        self.start = max(0, min(self.start, max(0, nb_frames - 1)))
        self.end = max(self.start, min(self.end, max(0, nb_frames - 1)))

    # ---- 显示/查询辅助 ----
    def is_open(self, nb_frames: int = 0) -> bool:
        """有一端贴在开头/末尾 = 半开放（虚线显示）。"""
        return self.start <= 0 or (nb_frames > 0 and self.end >= nb_frames - 1)

    def span(self, nb_frames: int = 0) -> tuple:
        """实际生效区间（夹到视频范围内）。"""
        lo = max(0, self.start)
        hi = min(self.end, nb_frames - 1) if nb_frames > 0 else self.end
        return lo, max(lo, hi)

    def span_text(self, nb_frames: int = 0) -> str:
        at_head = self.start <= 0
        at_tail = nb_frames > 0 and self.end >= nb_frames - 1
        if at_head and at_tail:
            return "全片"
        if at_head:
            return f"开头 - {self.end}"
        if at_tail:
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
    # 时域稳定（消除镜头摇移下逐帧修复的闪烁）
    temporal_alpha: float = 0.6   # 0 = 关闭时域平滑
    motion_comp: bool = True      # 按镜头位移对齐后再平均
    feather_px: int = 3           # 向内羽化像素，0 = 硬替换
    real_bg: bool = True          # 优先使用相邻无字幕帧的真实背景


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

    def replace_boxes(self, new_boxes: List[Box]) -> int:
        """重跑自动检测：清掉原有全部框（不分手动/自动），放入新框。

        返回被清除的旧框数量。
        """
        old = len(self.boxes)
        self.boxes = list(new_boxes)
        return old

    def active_boxes(self, frame: int) -> List[Box]:
        return [b for b in self.boxes if b.covers(frame)]

    def need_process(self) -> bool:
        return any(b.enabled for b in self.boxes)

    def clamp_frames(self, nb_frames: int) -> None:
        """把帧号夹进视频范围（含旧工程展开出的哨兵值）。"""
        last = max(0, nb_frames - 1)
        for b in self.boxes:
            b.start = max(0, min(b.start, last))
            b.end = max(b.start, min(b.end, last))

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
                start, end = int(b["start"]), int(b["end"])
                # 兼容旧工程（曾用 open_start/open_end 标志）：展开成边界帧号
                if b.get("open_start"):
                    start = 0
                if b.get("open_end"):
                    end = END_OPEN      # 稍后由 clamp_frames/clamp 夹成真实末帧号
                box = Box(
                    x=int(b["x"]), y=int(b["y"]), w=int(b["w"]), h=int(b["h"]),
                    start=start, end=end,
                    enabled=bool(b.get("enabled", True)),
                    label=str(b.get("label", "")),
                    id=str(b.get("id") or uuid.uuid4().hex[:8]),
                )
                p.boxes.append(box)
            except (KeyError, ValueError, TypeError):
                continue
        nb = int(p.meta.get("nb_frames") or 0)
        if nb > 0:
            p.clamp_frames(nb)
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
