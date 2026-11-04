# -*- coding: utf-8 -*-
"""时间轴：总长 + 各框区间条（分轨道显示）+ 当前帧指针，点击/拖动跳转。

分轨道规则：按生效区间起点排序，贪心放入第一条"与该轨道已有框时间不重叠"的
轨道；重叠的框落到下一轨。轨道数上限 MAX_TRACKS，放不下的框画在最后一轨并
加虚线边框提示。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import QWidget

from ..project import Box
from .frame_view import color_of

TRACK_H = 14          # 每条轨道条高
TRACK_GAP = 3         # 轨道间隙
MAX_TRACKS = 4        # 轨道上限
PAD_TOP = 8           # 顶部留白（指针圆头）
PAD_BOTTOM = 8        # 底部留白（刻度）


class TimelineWidget(QWidget):
    seeked = Signal(int)
    box_clicked = Signal(str)     # 点击轨道上的框色块 -> box_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.total = 1
        self.frame = 0
        self.boxes: List[Box] = []
        self._dragging = False
        self._tracks: List[List[Tuple[int, int, int]]] = []   # 每轨: [(box_idx, lo, hi)]
        self._overflow: List[Tuple[int, int, int]] = []
        self._set_height(1)

    # ---------------- 数据 ----------------
    def set_meta(self, total: int) -> None:
        self.total = max(1, total)
        self._assign_tracks()
        self.update()

    def set_boxes(self, boxes: List[Box]) -> None:
        self.boxes = boxes
        self._assign_tracks()
        self.update()

    def set_frame(self, idx: int) -> None:
        self.frame = idx
        self.update()

    # ---------------- 轨道分配 ----------------
    def _assign_tracks(self) -> None:
        order = sorted(
            range(len(self.boxes)),
            key=lambda i: self.boxes[i].span(self.total)[0])
        tracks: List[List[Tuple[int, int, int]]] = []
        overflow: List[Tuple[int, int, int]] = []
        for i in order:
            lo, hi = self.boxes[i].span(self.total)
            placed = False
            for tr in tracks:
                # 轨道内已按 lo 升序，只需跟最后一个比
                if lo > tr[-1][2]:
                    tr.append((i, lo, hi))
                    placed = True
                    break
            if not placed:
                if len(tracks) < MAX_TRACKS:
                    tracks.append([(i, lo, hi)])
                else:
                    overflow.append((i, lo, hi))
        for tr in tracks:
            tr.sort(key=lambda t: t[0])     # 轨道内保持列表序号顺序
        self._tracks = tracks
        self._overflow = overflow
        self._set_height(max(1, len(tracks)))

    def _set_height(self, n_tracks: int) -> None:
        h = PAD_TOP + n_tracks * TRACK_H + max(0, n_tracks - 1) * TRACK_GAP + PAD_BOTTOM
        self.setFixedHeight(h)

    # ---------------- 坐标 ----------------
    def _x_of(self, frame: int) -> int:
        return int(frame / self.total * self.width())

    def _frame_at(self, x: int) -> int:
        r = max(0.0, min(1.0, x / max(1, self.width())))
        return int(round(r * (self.total - 1)))

    def _track_rect(self, tr_idx: int) -> QRect:
        y = PAD_TOP + tr_idx * (TRACK_H + TRACK_GAP)
        return QRect(0, y, self.width(), TRACK_H)

    def _box_rect(self, lo: int, hi: int, tr_idx: int) -> QRect:
        tr = self._track_rect(tr_idx)
        x1 = self._x_of(lo)
        x2 = max(x1 + 2, self._x_of(hi + 1))
        return QRect(x1, tr.y(), x2 - x1, tr.height())

    # ---------------- 交互 ----------------
    def _box_at(self, p) -> Optional[str]:
        """命中轨道色块则返回 box_id（自上而下找）。"""
        for ti, tr in enumerate(self._tracks):
            if not self._track_rect(ti).adjusted(0, -2, 0, 2).contains(p):
                continue
            for i, lo, hi in tr:
                if self._box_rect(lo, hi, ti).adjusted(-1, 0, 1, 0).contains(p):
                    return self.boxes[i].id
            break
        if self._overflow and self._tracks:      # 溢出框叠画在最后一轨
            last = len(self._tracks) - 1
            for i, lo, hi in self._overflow:
                if self._box_rect(lo, hi, last).contains(p):
                    return self.boxes[i].id
        return None

    def mousePressEvent(self, e: QMouseEvent) -> None:
        p = e.position().toPoint()
        self._dragging = True
        bid = self._box_at(p)
        if bid:
            self.box_clicked.emit(bid)
        self.seeked.emit(self._frame_at(p.x()))

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._dragging:
            self.seeked.emit(self._frame_at(e.position().toPoint().x()))

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._dragging = False

    # ---------------- 绘制 ----------------
    def paintEvent(self, e: QPaintEvent) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#202024"))
        w, h = self.width(), self.height()

        # 轨道底 + 框区间
        for ti, tr in enumerate(self._tracks):
            p.fillRect(self._track_rect(ti), QColor("#2c2c33"))
            p.setPen(QPen(QColor("#3a3a44")))
            p.drawRect(self._track_rect(ti))
            for i, lo, hi in tr:
                col = color_of(i)
                if not self.boxes[i].enabled:
                    col = QColor("#666")
                p.fillRect(self._box_rect(lo, hi, ti), col)

        # 超出上限的框：叠画在最后一轨，虚线边框提示
        if self._overflow and self._tracks:
            last = len(self._tracks) - 1
            for i, lo, hi in self._overflow:
                col = QColor(color_of(i))
                if not self.boxes[i].enabled:
                    col = QColor("#666")
                col.setAlpha(150)
                p.fillRect(self._box_rect(lo, hi, last), col)
                p.setPen(QPen(QColor("#ffaa00"), 1, Qt.PenStyle.DashLine))
                p.drawRect(self._box_rect(lo, hi, last))

        # 刻度（每 1/10，画在轨道区下方）
        n = max(1, len(self._tracks))
        tick_y = PAD_TOP + n * TRACK_H + (n - 1) * TRACK_GAP
        p.setPen(QPen(QColor("#555")))
        for i in range(1, 10):
            x = int(w * i / 10)
            p.drawLine(x, tick_y, x, tick_y + 4)

        # 当前帧指针
        px = self._x_of(self.frame)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(px, 2, px, h - 2)
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(px - 4, 0, 8, 8)
