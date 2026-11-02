# -*- coding: utf-8 -*-
"""时间轴：总长 + 各框区间条 + 当前帧指针，点击/拖动跳转。"""
from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import QWidget

from ..project import Box
from .frame_view import color_of


class TimelineWidget(QWidget):
    seeked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self.setMinimumWidth(300)
        self.total = 1
        self.frame = 0
        self.boxes: List[Box] = []
        self._dragging = False

    def set_meta(self, total: int) -> None:
        self.total = max(1, total)
        self.update()

    def set_boxes(self, boxes: List[Box]) -> None:
        self.boxes = boxes
        self.update()

    def set_frame(self, idx: int) -> None:
        self.frame = idx
        self.update()

    def _x_of(self, frame: int) -> int:
        return int(frame / self.total * self.width())

    def _frame_at(self, x: int) -> int:
        r = max(0.0, min(1.0, x / max(1, self.width())))
        return int(round(r * (self.total - 1)))

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self._dragging = True
        self.seeked.emit(self._frame_at(e.position().toPoint().x()))

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._dragging:
            self.seeked.emit(self._frame_at(e.position().toPoint().x()))

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._dragging = False

    def paintEvent(self, e: QPaintEvent) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#202024"))
        w, h = self.width(), self.height()

        track = QRect(0, 8, w, 28)
        p.fillRect(track, QColor("#2c2c33"))
        p.setPen(QPen(QColor("#3a3a44")))
        p.drawRect(track)

        # 刻度（每 1/10）
        p.setPen(QPen(QColor("#555")))
        for i in range(1, 10):
            x = int(w * i / 10)
            p.drawLine(x, track.bottom(), x, track.bottom() + 4)

        # 框区间
        for i, b in enumerate(self.boxes):
            lo, hi = b.span(self.total)      # 未定端展开到开头/末尾
            x1 = self._x_of(lo)
            x2 = max(x1 + 2, self._x_of(hi + 1))
            col = color_of(i)
            if not b.enabled:
                col = QColor("#666")
            p.fillRect(QRect(x1, 10, x2 - x1, 24), col)

        # 当前帧指针
        px = self._x_of(self.frame)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(px, 4, px, h - 4)
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(px - 4, 2, 8, 8)
