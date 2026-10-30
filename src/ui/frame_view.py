# -*- coding: utf-8 -*-
"""画面显示 + 框选交互：拖框新建、拖动移动、四角缩放。

坐标系：框始终以源视频像素存储，显示时按缩放比映射。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QRect, QRectF, QPoint, Signal
from PySide6.QtGui import (QColor, QImage, QPainter, QPen, QFont, QBrush,
                           QMouseEvent, QPaintEvent)
from PySide6.QtWidgets import QWidget

from ..project import Box

HANDLE = 8
COLORS = ["#ff5252", "#4caf50", "#2196f3", "#ff9800", "#9c27b0", "#00bcd4", "#e91e63"]


def color_of(idx: int) -> QColor:
    return QColor(COLORS[idx % len(COLORS)])


class FrameView(QWidget):
    box_created = Signal(object)
    box_changed = Signal()
    selection_changed = Signal(object)   # Box or None

    MODE_NONE, MODE_NEW, MODE_MOVE, MODE_RESIZE = 0, 1, 2, 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        self._img: Optional[QImage] = None
        self._rgb: Optional[np.ndarray] = None
        self.video_w = 0
        self.video_h = 0
        self.view_rect = QRect()      # 画面在控件中的显示区域
        self.boxes: List[Box] = []
        self.frame_idx = 0
        self.selected_id: Optional[str] = None
        self.preview_mask = False

        self.mode = self.MODE_NONE
        self._start = QPoint()
        self._orig: Optional[Tuple[int, int, int, int]] = None
        self._handle = 0
        self._temp_rect: Optional[QRect] = None

    # ---------- 帧 ----------
    def set_frame(self, bgr: Optional[np.ndarray]) -> None:
        if bgr is None:
            self._img = None
            self.update()
            return
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        self._rgb = rgb  # 保持引用，避免 data 失效
        self._img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.video_w, self.video_h = w, h
        self.update_view_rect()
        self.update()

    def set_boxes(self, boxes: List[Box]) -> None:
        self.boxes = boxes

    def set_frame_idx(self, idx: int) -> None:
        self.frame_idx = idx
        self.update()

    def set_selected(self, box_id: Optional[str]) -> None:
        self.selected_id = box_id
        self.update()

    # ---------- 坐标换算 ----------
    def _compute_view_rect(self) -> QRect:
        if not self._img or self.video_w == 0:
            return QRect()
        cw, ch = self.width(), self.height()
        scale = min(cw / self.video_w, ch / self.video_h)
        w = int(self.video_w * scale)
        h = int(self.video_h * scale)
        return QRect((cw - w) // 2, (ch - h) // 2, w, h)

    def update_view_rect(self) -> None:
        """显式刷新显示区域：offscreen/未重绘时也要保证坐标换算可用。"""
        self.view_rect = self._compute_view_rect()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.update_view_rect()

    def scale(self) -> float:
        return self.view_rect.width() / self.video_w if self.video_w else 1.0

    def to_widget(self, x: int, y: int) -> QPoint:
        s = self.scale()
        return QPoint(int(self.view_rect.x() + x * s), int(self.view_rect.y() + y * s))

    def to_video(self, p: QPoint) -> QPoint:
        s = self.scale() or 1.0
        return QPoint(int((p.x() - self.view_rect.x()) / s),
                      int((p.y() - self.view_rect.y()) / s))

    def box_rect_widget(self, b: Box) -> QRect:
        p1 = self.to_widget(b.x, b.y)
        p2 = self.to_widget(b.x + b.w, b.y + b.h)
        return QRect(p1, p2)

    # ---------- 命中测试 ----------
    def _active_boxes(self) -> List[Box]:
        return [b for b in self.boxes if b.covers(self.frame_idx)]

    def _hit_handle(self, b: Box, p: QPoint) -> int:
        r = self.box_rect_widget(b)
        pts = [r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]
        for i, hp in enumerate(pts):
            if QRect(hp.x() - HANDLE // 2, hp.y() - HANDLE // 2, HANDLE, HANDLE).contains(p):
                return i + 1
        return 0

    # ---------- 鼠标 ----------
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if not self._img:
            return
        p = e.position().toPoint()
        sel = self._box_by_id(self.selected_id)
        if sel is not None:
            h = self._hit_handle(sel, p)
            if h:
                self.mode = self.MODE_RESIZE
                self._handle = h
                self._orig = (sel.x, sel.y, sel.w, sel.h)
                self._start = p
                return
        # 所有框都可点选（非活跃框也要能选中，否则无法再改它的时段）
        for b in reversed(self.boxes):
            if self.box_rect_widget(b).contains(p):
                self.mode = self.MODE_MOVE
                self._orig = (b.x, b.y, b.w, b.h)
                self._start = p
                self.selected_id = b.id
                self.selection_changed.emit(b)
                self.update()
                return
        self.mode = self.MODE_NEW
        self._start = p
        self._temp_rect = QRect(p, p)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.mode == self.MODE_NONE:
            return
        p = e.position().toPoint()
        if self.mode == self.MODE_NEW:
            self._temp_rect = QRect(self._start, p).normalized()
        elif self.mode in (self.MODE_MOVE, self.MODE_RESIZE):
            b = self._box_by_id(self.selected_id)
            if b is None or self._orig is None:
                return
            if self.mode == self.MODE_MOVE:
                dp = self.to_video(p) - self.to_video(self._start)
                b.x = self._orig[0] + dp.x()
                b.y = self._orig[1] + dp.y()
            else:
                ox, oy, ow, oh = self._orig
                cur = self.to_video(p)
                if self._handle in (1, 4):      # 左上 / 左下 -> 影响 x,w
                    nx = min(cur.x(), ox + ow - 4)
                    b.x, b.w = nx, ox + ow - nx
                if self._handle in (2, 3):
                    b.w = max(4, cur.x() - ox)
                if self._handle in (1, 2):
                    ny = min(cur.y(), oy + oh - 4)
                    b.y, b.h = ny, oy + oh - ny
                if self._handle in (3, 4):
                    b.h = max(4, cur.y() - oy)
            self.box_changed.emit()
        self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self.mode == self.MODE_NEW:
            r = self._temp_rect
            self._temp_rect = None
            if r is not None and r.width() > 4 and r.height() > 4:
                p1 = self.to_video(r.topLeft())
                p2 = self.to_video(r.bottomRight())
                b = Box(x=max(0, p1.x()), y=max(0, p1.y()),
                        w=max(4, p2.x() - p1.x()), h=max(4, p2.y() - p1.y()),
                        start=self.frame_idx, end=self.frame_idx)
                self.selected_id = b.id
                self.box_created.emit(b)
                self.selection_changed.emit(b)
        elif self.mode in (self.MODE_MOVE, self.MODE_RESIZE):
            self.box_changed.emit()
        self.mode = self.MODE_NONE
        self.update()

    def _box_by_id(self, box_id: Optional[str]) -> Optional[Box]:
        for b in self.boxes:
            if b.id == box_id:
                return b
        return None

    # ---------- 绘制 ----------
    def paintEvent(self, e: QPaintEvent) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1b1b1f"))
        self.view_rect = self._compute_view_rect()
        if self._img:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(QRectF(self.view_rect), self._img)
        else:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "打开视频后在此框选文字")
            return

        for i, b in enumerate(self.boxes):
            active = b.covers(self.frame_idx)
            sel = b.id == self.selected_id
            col = color_of(i)
            pen = QPen(col, 2 if sel else 1)
            pen.setStyle(Qt.PenStyle.SolidLine if active else Qt.PenStyle.DashLine)
            if not active:
                col = QColor(col)
                col.setAlpha(90)
                pen.setColor(col)
            p.setPen(pen)
            r = self.box_rect_widget(b)
            p.drawRect(r)
            if sel:
                p.setBrush(QBrush(col))
                for hp in (r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()):
                    p.drawRect(QRect(hp.x() - 3, hp.y() - 3, 6, 6))
                p.setBrush(Qt.BrushStyle.NoBrush)
            if active or sel:
                p.setPen(QPen(QColor("#ffffff")))
                f = QFont()
                f.setPointSize(8)
                p.setFont(f)
                p.drawText(r.topLeft().x(), max(10, r.topLeft().y() - 4),
                           f"#{i+1} {b.start}-{b.end}")

        if self._temp_rect:
            p.setPen(QPen(QColor("#ffe082"), 1, Qt.PenStyle.DashLine))
            p.drawRect(self._temp_rect)
