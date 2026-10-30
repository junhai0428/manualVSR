# -*- coding: utf-8 -*-
"""右侧面板：框列表 + 选中框属性（帧区间/坐标）+ 擦除设置。"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                               QTableWidget, QTableWidgetItem, QAbstractItemView,
                               QSpinBox, QPushButton, QComboBox, QCheckBox, QFormLayout,
                               QHeaderView)

from ..project import Box, Project


class BoxPanel(QWidget):
    selection_changed = Signal(object)   # box_id or None
    boxes_changed = Signal()
    delete_requested = Signal(str)
    goto_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self.frame_idx = 0
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ---- 框列表 ----
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "时段(帧)", "位置", "帧数"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.currentCellChanged.connect(self._on_row_changed)
        root.addWidget(self.table, 3)

        row = QHBoxLayout()
        self.btn_del = QPushButton("删除选中 (Del)")
        self.btn_del.clicked.connect(self._on_delete)
        self.btn_goto = QPushButton("跳到入点")
        self.btn_goto.clicked.connect(self._on_goto)
        row.addWidget(self.btn_del)
        row.addWidget(self.btn_goto)
        root.addLayout(row)

        # ---- 选中框属性 ----
        gb = QGroupBox("选中框  (I=入点  O=出点)")
        form = QFormLayout(gb)
        self.sp_start = QSpinBox(); self.sp_start.setRange(0, 10_000_000)
        self.sp_end = QSpinBox(); self.sp_end.setRange(0, 10_000_000)
        self.sp_x = QSpinBox(); self.sp_x.setRange(0, 100000)
        self.sp_y = QSpinBox(); self.sp_y.setRange(0, 100000)
        self.sp_w = QSpinBox(); self.sp_w.setRange(1, 100000)
        self.sp_h = QSpinBox(); self.sp_h.setRange(1, 100000)
        for sp in (self.sp_start, self.sp_end, self.sp_x, self.sp_y, self.sp_w, self.sp_h):
            sp.valueChanged.connect(self._on_prop_changed)
        form.addRow("起始帧", self.sp_start)
        form.addRow("结束帧", self.sp_end)
        form.addRow("X / Y", self._pair(self.sp_x, self.sp_y))
        form.addRow("宽 / 高", self._pair(self.sp_w, self.sp_h))
        row2 = QHBoxLayout()
        self.btn_in = QPushButton("入点=当前 (I)")
        self.btn_out = QPushButton("出点=当前 (O)")
        self.btn_in.clicked.connect(self._set_in)
        self.btn_out.clicked.connect(self._set_out)
        row2.addWidget(self.btn_in)
        row2.addWidget(self.btn_out)
        form.addRow(row2)
        root.addWidget(gb)

        # ---- 设置 ----
        gs = QGroupBox("擦除设置")
        fs = QFormLayout(gs)
        self.sp_margin = QSpinBox(); self.sp_margin.setRange(0, 512); self.sp_margin.setValue(48)
        self.sp_dilate = QSpinBox(); self.sp_dilate.setRange(0, 64); self.sp_dilate.setValue(3)
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(0, 32); self.sp_batch.setValue(0)
        self.sp_batch.setToolTip("0 = 按显存预算自动")
        self.cb_encoder = QComboBox(); self.cb_encoder.addItems(["auto", "nvenc", "x264"])
        self.sp_crf = QSpinBox(); self.sp_crf.setRange(10, 35); self.sp_crf.setValue(18)
        self.ck_audio = QCheckBox("保留原音轨"); self.ck_audio.setChecked(True)
        for w in (self.sp_margin, self.sp_dilate, self.sp_batch, self.sp_crf):
            w.valueChanged.connect(self._on_settings_changed)
        self.cb_encoder.currentTextChanged.connect(self._on_settings_changed)
        self.ck_audio.toggled.connect(self._on_settings_changed)
        fs.addRow("外扩 margin", self.sp_margin)
        fs.addRow("mask 膨胀", self.sp_dilate)
        fs.addRow("batch(0=自动)", self.sp_batch)
        fs.addRow("编码器", self.cb_encoder)
        fs.addRow("质量 cq/crf", self.sp_crf)
        fs.addRow(self.ck_audio)
        root.addWidget(gs)
        root.addStretch(1)

    @staticmethod
    def _pair(a: QWidget, b: QWidget) -> QWidget:
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(a)
        l.addWidget(b)
        return w

    # ---------- 数据绑定 ----------
    def set_project(self, project: Project, nb_frames: int) -> None:
        self.project = project
        for sp in (self.sp_start, self.sp_end):
            sp.setMaximum(max(0, nb_frames - 1))
        s = project.settings
        self._updating = True
        self.sp_margin.setValue(s.margin)
        self.sp_dilate.setValue(s.dilate)
        self.sp_batch.setValue(s.batch_size)
        self.sp_crf.setValue(s.crf)
        self.cb_encoder.setCurrentText(s.encoder)
        self.ck_audio.setChecked(s.keep_audio)
        self._updating = False
        self.refresh()

    def set_frame_idx(self, idx: int) -> None:
        self.frame_idx = idx

    def refresh(self) -> None:
        if not self.project:
            return
        self._updating = True
        self.table.setRowCount(len(self.project.boxes))
        for i, b in enumerate(self.project.boxes):
            vals = [str(i + 1), f"{b.start} - {b.end}",
                    f"{b.x},{b.y} {b.w}x{b.h}", str(b.end - b.start + 1)]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(Qt.ItemDataRole.UserRole, b.id)
                if not b.enabled:
                    it.setForeground(Qt.GlobalColor.gray)
                self.table.setItem(i, c, it)
        sel = self.current_box()
        if sel:
            self.sp_start.setValue(sel.start)
            self.sp_end.setValue(sel.end)
            self.sp_x.setValue(sel.x); self.sp_y.setValue(sel.y)
            self.sp_w.setValue(sel.w); self.sp_h.setValue(sel.h)
        self._updating = False

    def selected_box_id(self) -> Optional[str]:
        r = self.table.currentRow()
        if r < 0 or not self.project or r >= len(self.project.boxes):
            return None
        return self.project.boxes[r].id

    def current_box(self) -> Optional[Box]:
        bid = self.selected_box_id()
        if not bid or not self.project:
            return None
        for b in self.project.boxes:
            if b.id == bid:
                return b
        return None

    def select_box(self, box_id: Optional[str]) -> None:
        if not self.project or not box_id:
            self.table.clearSelection()
            return
        for i, b in enumerate(self.project.boxes):
            if b.id == box_id:
                self.table.setCurrentCell(i, 0)
                return

    # ---------- 交互 ----------
    def _on_row_changed(self, r, c, pr, pc) -> None:
        b = self.current_box()
        if b and not self._updating:
            self._updating = True
            self.sp_start.setValue(b.start); self.sp_end.setValue(b.end)
            self.sp_x.setValue(b.x); self.sp_y.setValue(b.y)
            self.sp_w.setValue(b.w); self.sp_h.setValue(b.h)
            self._updating = False
        self.selection_changed.emit(b.id if b else None)

    def _on_prop_changed(self, _v=None) -> None:
        if self._updating or not self.project:
            return
        b = self.current_box()
        if not b:
            return
        b.start = min(self.sp_start.value(), self.sp_end.value())
        b.end = max(self.sp_start.value(), self.sp_end.value())
        b.x, b.y = self.sp_x.value(), self.sp_y.value()
        b.w, b.h = self.sp_w.value(), self.sp_h.value()
        self.boxes_changed.emit()

    def _set_in(self) -> None:
        b = self.current_box()
        if b:
            b.start = min(self.frame_idx, b.end)
            self.refresh()
            self.boxes_changed.emit()

    def _set_out(self) -> None:
        b = self.current_box()
        if b:
            b.end = max(self.frame_idx, b.start)
            self.refresh()
            self.boxes_changed.emit()

    def _on_delete(self) -> None:
        bid = self.selected_box_id()
        if bid:
            self.delete_requested.emit(bid)

    def _on_goto(self) -> None:
        b = self.current_box()
        if b:
            self.goto_requested.emit(b.start)

    def _on_settings_changed(self, _v=None) -> None:
        if self._updating or not self.project:
            return
        s = self.project.settings
        s.margin = self.sp_margin.value()
        s.dilate = self.sp_dilate.value()
        s.batch_size = self.sp_batch.value()
        s.encoder = self.cb_encoder.currentText()
        s.crf = self.sp_crf.value()
        s.keep_audio = self.ck_audio.isChecked()
