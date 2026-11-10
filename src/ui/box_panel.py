# -*- coding: utf-8 -*-
"""右侧面板：框列表 + 选中框属性（帧区间/坐标）+ 擦除设置。"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
                               QTableWidget, QTableWidgetItem, QAbstractItemView,
                               QSpinBox, QDoubleSpinBox, QPushButton, QComboBox,
                               QCheckBox, QFormLayout, QHeaderView, QLineEdit)

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
        self._nb_frames = 0
        self._updating = False

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ---- 框列表 ----
        frow = QHBoxLayout()
        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("筛选：编号/时段/位置/标签（如 自动）")
        self.ed_filter.setClearButtonEnabled(True)
        self.ed_filter.textChanged.connect(lambda _t: self.refresh())
        self.ck_active = QCheckBox("仅当前帧生效")
        self.ck_active.toggled.connect(lambda _on: self.refresh())
        self.ck_active.setToolTip("只显示当前帧正在生效的框，方便从大量自动框里找")
        frow.addWidget(self.ed_filter, 1)
        frow.addWidget(self.ck_active)
        root.addLayout(frow)

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
        self.sp_start.valueChanged.connect(self._on_start_changed)
        self.sp_end.valueChanged.connect(self._on_end_changed)
        for sp in (self.sp_x, self.sp_y, self.sp_w, self.sp_h):
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
        row3 = QHBoxLayout()
        self.btn_in0 = QPushButton("入点=开头 ([)")
        self.btn_out9 = QPushButton("出点=末尾 (])")
        self.btn_in0.setToolTip("入点未定：从第一帧开始生效（虚线）")
        self.btn_out9.setToolTip("出点未定：一直生效到最后一帧（虚线）")
        self.btn_in0.clicked.connect(self._set_in_head)
        self.btn_out9.clicked.connect(self._set_out_tail)
        row3.addWidget(self.btn_in0)
        row3.addWidget(self.btn_out9)
        form.addRow(row3)
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
        self.sp_talpha = QDoubleSpinBox(); self.sp_talpha.setRange(0.0, 0.95)
        self.sp_talpha.setSingleStep(0.05); self.sp_talpha.setValue(0.6)
        self.sp_talpha.setToolTip("相邻帧修复结果加权平均的强度：0=关闭，越大越稳（镜头摇移不会拖影）")
        self.ck_motion = QCheckBox("运动补偿（镜头摇移）"); self.ck_motion.setChecked(True)
        self.ck_motion.setToolTip("先按镜头位移对齐上一帧结果再平均，避免摇移时产生拖影")
        self.sp_feather = QSpinBox(); self.sp_feather.setRange(0, 16)
        self.sp_feather.setValue(3); self.sp_feather.setSuffix(" px")
        self.sp_feather.setToolTip("向内羽化：只柔化框内边缘，0=硬替换")
        self.ck_realbg = QCheckBox("真实背景优先"); self.ck_realbg.setChecked(True)
        self.ck_realbg.setToolTip("字幕出现前的真实背景对齐后可用时直接贴回（零闪烁、无生成伪影）")
        for w in (self.sp_margin, self.sp_dilate, self.sp_batch, self.sp_crf,
                  self.sp_talpha, self.sp_feather):
            w.valueChanged.connect(self._on_settings_changed)
        self.cb_encoder.currentTextChanged.connect(self._on_settings_changed)
        for ck in (self.ck_audio, self.ck_motion, self.ck_realbg):
            ck.toggled.connect(self._on_settings_changed)
        fs.addRow("外扩 margin", self.sp_margin)
        fs.addRow("mask 膨胀", self.sp_dilate)
        fs.addRow("batch(0=自动)", self.sp_batch)
        fs.addRow("编码器", self.cb_encoder)
        fs.addRow("质量 cq/crf", self.sp_crf)
        fs.addRow(self.ck_audio)
        fs.addRow("时域平滑 α", self.sp_talpha)
        fs.addRow(self.ck_motion)
        fs.addRow("向内羽化", self.sp_feather)
        fs.addRow(self.ck_realbg)
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
        self._nb_frames = nb_frames
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
        self.sp_talpha.setValue(s.temporal_alpha)
        self.ck_motion.setChecked(s.motion_comp)
        self.sp_feather.setValue(s.feather_px)
        self.ck_realbg.setChecked(s.real_bg)
        self._updating = False
        self.refresh()

    def set_frame_idx(self, idx: int) -> None:
        self.frame_idx = idx

    def _visible_boxes(self) -> List[tuple]:
        """[(原始序号, Box)]，按筛选条件过滤。"""
        if not self.project:
            return []
        flt = self.ed_filter.text().strip().lower()
        out = []
        for i, b in enumerate(self.project.boxes):
            if self.ck_active.isChecked() and not b.covers(self.frame_idx):
                continue
            if flt:
                hay = f"#{i + 1} {b.span_text(self._nb_frames)} " \
                      f"{b.x},{b.y} {b.w}x{b.h} {b.label}".lower()
                if flt not in hay:
                    continue
            out.append((i, b))
        return out

    def refresh(self) -> None:
        if not self.project:
            return
        self._updating = True
        visible = self._visible_boxes()
        self.table.setRowCount(len(visible))
        for r, (i, b) in enumerate(visible):
            lo, hi = b.span(self._nb_frames)
            vals = [str(i + 1), b.span_text(self._nb_frames),
                    f"{b.x},{b.y} {b.w}x{b.h}", str(hi - lo + 1)]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setData(Qt.ItemDataRole.UserRole, b.id)
                if not b.enabled:
                    it.setForeground(Qt.GlobalColor.gray)
                self.table.setItem(r, c, it)
        sel = self.current_box()
        if sel:
            self.sp_start.setValue(sel.start)
            self.sp_end.setValue(sel.end)
            self.sp_x.setValue(sel.x); self.sp_y.setValue(sel.y)
            self.sp_w.setValue(sel.w); self.sp_h.setValue(sel.h)
        self._updating = False

    def selected_box_id(self) -> Optional[str]:
        r = self.table.currentRow()
        if r < 0:
            return None
        it = self.table.item(r, 0)
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def current_box(self) -> Optional[Box]:
        bid = self.selected_box_id()
        if not bid or not self.project:
            return None
        for b in self.project.boxes:
            if b.id == bid:
                return b
        return None

    def select_box(self, box_id: Optional[str]) -> None:
        if not box_id:
            self.table.clearSelection()
            return
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.data(Qt.ItemDataRole.UserRole) == box_id:
                self.table.setCurrentCell(r, 0)
                return
        # 目标框被筛选隐藏时，清掉筛选再选
        if self.ed_filter.text() or self.ck_active.isChecked():
            self._updating = True
            self.ed_filter.clear()
            self.ck_active.setChecked(False)
            self._updating = False
            self.refresh()
            self.select_box(box_id)

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
        b.x, b.y = self.sp_x.value(), self.sp_y.value()
        b.w, b.h = self.sp_w.value(), self.sp_h.value()
        self.boxes_changed.emit()

    def _on_start_changed(self, v: int) -> None:
        """手动改起始帧：入点就写这个数字（0 即代表从开头）。"""
        if self._updating or not self.project:
            return
        b = self.current_box()
        if not b:
            return
        b.start = v
        if b.end < b.start:
            b.end = b.start
        self.refresh()
        self.boxes_changed.emit()

    def _on_end_changed(self, v: int) -> None:
        """手动改结束帧：出点就写这个数字（末帧号即代表到末尾）。"""
        if self._updating or not self.project:
            return
        b = self.current_box()
        if not b:
            return
        b.end = v
        if b.start > b.end:
            b.start = b.end
        self.refresh()
        self.boxes_changed.emit()

    def _set_in(self) -> None:
        b = self.current_box()
        if b:
            b.start = self.frame_idx
            if b.end < b.start:
                b.end = b.start
            self.refresh()
            self.boxes_changed.emit()

    def _set_out(self) -> None:
        b = self.current_box()
        if b:
            b.end = self.frame_idx
            if b.start > b.end:
                b.start = b.end
            self.refresh()
            self.boxes_changed.emit()

    def _set_in_head(self) -> None:
        """入点 = 开头：起始帧写成 0。"""
        b = self.current_box()
        if b:
            b.start = 0
            if b.end < b.start:
                b.end = b.start
            self.refresh()
            self.boxes_changed.emit()

    def _set_out_tail(self) -> None:
        """出点 = 末尾：结束帧写成末帧号。"""
        b = self.current_box()
        if b:
            b.end = max(0, (self._nb_frames or b.end + 1) - 1)
            if b.start > b.end:
                b.start = b.end
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
        s.temporal_alpha = self.sp_talpha.value()
        s.motion_comp = self.ck_motion.isChecked()
        s.feather_px = self.sp_feather.value()
        s.real_bg = self.ck_realbg.isChecked()
