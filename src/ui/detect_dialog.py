# -*- coding: utf-8 -*-
"""自动检测参数对话框。"""
from __future__ import annotations

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QSpinBox)


class DetectParamsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自动检测参数")
        form = QFormLayout(self)

        self.sp_step = QDoubleSpinBox()
        self.sp_step.setRange(0.1, 5.0)
        self.sp_step.setSingleStep(0.1)
        self.sp_step.setValue(0.5)
        self.sp_step.setSuffix(" s")
        self.sp_step.setToolTip("隔多久抽一帧做检测：越小越准越慢")
        form.addRow("抽帧间隔", self.sp_step)

        self.sp_iou = QDoubleSpinBox()
        self.sp_iou.setRange(0.05, 0.9)
        self.sp_iou.setSingleStep(0.05)
        self.sp_iou.setValue(0.3)
        self.sp_iou.setToolTip("相邻采样帧检测框的重合度阈值，低于它视为不同文字")
        form.addRow("合并 IoU", self.sp_iou)

        self.sp_margin = QSpinBox()
        self.sp_margin.setRange(0, 64)
        self.sp_margin.setValue(8)
        self.sp_margin.setSuffix(" px")
        self.sp_margin.setToolTip("检测框向外扩展，避免贴字太紧擦不干净")
        form.addRow("外扩边距", self.sp_margin)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> tuple:
        return (self.sp_step.value(), self.sp_iou.value(), self.sp_margin.value())
