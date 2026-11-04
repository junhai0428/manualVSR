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
        self.sp_step.setSingleStep(0.05)
        self.sp_step.setValue(0.1)
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
        self.sp_margin.setValue(3)
        self.sp_margin.setSuffix(" px")
        self.sp_margin.setToolTip("检测框向外扩展，避免贴字太紧擦不干净")
        form.addRow("外扩边距", self.sp_margin)

        self.sp_thresh = QDoubleSpinBox()
        self.sp_thresh.setRange(0.05, 0.9)
        self.sp_thresh.setSingleStep(0.05)
        self.sp_thresh.setValue(0.6)
        self.sp_thresh.setToolTip("DBNet 概率图二值化阈值：调小框更松更全（能框住浅色/模糊字），调大框更紧更准")
        form.addRow("二值化阈值 thresh", self.sp_thresh)

        self.sp_box = QDoubleSpinBox()
        self.sp_box.setRange(0.1, 0.9)
        self.sp_box.setSingleStep(0.05)
        self.sp_box.setValue(0.8)
        self.sp_box.setToolTip("候选框整体置信度门槛：调小保留更多弱候选（漏检少但碎框多），调大只留高置信度")
        form.addRow("置信度门槛 box_thresh", self.sp_box)

        self.sp_rec = QDoubleSpinBox()
        self.sp_rec.setRange(0.1, 0.95)
        self.sp_rec.setSingleStep(0.05)
        self.sp_rec.setValue(0.95)
        self.sp_rec.setToolTip("OCR 识别过滤：候选框要能认出文字且置信度达标才保留。"
                               "调大=误检更少（杀误检主手段），调小=框更多")
        form.addRow("识别过滤分数", self.sp_rec)

        self.sp_minh = QSpinBox()
        self.sp_minh.setRange(0, 200)
        self.sp_minh.setValue(20)
        self.sp_minh.setSuffix(" px")
        self.sp_minh.setToolTip("最小文字高度：检测框高度低于它的一律丢弃，"
                                "用于过滤商品包装/衣服上的小字（0=不过滤）")
        form.addRow("最小文字高度", self.sp_minh)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> tuple:
        return (self.sp_step.value(), self.sp_iou.value(), self.sp_margin.value(),
                self.sp_thresh.value(), self.sp_box.value(), self.sp_rec.value(),
                self.sp_minh.value())
