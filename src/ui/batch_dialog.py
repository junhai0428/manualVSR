# -*- coding: utf-8 -*-
"""批量擦除：一次加载多个工程 json，串行跑，支持断点续跑与中途取消。"""
from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                               QTableWidgetItem, QAbstractItemView, QPushButton,
                               QCheckBox, QLabel, QMessageBox, QFileDialog,
                               QHeaderView, QProgressBar, QGroupBox)

from ..project import Project, default_output_path
from ..inpaint_engine import InpaintEngine
from ..workers import BatchItem, BatchWorker


class BatchDialog(QDialog):
    def __init__(self, engine: Optional[InpaintEngine], parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量擦除")
        self.resize(900, 520)
        self.engine = engine
        self.items: List[BatchItem] = []
        self.worker: Optional[BatchWorker] = None

        root = QVBoxLayout(self)
        tip = QLabel("选择一个或多个工程 json（视频同目录的 *.suberase.json）。"
                     "每个工程按自己的 settings 擦除，输出路径取工程里的 output 字段。")
        tip.setWordWrap(True)
        root.addWidget(tip)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["工程", "视频", "框数", "输出", "状态"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        opts = QGroupBox("选项")
        ol = QHBoxLayout(opts)
        self.ck_skip = QCheckBox("跳过输出已存在的（断点续跑）")
        self.ck_skip.setChecked(True)
        ol.addWidget(self.ck_skip)
        ol.addStretch(1)
        root.addWidget(opts)

        row = QHBoxLayout()
        self.btn_add = QPushButton("添加工程…")
        self.btn_add.clicked.connect(self.add_files_dialog)
        self.btn_remove = QPushButton("移除选中")
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_start = QPushButton("开始批量擦除")
        self.btn_start.clicked.connect(self.start)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.reject)
        for w in (self.btn_add, self.btn_remove, self.btn_clear, self.btn_start,
                  self.btn_cancel, self.btn_close):
            row.addWidget(w)
        root.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)
        self.lb_status = QLabel("就绪")
        root.addWidget(self.lb_status)

    # ---------- 列表 ----------
    def add_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择工程文件", "",
                                                "工程 (*.suberase.json *.json);;所有文件 (*.*)")
        if paths:
            self.add_files(paths)

    def add_files(self, paths: List[str]) -> int:
        added = 0
        for p in paths:
            try:
                pj = Project.load(p)
            except Exception as e:
                self.lb_status.setText(f"载入失败 {os.path.basename(p)}: {e}")
                continue
            video = pj.video
            if not video or not os.path.exists(video):
                self.lb_status.setText(f"缺少视频 {pj.video or '?'}：{os.path.basename(p)}")
                continue
            if not pj.need_process():
                self.lb_status.setText(f"没有启用的框：{os.path.basename(p)}")
                continue
            out = pj.output or default_output_path(video)
            self.items.append(BatchItem(project_path=p, project=pj,
                                        video=video, out=out))
            added += 1
        self.refresh()
        return added

    def remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.items):
                self.items.pop(r)
        self.refresh()

    def clear_all(self) -> None:
        self.items.clear()
        self.refresh()

    def refresh(self) -> None:
        self.table.setRowCount(len(self.items))
        for i, it in enumerate(self.items):
            vals = [os.path.basename(it.project_path), os.path.basename(it.video),
                    str(len(it.project.boxes)), os.path.basename(it.out), it.status]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setToolTip(vals[1] if c == 1 else v)
                self.table.setItem(i, c, item)

    # ---------- 执行 ----------
    def start(self) -> None:
        if self.engine is None:
            QMessageBox.information(self, "提示", "模型还没加载完，请稍候")
            return
        if not self.items:
            QMessageBox.information(self, "提示", "还没有添加工程")
            return
        self.set_running(True)
        self.progress.setValue(0)
        self.worker = BatchWorker(self.items, self.engine, self.ck_skip.isChecked(), self)
        self.worker.task_started.connect(self._on_task_started)
        self.worker.task_progress.connect(self._on_task_progress)
        self.worker.task_done.connect(self._on_task_done)
        self.worker.task_failed.connect(self._on_task_failed)
        self.worker.finished_all.connect(self._on_all_done)
        self.worker.start()

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.lb_status.setText("取消中：跑完当前任务后停止")

    def set_running(self, running: bool) -> None:
        for w in (self.btn_add, self.btn_remove, self.btn_clear, self.btn_start, self.btn_close):
            w.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        self.ck_skip.setEnabled(not running)

    def _on_task_started(self, idx: int, name: str) -> None:
        if 0 <= idx < len(self.items):
            self.items[idx].status = "进行中"
            self.refresh()
        self.lb_status.setText(f"[{idx + 1}/{len(self.items)}] {name}")

    def _on_task_progress(self, idx: int, done: int, total: int, msg: str) -> None:
        if total > 0:
            span = 100.0 / max(1, len(self.items))
            self.progress.setValue(int((idx + done / total) * span))
        self.lb_status.setText(f"[{idx + 1}/{len(self.items)}] {msg} ({done}/{total})")

    def _on_task_done(self, idx: int, stats: dict) -> None:
        if 0 <= idx < len(self.items):
            it = self.items[idx]
            if not stats.get("skipped"):
                it.frames = stats.get("frames", 0)
                it.seconds = stats.get("seconds", 0.0)
            self.refresh()

    def _on_task_failed(self, idx: int, err: str) -> None:
        if 0 <= idx < len(self.items):
            self.items[idx].status = f"失败: {err[:60]}"
            self.refresh()

    def _on_all_done(self, summary: dict) -> None:
        self.set_running(False)
        self.progress.setValue(100)
        self.refresh()
        msg = (f"成功 {summary['ok']} · 跳过 {summary['skipped']} · 失败 {summary['fail']}\n"
               f"共处理 {summary['frames']} 帧（擦除 {summary['inpainted']} 帧）\n"
               f"纯推理+编码耗时 {summary['seconds']}s")
        self.lb_status.setText(msg.replace("\n", " | "))
        QMessageBox.information(self, "批量完成", msg)

    def closeEvent(self, e) -> None:
        if self.worker and self.worker.isRunning():
            r = QMessageBox.question(self, "正在运行", "任务还在跑，确定关闭？（会先停止）",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.worker.cancel()
            self.worker.wait(20000)
        super().closeEvent(e)
