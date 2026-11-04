# -*- coding: utf-8 -*-
"""主窗口：加载视频 → 框选/定时 → 一键擦除。"""
from __future__ import annotations

import os
import time
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                               QFileDialog, QPushButton, QLabel, QSpinBox, QMessageBox,
                               QProgressBar, QStatusBar, QToolBar, QFrame, QDialog)

from ..project import Project, Box, Settings, default_project_path, default_output_path
from ..video_io import probe, FrameReader, VideoMeta, VideoIOError
from ..workers import EngineLoader, EraseWorker, DetectWorker
from ..inpaint_engine import InpaintEngine
from .frame_view import FrameView
from .timeline import TimelineWidget
from .box_panel import BoxPanel
from .batch_dialog import BatchDialog
from .detect_dialog import DetectParamsDialog


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动字幕擦除 · SubtitleEraser")
        self.resize(1360, 820)

        self.project = Project()
        self.meta: Optional[VideoMeta] = None
        self.reader: Optional[FrameReader] = None
        self.engine: Optional[InpaintEngine] = None
        self.worker: Optional[EraseWorker] = None
        self.detect_worker: Optional[DetectWorker] = None
        self.frame_idx = 0
        self.dirty = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._play_t0 = 0.0
        self._play_f0 = 0

        self._build_ui()
        self._build_shortcuts()
        self._start_engine()

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_open = QAction("打开视频", self); self.act_open.triggered.connect(self.open_video_dialog)
        self.act_save = QAction("保存工程", self); self.act_save.triggered.connect(self.save_project)
        self.act_load = QAction("载入工程", self); self.act_load.triggered.connect(self.load_project_dialog)
        self.act_erase = QAction("开始擦除", self); self.act_erase.triggered.connect(self.start_erase)
        self.act_cancel = QAction("取消", self); self.act_cancel.setEnabled(False)
        self.act_cancel.triggered.connect(self.cancel_erase)
        self.act_batch = QAction("批量擦除 (Ctrl+B)", self)
        self.act_batch.triggered.connect(self.open_batch)
        self.act_detect = QAction("自动检测框", self)
        self.act_detect.setToolTip("OCR 自动找出文字区域，生成擦除框（可再手动调整）")
        self.act_detect.triggered.connect(self.auto_detect)
        self.act_ghost = QAction("显示非活跃框", self)
        self.act_ghost.setCheckable(True)
        self.act_ghost.setChecked(False)
        self.act_ghost.toggled.connect(self._on_ghost_toggled)
        self.act_ghost.setToolTip("勾选后，当前帧不生效的框也会以淡虚线显示")
        for a in (self.act_open, self.act_save, self.act_load):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_erase)
        tb.addAction(self.act_cancel)
        tb.addSeparator()
        tb.addAction(self.act_detect)
        tb.addAction(self.act_batch)
        tb.addAction(self.act_ghost)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        self.view = FrameView()
        lv.addWidget(self.view, 1)

        # 帧控制条
        bar = QHBoxLayout()
        self.btn_first = QPushButton("⏮"); self.btn_first.clicked.connect(lambda: self.seek(0))
        self.btn_m10 = QPushButton("◀◀"); self.btn_m10.clicked.connect(lambda: self.seek(self.frame_idx - 10))
        self.btn_m1 = QPushButton("◀"); self.btn_m1.clicked.connect(lambda: self.seek(self.frame_idx - 1))
        self.btn_p1 = QPushButton("▶"); self.btn_p1.clicked.connect(lambda: self.seek(self.frame_idx + 1))
        self.btn_p10 = QPushButton("▶▶"); self.btn_p10.clicked.connect(lambda: self.seek(self.frame_idx + 10))
        self.btn_last = QPushButton("⏭"); self.btn_last.clicked.connect(
            lambda: self.seek(max(0, (self.meta.nb_frames if self.meta else 1) - 1)))
        self.btn_play = QPushButton("播放 (Space)"); self.btn_play.clicked.connect(self.toggle_play)
        self.sp_frame = QSpinBox(); self.sp_frame.setRange(0, 0)
        self.sp_frame.valueChanged.connect(lambda v: self.seek(v))
        self.lb_time = QLabel("00:00.00 / 00:00.00")
        self.lb_fps = QLabel("")
        for w in (self.btn_first, self.btn_m10, self.btn_m1, self.sp_frame,
                  self.btn_p1, self.btn_p10, self.btn_last, self.btn_play,
                  self.lb_time, self.lb_fps):
            bar.addWidget(w)
        bar.addStretch(1)
        lv.addLayout(bar)

        self.timeline = TimelineWidget()
        self.timeline.seeked.connect(self.seek)
        self.timeline.box_clicked.connect(self._on_select)
        lv.addWidget(self.timeline)

        self.panel = BoxPanel()
        self.panel.selection_changed.connect(self._on_select)
        self.panel.boxes_changed.connect(self._on_boxes_changed)
        self.panel.delete_requested.connect(self._on_delete_box)
        self.panel.goto_requested.connect(self.seek)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([980, 340])
        self.setCentralWidget(split)

        self.view.box_created.connect(self._on_box_created)
        self.view.box_changed.connect(self._on_boxes_changed)
        self.view.selection_changed.connect(self._on_select)

        st = QStatusBar()
        self.setStatusBar(st)
        self.lb_status = QLabel("就绪")
        self.lb_engine = QLabel("模型加载中…")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        self.progress.setFixedWidth(240)
        st.addWidget(self.lb_status, 1)
        st.addPermanentWidget(self.lb_engine)
        st.addPermanentWidget(self.progress)

    def _build_shortcuts(self) -> None:
        K = Qt.Key
        pairs = [
            (K.Key_Left, lambda: self.seek(self.frame_idx - 1)),
            (K.Key_Right, lambda: self.seek(self.frame_idx + 1)),
            (QKeySequence("Shift+Left"), lambda: self.seek(self.frame_idx - 10)),
            (QKeySequence("Shift+Right"), lambda: self.seek(self.frame_idx + 10)),
            (QKeySequence("Ctrl+Left"), lambda: self.seek(self.frame_idx - int(self.meta.fps or 25))),
            (QKeySequence("Ctrl+Right"), lambda: self.seek(self.frame_idx + int(self.meta.fps or 25))),
            (K.Key_Home, lambda: self.seek(0)),
            (K.Key_End, lambda: self.seek(max(0, (self.meta.nb_frames if self.meta else 1) - 1))),
            (K.Key_Space, self.toggle_play),
            (K.Key_I, self._set_in),
            (K.Key_O, self._set_out),
            (K.Key_BracketLeft, self._set_in_head),      # [ = 入点设为开头(第0帧)
            (K.Key_BracketRight, self._set_out_tail),    # ] = 出点设为末尾(末帧)
            (K.Key_Delete, self._delete_selected),
            (K.Key_Escape, lambda: self._on_select(None)),
            (QKeySequence("Ctrl+S"), self.save_project),
            (QKeySequence("Ctrl+O"), self.open_video_dialog),
            (QKeySequence("Ctrl+B"), self.open_batch),
        ]
        self._shortcuts = []
        for key, fn in pairs:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

    # ---------------- 引擎 ----------------
    def _start_engine(self) -> None:
        self.loader = EngineLoader("lama", "auto", self)
        self.loader.ready.connect(self._on_engine_ready)
        self.loader.failed.connect(lambda e: self.lb_engine.setText(f"模型加载失败: {e}"))
        self.loader.start()

    def _on_engine_ready(self, eng: InpaintEngine) -> None:
        self.engine = eng
        self.lb_engine.setText(eng.summary())

    # ---------------- 视频 ----------------
    def open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频 (*.mp4 *.mov *.mkv *.avi *.flv *.wmv *.webm);;所有文件 (*.*)")
        if path:
            self.open_video(path)

    def open_video(self, path: str) -> None:
        try:
            meta = probe(path)
        except VideoIOError as e:
            QMessageBox.critical(self, "打开失败", str(e))
            return
        self._close_reader()
        self.meta = meta
        self.project = Project(video=path, meta={
            "fps": meta.fps, "width": meta.width, "height": meta.height,
            "duration": meta.duration, "nb_frames": meta.nb_frames})
        self.project.output = default_output_path(path)
        self.project.settings = Settings()
        pjp = default_project_path(path)
        if os.path.exists(pjp):
            try:
                p = Project.load(pjp)
                if p.video == path:
                    self.project.boxes = p.boxes
                    self.project.settings = p.settings
                    self.lb_status.setText(f"已自动载入工程 {os.path.basename(pjp)}")
            except Exception:
                pass
        self.project.clamp_frames(meta.nb_frames)   # 旧工程展开的哨兵 -> 真实末帧号
        for b in self.project.boxes:
            b.clamp(meta.width, meta.height, meta.nb_frames)
        self.reader = FrameReader(path, meta)
        self.view.total_frames = meta.nb_frames
        self.sp_frame.setRange(0, max(0, meta.nb_frames - 1))
        self.sp_frame.setMaximum(max(0, meta.nb_frames - 1))
        self.timeline.set_meta(meta.nb_frames)
        self.panel.set_project(self.project, meta.nb_frames)
        self.view.set_boxes(self.project.boxes)
        self.timeline.set_boxes(self.project.boxes)
        self.setWindowTitle(f"手动字幕擦除 · {os.path.basename(path)}")
        self.lb_fps.setText(f"{meta.width}x{meta.height} {meta.fps:.3f}fps "
                            f"{meta.nb_frames}帧{'（估算）' if not meta.nb_frames_exact else ''}")
        if meta.note:
            self.lb_status.setText(meta.note)
        self.seek(0)

    def _close_reader(self) -> None:
        if self.reader:
            self.reader.release()
            self.reader = None

    def seek(self, idx: int) -> None:
        if not self.meta or not self.reader:
            return
        idx = max(0, min(idx, self.meta.nb_frames - 1))
        self.frame_idx = idx
        frame = self.reader.read(idx)
        self.view.set_frame(frame)
        self.view.set_frame_idx(idx)
        self.timeline.set_frame(idx)
        self.panel.set_frame_idx(idx)
        self.sp_frame.blockSignals(True)
        self.sp_frame.setValue(idx)
        self.sp_frame.blockSignals(False)
        self.lb_time.setText(f"{self.meta.fmt_time(idx)} / {self.meta.fmt_time(self.meta.nb_frames - 1)}")

    def toggle_play(self) -> None:
        if not self.meta:
            return
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.btn_play.setText("播放 (Space)")
        else:
            self._play_t0 = time.time()
            self._play_f0 = self.frame_idx
            self._play_timer.start(max(8, int(1000 / max(1, self.meta.fps))))
            self.btn_play.setText("暂停 (Space)")

    def _on_play_tick(self) -> None:
        if not self.meta:
            return
        el = time.time() - self._play_t0
        target = self._play_f0 + int(el * self.meta.fps)
        if target >= self.meta.nb_frames - 1:
            self.seek(self.meta.nb_frames - 1)
            self.toggle_play()
            return
        self.seek(target)

    # ---------------- 框 ----------------
    def _on_box_created(self, b: Box) -> None:
        b.clamp(self.meta.width, self.meta.height, self.meta.nb_frames)
        self.project.add_box(b)
        self.dirty = True
        self._refresh_boxes()
        self.panel.select_box(b.id)
        self.lb_status.setText(
            f"新增框：入点={self.frame_idx}，出点=末帧 {b.end}（虚线）；"
            f"往后找出点后按 O 改成当前帧")

    def _on_boxes_changed(self) -> None:
        self.dirty = True
        self._refresh_boxes()

    def _on_delete_box(self, box_id: str) -> None:
        self.project.remove_box(box_id)
        self.dirty = True
        self._refresh_boxes()

    def _delete_selected(self) -> None:
        bid = self.panel.selected_box_id()
        if bid:
            self._on_delete_box(bid)

    def _on_select(self, box_id) -> None:
        if box_id is not None and hasattr(box_id, "id"):   # 兼容误传 Box 对象
            box_id = box_id.id
        self.view.set_selected(box_id)
        self.panel.select_box(box_id)

    def _refresh_boxes(self) -> None:
        self.view.set_boxes(self.project.boxes)
        self.timeline.set_boxes(self.project.boxes)
        self.panel.refresh()
        self.view.update()
        self.timeline.update()

    def _set_in(self) -> None:
        self.panel._set_in()

    def _set_out(self) -> None:
        self.panel._set_out()

    def _set_in_head(self) -> None:
        self.panel._set_in_head()

    def _set_out_tail(self) -> None:
        self.panel._set_out_tail()

    # ---------------- 工程 ----------------
    def save_project(self) -> None:
        if not self.project.video:
            return
        try:
            p = self.project.save()
            self.dirty = False
            self.lb_status.setText(f"工程已保存: {p}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def load_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "载入工程", "", "工程 (*.json);;所有文件 (*.*)")
        if not path:
            return
        try:
            p = Project.load(path)
        except Exception as e:
            QMessageBox.warning(self, "载入失败", str(e))
            return
        if p.video and os.path.exists(p.video):
            self.open_video(p.video)
            self.project.boxes = p.boxes
            self.project.settings = p.settings
            self.panel.set_project(self.project, self.meta.nb_frames)
            self._refresh_boxes()
        else:
            QMessageBox.warning(self, "载入失败", f"工程中的视频不存在: {p.video}")

    # ---------------- 擦除 ----------------
    def start_erase(self) -> None:
        if not self.meta:
            QMessageBox.information(self, "提示", "先打开一个视频")
            return
        if not self.project.need_process():
            QMessageBox.information(self, "提示", "还没有任何启用的框")
            return
        if self.engine is None:
            QMessageBox.information(self, "提示", "模型还在加载，请稍候")
            return
        out = self.project.output or default_output_path(self.project.video)
        out, _ = QFileDialog.getSaveFileName(self, "输出视频", out, "MP4 (*.mp4);;所有文件 (*.*)")
        if not out:
            return
        self.project.output = out
        self.save_project()

        self.act_erase.setEnabled(False)
        self.act_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.worker = EraseWorker(self.project, self.meta, out, self.engine, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self.lb_status.setText("擦除中…")

    def cancel_erase(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.lb_status.setText("正在取消…")

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setValue(min(100, int(done * 100 / total)))
        self.lb_status.setText(f"{msg}  ({done}/{total})")

    def _on_finished(self, stats: dict) -> None:
        self.progress.setVisible(False)
        self.act_erase.setEnabled(True)
        self.act_cancel.setEnabled(False)
        sp = stats["frames"] / max(1e-6, stats["seconds"])
        QMessageBox.information(
            self, "完成",
            f"输出: {self.project.output}\n"
            f"帧数 {stats['frames']} · 擦除 {stats['inpainted']} 帧\n"
            f"用时 {stats['seconds']}s ({sp:.1f} 帧/秒) · 编码器 {stats['encoder']}")
        self.lb_status.setText(f"完成：{stats['frames']}帧/{stats['seconds']}s")

    def _on_failed(self, err: str) -> None:
        self.progress.setVisible(False)
        self.act_erase.setEnabled(True)
        self.act_cancel.setEnabled(False)
        QMessageBox.critical(self, "擦除失败", err)
        self.lb_status.setText(f"失败: {err}")

    def _on_ghost_toggled(self, on: bool) -> None:
        self.view.set_show_inactive(on)

    def open_batch(self) -> None:
        dlg = BatchDialog(self.engine, self)
        dlg.exec()

    # ---------------- 自动检测 ----------------
    def auto_detect(self) -> None:
        if not self.meta:
            QMessageBox.information(self, "提示", "先打开一个视频")
            return
        if self.detect_worker and self.detect_worker.isRunning():
            return
        dlg = DetectParamsDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        step, iou, margin, thresh, box_thr, rec_score, min_h = dlg.values()
        self.act_detect.setEnabled(False)
        self.act_erase.setEnabled(False)
        self.progress.setRange(0, self.meta.nb_frames - 1)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lb_status.setText("自动检测中…（OCR GPU）")
        self.detect_worker = DetectWorker(self.project.video, step, iou, margin,
                                          thresh, box_thr, rec_score, min_h, self)
        self.detect_worker.progress.connect(self._on_detect_progress)
        self.detect_worker.finished_ok.connect(self._on_detect_done)
        self.detect_worker.failed.connect(self._on_detect_failed)
        self.detect_worker.start()

    def _on_detect_progress(self, frame_no: int) -> None:
        self.progress.setValue(min(frame_no, self.progress.maximum()))

    def _on_detect_done(self, boxes: list, summary: str) -> None:
        self._on_detect_ui_reset()
        if not boxes:
            self.lb_status.setText("自动检测完成：未发现文字区域")
            return
        for b in boxes:
            b.clamp(self.meta.width, self.meta.height, self.meta.nb_frames)
        # 重跑检测：清掉原有全部框（不分手动/自动），换成新结果
        cleared = self.project.replace_boxes(boxes)
        self.dirty = True
        self._refresh_boxes()
        tip = f"（已清除原有 {cleared} 个框）" if cleared else ""
        self.lb_status.setText(f"自动检测完成：{len(boxes)} 个框{tip} · {summary}")

    def _on_detect_failed(self, err: str) -> None:
        self._on_detect_ui_reset()
        QMessageBox.critical(self, "自动检测失败", err)
        self.lb_status.setText(f"检测失败: {err}")

    def _on_detect_ui_reset(self) -> None:
        self.act_detect.setEnabled(True)
        self.act_erase.setEnabled(True)
        self.progress.setVisible(False)

    def closeEvent(self, e) -> None:
        self._play_timer.stop()
        self._close_reader()
        if self.dirty and self.project.video:
            r = QMessageBox.question(self, "未保存", "工程有改动，是否保存？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes:
                self.save_project()
        super().closeEvent(e)
