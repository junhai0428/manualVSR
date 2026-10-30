# -*- coding: utf-8 -*-
"""内嵌 IOPaint(lama) 推理引擎：真 batch（[B,C,H,W] 一次前向）+ 显存自适应。

不启 HTTP 服务，直接在进程内调用 TorchScript big-lama，省掉 base64/HTTP 开销。
输入统一为 crop 后的小图（字幕带），显存占用低、吞吐高。
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.prepare_model import set_cache_env, ensure_model  # noqa: E402

set_cache_env()  # 必须在 import torch 之前

import torch  # noqa: E402
import cv2  # noqa: E402

PAD_MOD = 8
VRAM_SAFE_BYTES = 3 * 1024 ** 3      # 空闲显存低于此值不上 GPU
BATCH_PIXEL_BUDGET = 2_000_000       # 单 batch 总像素预算（fp32）


class EngineError(RuntimeError):
    pass


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            free, _total = torch.cuda.mem_get_info()
            if free > VRAM_SAFE_BYTES:
                return torch.device("cuda")
        except Exception:
            pass
        if prefer == "cuda":
            return torch.device("cuda")
    return torch.device("cpu")


class InpaintEngine:
    def __init__(self, model: str = "lama", device: str = "auto"):
        self.model_name = model
        self.torch_model = None
        self._device: Optional[torch.device] = None
        self.stats = {
            "device": "-", "load_s": 0.0, "last_frames": 0,
            "last_ms": 0.0, "ms_per_frame": 0.0, "vram_free_gb": 0.0,
            "backend": "-",
        }
        self.load(device)

    # ---------- 加载 ----------
    def load(self, device: str = "auto") -> None:
        t0 = time.time()
        ensure_model(verbose=False)
        self._device = pick_device(device)
        try:
            self._load_via_iopaint()
            self.stats["backend"] = "iopaint.LaMa"
        except Exception as e:  # 回落到直连权重
            try:
                self._load_direct()
                self.stats["backend"] = f"torch.jit(直连, 原因: {type(e).__name__})"
            except Exception as e2:
                raise EngineError(f"模型加载失败: {e} / {e2}") from e2
        self.torch_model.eval()
        self.stats["device"] = str(self._device)
        self.stats["load_s"] = round(time.time() - t0, 2)
        self.warmup()

    def _load_via_iopaint(self) -> None:
        from iopaint.model.lama import LaMa

        wrapper = LaMa(self._device)
        self.torch_model = wrapper.model
        self.pad_mod = getattr(wrapper, "pad_mod", PAD_MOD)

    def _load_direct(self) -> None:
        from tools.prepare_model import target_path

        path = target_path()
        m = torch.jit.load(path, map_location="cpu")
        self.torch_model = m.to(self._device)
        self.pad_mod = PAD_MOD

    def warmup(self) -> None:
        img = (np.random.rand(1, 64, 64, 3) * 255).astype(np.uint8)
        msk = np.zeros((1, 64, 64), np.uint8)
        msk[:, 20:40, 20:40] = 255
        try:
            self.inpaint_batch(img, msk)
        except Exception:
            pass

    # ---------- 推理 ----------
    @staticmethod
    def _pad(batch: np.ndarray, mod: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        b, h, w = batch.shape[:3]
        nh = int(np.ceil(h / mod) * mod)
        nw = int(np.ceil(w / mod) * mod)
        if (nh, nw) == (h, w):
            return batch, (h, w)
        pad = ((0, 0), (0, nh - h), (0, nw - w)) + ((0, 0),) * (batch.ndim - 3)
        return np.pad(batch, pad, mode="edge"), (h, w)

    def auto_batch(self, h: int, w: int, user_batch: int = 0) -> int:
        if user_batch and user_batch > 0:
            return max(1, user_batch)
        return max(1, min(16, int(BATCH_PIXEL_BUDGET / max(1, h * w))))

    @torch.inference_mode()
    def inpaint_batch(self, images_rgb: np.ndarray, masks: np.ndarray,
                      batch_size: int = 0) -> np.ndarray:
        """images_rgb: [B,H,W,3] uint8 RGB；masks: [B,H,W] uint8(0/255)。
        返回 [B,H,W,3] uint8 BGR。"""
        assert images_rgb.ndim == 4 and images_rgb.shape[0] == masks.shape[0]
        n, h, w = images_rgb.shape[:3]
        bs = self.auto_batch(h, w, batch_size)
        out = np.empty_like(images_rgb)

        i = 0
        while i < n:
            j = min(i + bs, n)
            chunk_img = images_rgb[i:j]
            chunk_msk = masks[i:j]
            cur_bs = j - i
            try:
                out[i:j] = self._forward(chunk_img, chunk_msk, (h, w))
                i = j
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg and cur_bs > 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    bs = max(1, cur_bs // 2)
                    continue
                raise
        return out

    def _forward(self, img: np.ndarray, msk: np.ndarray, origin: Tuple[int, int]) -> np.ndarray:
        t0 = time.time()
        pad_img, (oh, ow) = self._pad(img, self.pad_mod)
        pad_msk, _ = self._pad(msk[:, :, :, None], self.pad_mod)

        x = np.transpose(pad_img, (0, 3, 1, 2)).astype(np.float32) / 255.0
        m = np.transpose(pad_msk, (0, 3, 1, 2)).astype(np.float32) / 255.0
        m = (m > 0).astype(np.float32)

        xt = torch.from_numpy(np.ascontiguousarray(x)).to(self._device)
        mt = torch.from_numpy(np.ascontiguousarray(m)).to(self._device)
        res = self.torch_model(xt, mt)

        cur = res.permute(0, 2, 3, 1).detach().cpu().numpy()
        cur = np.clip(cur * 255, 0, 255).astype(np.uint8)[:, :oh, :ow, :]
        cur = cur[:, :, :, ::-1]  # RGB -> BGR

        dt = (time.time() - t0) * 1000
        self.stats["last_frames"] = int(img.shape[0])
        self.stats["last_ms"] = round(dt, 1)
        self.stats["ms_per_frame"] = round(dt / max(1, img.shape[0]), 1)
        if torch.cuda.is_available():
            try:
                free, _ = torch.cuda.mem_get_info()
                self.stats["vram_free_gb"] = round(free / 1024 ** 3, 2)
            except Exception:
                pass
        return cur

    # ---------- 工具 ----------
    @staticmethod
    def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return mask
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
        return cv2.dilate(mask, k)

    def summary(self) -> str:
        s = self.stats
        return (f"模型 lama | 后端 {s['backend']} | 设备 {s['device']} | "
                f"加载 {s['load_s']}s | 上次 {s['last_frames']}帧/{s['last_ms']}ms "
                f"({s['ms_per_frame']}ms/帧) | 空闲显存 {s['vram_free_gb']}GB")
