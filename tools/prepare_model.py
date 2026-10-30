# -*- coding: utf-8 -*-
"""把 lama 权重统一到 E 盘缓存目录，避免 C 盘爆满 + 路径错位导致模型扫描不到。

必须在 import torch 之前设置 XDG_CACHE_HOME（torch.hub.get_dir() 会读它）。
"""
import os
import shutil
import sys

CACHE_ROOT = r"<MODEL_DIR>"
CANDIDATE_SOURCES = [
    os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "big-lama.pt"),
    r"<USERPROFILE>\.cache\torch\hub\checkpoints\big-lama.pt",
]
MODEL_NAME = "big-lama.pt"
MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"


def target_path() -> str:
    return os.path.join(CACHE_ROOT, "torch", "hub", "checkpoints", MODEL_NAME)


def ensure_model(verbose: bool = True) -> str:
    dst = target_path()
    if os.path.exists(dst):
        if verbose:
            print(f"[model] 已就位: {dst} ({os.path.getsize(dst) / 1048576:.1f} MB)")
        return dst

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for src in CANDIDATE_SOURCES:
        if os.path.exists(src):
            if verbose:
                print(f"[model] 复制 {src} -> {dst}")
            shutil.copy2(src, dst)
            if verbose:
                print(f"[model] 完成 ({os.path.getsize(dst) / 1048576:.1f} MB)")
            return dst

    raise FileNotFoundError(
        "找不到 big-lama.pt，请先运行一次 iopaint 让它自动下载，或手动放到 " + dst
    )


def set_cache_env() -> None:
    os.environ.setdefault("XDG_CACHE_HOME", CACHE_ROOT)
    os.environ.setdefault("TORCH_HOME", os.path.join(CACHE_ROOT, "torch"))
    os.environ.setdefault("U2NET_HOME", CACHE_ROOT)
    # Windows 下 torch 满线程会段错误
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "4")


if __name__ == "__main__":
    set_cache_env()
    p = ensure_model()
    import hashlib

    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print("md5:", h.hexdigest(), "expected:", MD5, "OK" if h.hexdigest() == MD5 else "MISMATCH")
    sys.exit(0 if h.hexdigest() == MD5 else 1)
