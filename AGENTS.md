# 项目开发规范

- [1] 版本管理：每次改动之后，必须创建一个对应的 Git Commit，以便后续追踪和回滚。
- [2] 测试要求：每次改动之后，必须编写或更新相关的测试，并在交付给用户前，
      确保所有测试和验收全部通过。

## 运行环境（本机固定，勿改动）

| 项 | 值 |
|---|---|
| Python | `<MAIN_ENV>\Scripts\python.exe`（torch 2.6.0+cu124、iopaint 1.6.0、PySide6 6.12、cv2 4.11） |
| 推理设备 | CUDA 优先，`torch.cuda.mem_get_info()` free > 3GB 才上 GPU，否则回落 CPU 并告警 |
| 模型缓存 | `<MAIN_ENV>\models`（`XDG_CACHE_HOME` 必须指向它，否则 torch.hub 与 iopaint 扫描路径错位） |
| ffmpeg | 全局 PATH，编码走 NVENC |
| 网络 | 请求本地服务需绕过代理（本项目已内嵌推理，无 HTTP 依赖） |

## 硬性规避（踩过的坑）

1. `import torch` 之前必须设置 `OMP/MKL/OPENBLAS/VECLIB/NUMEXPR_NUM_THREADS=4`，
   否则 Windows 下满线程段错误（exit 139）。
2. 不要同时开 2 个 CUDA 推理进程。
3. 所有 ffmpeg/ffprobe 子进程必须带 `timeout`。
4. 中间帧一律内存托管，不落盘（产物才落盘）。
5. 长任务必须支持断点续跑 / 进度上报。

## 目录

```
src/      GUI 与核心逻辑
tests/    自测脚本
tools/    辅助工具（生成测试视频等）
```
