"""强制 vLLM 在 Hygon DCU 上使用 CudaPlatform（不修改 vLLM 源码）。

背景
----
DCU 通过 HIP 的 CUDA 兼容层把自己暴露为 CUDA（torch.cuda.is_available() == True），
但 vLLM 0.11 的平台探测依赖：
  - pynvml（NVIDIA NVML）  -> DCU 上没有
  - amdsmi（AMD SMI）      -> DCU 上也没有
两者都失败时会回落到 UnspecifiedPlatform（device_type == ""），
随后 DeviceConfig.__post_init__ 抛出：
    RuntimeError: Failed to infer device type

做法
----
vLLM 解析 CLI 参数时会首次访问 vllm.platforms.current_platform，
从而触发探测。本模块在那之前把 cuda 的探测函数替换成直接返回
CudaPlatform，使 vLLM 按 CUDA 路径使用 DCU（与 DTK 版 torch 一致）。

用法：在 import 任何 vllm 业务模块之前先 import 本模块。
"""
import os

import vllm.platforms as _platforms

# DCU 是 HIP / ROCm 血统：选 rocm 时 vLLM 会跳过 NVIDIA cutlass 的
# FP8/w8a8 路径（is_cuda() 为 False），改用 HIP/lmslim 实现。
# 若实测 rocm 平台有问题，可用 VLLM_DCU_PLATFORM=cuda 切回 CUDA 兼容模式。
_PLATFORM = os.getenv("VLLM_DCU_PLATFORM", "rocm").strip().lower()

if _PLATFORM == "cuda":
    _TARGET = "vllm.platforms.cuda.CudaPlatform"
else:
    _TARGET = "vllm.platforms.rocm.RocmPlatform"


def _dcu_platform_plugin() -> str:
    """DCU 上直接指定平台：探测依赖的 pynvml/amdsmi 在 DCU 上都不存在。"""
    return _TARGET


# 覆盖内建 cuda 探测：DCU 无 pynvml，原探测必然返回 None
_platforms.builtin_platform_plugins["cuda"] = _dcu_platform_plugin
