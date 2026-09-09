"""在 Hygon DCU 上锁定 vLLM 平台并补齐缺失算子（不修改 vLLM 源码）。

背景
----
DCU 通过 HIP 的 CUDA 兼容层把自己暴露为 CUDA（torch.cuda.is_available() == True），
但 vLLM 0.11 的平台探测依赖 pynvml / amdsmi，DCU 上两者都没有，探测失败会
回落到 UnspecifiedPlatform，随后 DeviceConfig 抛
    RuntimeError: Failed to infer device type

同时 das 版 vLLM 未编译 NVIDIA CUTLASS 算子，平台一旦被判成 CUDA，
w8a8_utils.py 在 import 期调用 cutlass_*_supported() 就会抛
    AttributeError: '_OpNamespace' '_C' object has no attribute
                    'cutlass_scaled_mm_supports_fp8'

做法
----
1. 把内建 cuda 插件指向 RocmPlatform（默认），并删除内建 rocm 键
   （vLLM 只允许一个内建插件激活）。
2. 调用 dcu_cutlass_patch 为缺失的「能力探测类」cutlass 算子兜底为 False。

用法：在 import 任何 vLLM 业务模块之前先 import 本模块。
平台切换：VLLM_DCU_PLATFORM=cuda 可退回 CUDA 兼容模式（一般不需要）。
"""
import os
import sys

import vllm.platforms as _platforms

# DCU 是 HIP / ROCm 血统：rocm 平台下 is_cuda() 为 False，vLLM 会跳过
# NVIDIA cutlass 的 FP8/w8a8 路径，改用 HIP/lmslim 实现。
_PLATFORM = os.getenv("VLLM_DCU_PLATFORM", "rocm").strip().lower()

if _PLATFORM == "cuda":
    _TARGET = "vllm.platforms.cuda.CudaPlatform"
else:
    _TARGET = "vllm.platforms.rocm.RocmPlatform"


def _dcu_platform_plugin() -> str:
    """DCU 上直接指定平台：探测依赖的 pynvml/amdsmi 在 DCU 上都不存在。"""
    return _TARGET


# 覆盖内建 cuda 探测：DCU 无 pynvml，原探测要么失败要么误判
_platforms.builtin_platform_plugins["cuda"] = _dcu_platform_plugin
# 删除内建 rocm 键：避免 cuda + rocm 同时激活触发
# "Only one platform plugin can be activated"
_platforms.builtin_platform_plugins.pop("rocm", None)

print(f"[dcu_platform_patch] platform -> {_TARGET}", file=sys.stderr)

# 补齐 das 版 vLLM 缺失的 CUTLASS 能力探测算子（幂等）
try:
    import dcu_cutlass_patch

    if dcu_cutlass_patch.patch_missing_cutlass_ops():
        print("[dcu_platform_patch] cutlass support-op stub installed",
              file=sys.stderr)
except Exception as _e:  # noqa: BLE001
    print(f"[dcu_platform_patch] cutlass patch skipped: {_e!r}", file=sys.stderr)
