"""海光 DCU 专用：给 vLLM 补上「能力探测类」CUTLASS 算子的安全兜底。

背景（2026-09-09 实战踩坑）
----------------------------
das 版 vLLM（0.11.0+das.opt1.dtk2604.torch290）的 `_C.abi3.so` 没有编译
NVIDIA CUTLASS 相关算子（DCU 无 FP8 单元，移植时直接省掉了）。实测：

    python -c "import torch, vllm._C; \
        print([n for n in dir(torch.ops._C) if 'cutlass' in n.lower()])"
    -> []                      # 一个都没有

而 vLLM 在 `quantization/utils/w8a8_utils.py` 的**模块级**就做能力探测：

    CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()      # line 94
    def cutlass_fp8_supported():                          # line 65
        if not current_platform.is_cuda():
            return False                                  # rocm 平台安全
        return ops.cutlass_scaled_mm_supports_fp8(capability)   # line 71 崩

只要平台被判定为 CUDA（is_cuda() == True），就会走到最后一行，抛：

    AttributeError: '_OpNamespace' '_C' object has no attribute
                    'cutlass_scaled_mm_supports_fp8'

这个崩溃发生在 import 期（Processor 初始化 → model_loader → fused_moe →
w8a8_utils），服务还没起来就挂了。

做法
----
给 `torch._ops._OpNamespace.__getattr__` 打兜底：凡是名字以 `cutlass` 开头
且含 `support` 的算子（即"能力探测"类，返回 bool），缺失时返回一个恒为
False 的桩函数。语义完全正确 —— DCU 确实不支持 FP8 CUTLASS。

**刻意不兜底**真正的计算算子（如 cutlass_scaled_mm、cutlass_group_gemm）：
它们名字里不含 support，缺失时仍会正常抛错，避免把静默错误藏起来。

用法
----
由 sitecustomize.py / dcu_platform_patch.py 自动调用，无需手动 import。
"""


def patch_missing_cutlass_ops() -> bool:
    """为缺失的 CUTLASS 能力探测算子注册 False 桩。返回是否实际打了补丁。"""
    try:
        import torch
        from torch._ops import _OpNamespace
    except Exception:  # noqa: BLE001  未装 torch / 未 source DTK env
        return False

    # 进程内幂等：子类或重复 import 也不重复包装
    if getattr(_OpNamespace, "_dcu_cutlass_patched", False):
        return False

    _orig_getattr = _OpNamespace.__getattr__

    def __getattr__(self, name):  # noqa: N807
        try:
            return _orig_getattr(self, name)
        except AttributeError:
            if name.startswith("cutlass") and "support" in name:
                # 能力探测：DCU 一律不支持 → False
                return lambda *args, **kwargs: False
            raise

    _OpNamespace.__getattr__ = __getattr__
    _OpNamespace._dcu_cutlass_patched = True
    return True
