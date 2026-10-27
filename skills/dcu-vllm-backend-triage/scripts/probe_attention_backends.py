#!/usr/bin/env python3
"""DCU 上 vLLM attention 后端体检：判断该用哪个 --attention-backend。

检查项：
  1) vllm._rocm_C 能否 import（ROCM_ATTN 的前提）
  2) vllm_flash_attn 能否 import（FLASH_ATTN 的前提；注意它 != 系统里的 flash_attn）
  3) aiter 是否可用（部分 DTK 镜像的替代算子库）
  4) 对指定模型，vLLM 列出的候选后端
  5) 给出建议

用法：
  python3 probe_attention_backends.py [--model-path /path/to/model]
不传 model-path 时只做扩展层体检。
"""
import argparse
import importlib
import sys


def check_mod(name):
    try:
        m = importlib.import_module(name)
        return True, getattr(m, "__version__", "?"), getattr(m, "__file__", "?")
    except Exception as e:
        return False, f"{type(e).__name__}", str(e)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()

    print("=== 1. 编译扩展 ===")
    rocm_c_ok = False
    for name in ("vllm._rocm_C", "vllm_flash_attn", "aiter", "flash_attn", "triton"):
        ok, ver, where = check_mod(name)
        if name == "vllm._rocm_C":
            rocm_c_ok = ok
        mark = "OK  " if ok else "MISS"
        print(f"  [{mark}] {name:20} {ver:28} {where}")

    print("\n=== 2. 后端可用性推断 ===")
    verdict = []
    if rocm_c_ok:
        verdict.append(("ROCM_ATTN", True, "_rocm_C 可导入"))
    else:
        verdict.append(("ROCM_ATTN", False, "缺 vllm._rocm_C → paged_attention 会 AttributeError"))

    fa_ok, fa_ver, _ = check_mod("vllm_flash_attn")
    verdict.append(("FLASH_ATTN", fa_ok,
                    "vllm_flash_attn 可导入" if fa_ok
                    else "缺 vllm_flash_attn（系统 flash_attn 不能替代）→ version not detected"))

    triton_ok, _, _ = check_mod("triton")
    verdict.append(("TRITON_ATTN", triton_ok,
                    "纯 Triton 实现，不依赖编译扩展" if triton_ok else "triton 不可用"))

    for n, ok, why in verdict:
        print(f"  {'✅' if ok else '❌'} {n:14} {why}")

    print("\n=== 3. 对指定模型的候选后端 ===")
    if a.model_path:
        try:
            import os

            os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
            from vllm.engine.arg_utils import EngineArgs
            from vllm.v1.attention.selector import get_attn_backend  # 仅作存在性提示

            print("  提示：直接起服务看日志里这行最准 ——")
            print('    "Found incompatible backend(s) [...] Overriding with X out of potential backends: [...]"')
        except Exception as e:
            print(f"  跳过（{type(e).__name__}: {str(e)[:80]}）")
    else:
        print("  未指定 --model-path，跳过")

    print("\n=== 4. 建议 ===")
    usable = [n for n, ok, _ in verdict if ok]
    if "ROCM_ATTN" in usable:
        rec = "ROCM_ATTN（显式指定，别依赖 auto）"
    elif "FLASH_ATTN" in usable:
        rec = "FLASH_ATTN"
    elif "TRITON_ATTN" in usable:
        rec = "TRITON_ATTN（当前唯一可用）"
    else:
        print("  ✗ 没有一个后端可用，先解决 triton / 驱动问题")
        sys.exit(1)
    print(f"  → vllm serve ... --attention-backend {rec.split('（')[0]}")
    print("  跑通阶段优先用可用的那个；性能优化阶段再回头评估换后端的代价。")


if __name__ == "__main__":
    main()
