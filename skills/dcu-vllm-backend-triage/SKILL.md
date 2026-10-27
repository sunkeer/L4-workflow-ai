---
name: dcu-vllm-backend-triage
description: 在海光 DCU / DTK 镜像上排查 vLLM 起服务时「attention 后端不可用」这一类崩溃，一次定位到该用哪个 --attention-backend。当出现 AttributeError '_OpNamespace' '_rocm_C' object has no attribute 'paged_attention'、AssertionError 'FlashAttention version not detected'、import 期 warning 'Failed to import from vllm._rocm_C'、RuntimeError 'Failed to infer device type'、或 vLLM 在 DCU 上一加载完权重就崩在 attention 相关代码时使用。
agent_created: true
---

# dcu-vllm-backend-triage —— DCU 上 vLLM attention 后端选型与排障

DTK 版 vLLM 常常缺几个编译期扩展，导致默认挑中的 attention 后端不可用。
现象都是「权重加载成功、一开始跑推理就崩」，看着像模型问题，其实是后端选错了。

## 先跑体检

```bash
python3 scripts/probe_attention_backends.py
```

输出三块：① `vllm._rocm_C` / `vllm_flash_attn` / `aiter` 是否存在；
② vLLM 对该模型会选哪些候选后端；③ 直接给出建议的 `--attention-backend`。

## 判定树

```
起服务崩在 attention
├─ import 期有 "Failed to import from vllm._rocm_C"
│   ├─ 崩在 torch.ops._rocm_C.paged_attention
│   │     → ROCM_ATTN 不可用（_rocm_C 没编）
│   └─ 崩在别的 _rocm_C 算子
│         → 见下方「与 cutlass 那类 _C 缺失的区别」
├─ AssertionError "FlashAttention version not detected"
│   → FLASH_ATTN 不可用：机器上装的是普通 flash_attn，
│     vLLM 要的是它自己的 vllm_flash_attn（Python 侧版本探测读不到）
└─ 两个都不可用
      → 用 --attention-backend TRITON_ATTN（纯 Triton，不依赖任何编译扩展）
```

## 实测（2026-10-27，BW1000 / DTK 26.04 / vLLM 0.29 / torch 2.11.0+das）

| 后端 | 结果 |
| --- | --- |
| 默认（auto → ROCM_ATTN） | ❌ `AttributeError: '_OpNamespace' '_rocm_C' object has no attribute 'paged_attention'` |
| `FLASH_ATTN` | ❌ `AssertionError: FlashAttention version not detected` |
| `TRITON_ATTN` | ✅ 正常 |

`pip list` 里 `flash_attn 2.8.3+dtk2604` 是有的，但 `import vllm_flash_attn` 报
ModuleNotFoundError —— 这是两个不同的包，别被包名骗了。

## 修复

```bash
vllm serve <MODEL> ... --attention-backend TRITON_ATTN
```

启动日志里应该能看到后端被显式锁定；如果还有 `Overriding with ROCM_ATTN`
说明参数没生效（拼错或没传到 EngineCore）。

## 与「cutlass 那类 `_C` 缺失」的区别

同样是 `'_OpNamespace' '_C' object has no attribute ...`，但根因完全不同，别混：

| 特征 | attention 后端问题（本 skill） | cutlass 算子能力探测问题 |
| --- | --- | --- |
| 报错时机 | **加载完权重、第一次推理** | **import 期**，服务还没起 |
| 报错算子 | `paged_attention` / `paged_attention_v1/v2` | `cutlass_scaled_mm_supports_fp8` 之类 `*_supported` |
| 位置 | `vllm/v1/attention/backends/rocm_attn.py` | `vllm/model_executor/layers/quantization/utils/w8a8_utils.py` |
| 解法 | 换 `--attention-backend` | 打「能力探测兜底返回 False」的补丁（见 `dcu-vllm-deploy` 的 `dcu_cutlass_patch.py`） |

## 附带踩到的两个小坑

| 现象 | 说明 |
| --- | --- |
| import 期 `RuntimeError: Failed to infer device type` | DCU 无 pynvml/amdsmi，需平台补丁锁 RocmPlatform（见 `dcu-vllm-deploy`） |
| 日志刷 `cpuinfo ... JSONDecodeError` | vLLM 的匿名用量上报拿不到 CPU 信息（无外网），**无害**，不影响服务 |
| `WARNING: /opt/hyhal/lib/cmake/rocm_smi doesn't exist` | 驱动版本提示，`rocm_smi_lib` 兜底，**无害** |

## 边界

- 只解决「后端选型」这一类；模型结构不兼容、OOM、权重加载失败不在此列
- 换后端会影响性能（TRITON_ATTN 通常比调优过的 ROCm kernel 慢），
  **跑通优先用它，性能优化阶段再回头试 ROCM_ATTN / FLASH_ATTN（需要先补上对应扩展）**
