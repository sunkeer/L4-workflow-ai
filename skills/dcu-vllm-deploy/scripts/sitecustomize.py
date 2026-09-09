"""sitecustomize —— 让 DCU 补丁在每个 Python 进程（含 spawn worker）自动生效。

背景
----
vLLM 0.11 的 current_platform 是惰性解析的（vllm/platforms/__init__.py 的
模块级 __getattr__，首次访问时才探测）。vLLM V1 引擎的 EngineCore worker
是用 multiprocessing spawn 启动的独立进程：它在反序列化 import 阶段就会
触发平台探测，而 start.sh 里 `import dcu_platform_patch` 只对主进程生效。

Python 的 site 机制会在解释器启动时（任何业务 import 之前）自动导入
sitecustomize 模块 —— 包括 multiprocessing spawn 出来的子进程。把补丁放
这里即可全局生效，无需修改 vLLM 源码。

这里做两件事：
  1. 平台锁定：把 vLLM 内建 cuda 插件指向 RocmPlatform（DCU 是 HIP 血统）。
     必须**同时删掉内建 rocm 键**：vLLM 的 resolve 逻辑只允许一个内建插件
     激活，两个都激活会抛 "Only one platform plugin can be activated"。
  2. CUTLASS 兜底：das 版 vLLM 没编译 cutlass_*_supported 系列算子，而
     w8a8_utils.py 在 import 期就会调用 → AttributeError 直接崩。见
     dcu_cutlass_patch.py。

用法
----
把本文件放到 PYTHONPATH 中的某个目录（部署里是 /root/mineru_deploy，
start.sh 已 export PYTHONPATH=/root/mineru_deploy）。
平台可用环境变量 VLLM_DCU_PLATFORM=cuda 切回 CUDA 兼容模式（一般不推荐）。
"""
import os
import sys

try:
    import vllm.platforms as _platforms

    # ---- 1. 平台锁定 ----
    # DCU 是 HIP / ROCm 血统：rocm 平台下 is_cuda() 为 False，vLLM 会跳过
    # NVIDIA cutlass 的 FP8/w8a8 路径，改用 HIP/lmslim 实现。
    _platform = os.getenv("VLLM_DCU_PLATFORM", "rocm").strip().lower()
    _target = (
        "vllm.platforms.cuda.CudaPlatform"
        if _platform == "cuda"
        else "vllm.platforms.rocm.RocmPlatform"
    )

    def _dcu_platform_plugin():
        """DCU 上直接指定平台：探测依赖的 pynvml/amdsmi 在 DCU 上都不存在。"""
        return _target

    # 覆盖内建 cuda 探测键，使惰性探测首次触发即命中目标平台
    _platforms.builtin_platform_plugins["cuda"] = _dcu_platform_plugin
    # 删除内建 rocm 键：否则 cuda 与 rocm 两个内建插件同时激活，
    # vLLM 会抛 "Only one platform plugin can be activated"
    _platforms.builtin_platform_plugins.pop("rocm", None)

    # 诊断日志：出现问题时一眼看出实际锁定的平台
    print(f"[dcu:sitecustomize] platform -> {_target}", file=sys.stderr)

    # ---- 2. CUTLASS 能力探测算子兜底 ----
    # 注意：sitecustomize 每个解释器进程只导入一次，无需额外去重守卫；
    # 更不能用环境变量做标记 —— spawn 子进程会继承环境变量导致跳过 patch。
    try:
        import dcu_cutlass_patch

        if dcu_cutlass_patch.patch_missing_cutlass_ops():
            print("[dcu:sitecustomize] cutlass support-op stub installed",
                  file=sys.stderr)
    except Exception as _e:  # noqa: BLE001
        print(f"[dcu:sitecustomize] cutlass patch skipped: {_e!r}",
              file=sys.stderr)
except Exception as _e:  # noqa: BLE001  sitecustomize 绝不能拖垮解释器
    # 未装 vLLM / 未 source DTK env 时（如普通 pip 调用）静默跳过
    print(f"[dcu:sitecustomize] not applied: {_e!r}", file=sys.stderr)
