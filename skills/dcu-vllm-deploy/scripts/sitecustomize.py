"""sitecustomize —— 让 vLLM 平台补丁在每个 Python 进程（含 spawn worker）自动生效。

背景
----
vLLM 0.11 的 current_platform 是惰性解析的（vllm/platforms/__init__.py 的
模块级 __getattr__，首次访问时才探测）。vLLM V1 引擎的 EngineCore worker
是用 multiprocessing spawn 启动的独立进程：它在反序列化 import 阶段就会
触发平台探测，而 start.sh 里 `import dcu_platform_patch` 只对主进程生效，
导致 worker 里平台被解析成 UnspecifiedPlatform，随后
Platform.set_device() 抛 NotImplementedError，引擎初始化失败。

Python 的 site 机制会在解释器启动时（任何业务 import 之前）自动导入
sitecustomize 模块 —— 包括 multiprocessing spawn 出来的子进程。
把补丁放这里即可全局生效，无需修改 vLLM 源码。

用法
----
把本文件放到 PYTHONPATH 中的某个目录（部署里是 /root/mineru_deploy，
start.sh 已 export PYTHONPATH=/root/mineru_deploy）。平台可选 rocm（默认）
或 cuda，通过环境变量 VLLM_DCU_PLATFORM 切换。
"""
import os

try:
    import vllm.platforms as _platforms

    # DCU 是 HIP / ROCm 血统：选 rocm 时 vLLM 会跳过 NVIDIA cutlass 的
    # FP8/w8a8 路径（is_cuda() 为 False），改用 HIP/lmslim 实现。
    # 若实测 rocm 平台有问题，可用 VLLM_DCU_PLATFORM=cuda 切回。
    # 注意：sitecustomize 每个解释器进程只导入一次，无需额外去重守卫；
    # 更不能用环境变量做标记 —— spawn 子进程会继承环境变量导致跳过 patch。
    _platform = os.getenv("VLLM_DCU_PLATFORM", "rocm").strip().lower()
    _target = (
        "vllm.platforms.cuda.CudaPlatform"
        if _platform == "cuda"
        else "vllm.platforms.rocm.RocmPlatform"
    )

    def _dcu_platform_plugin():
        """DCU 上直接指定平台：探测依赖的 pynvml/amdsmi 在 DCU 上都不存在。"""
        return _target

    # 覆盖内建 cuda 探测键，使惰性探测首次触发即命中 rocm/cuda 平台
    _platforms.builtin_platform_plugins["cuda"] = _dcu_platform_plugin
except Exception:  # noqa: BLE001  sitecustomize 绝不能拖垮解释器
    # 未装 vLLM / 未 source DTK env 时（如普通 pip 调用）静默跳过
    pass
