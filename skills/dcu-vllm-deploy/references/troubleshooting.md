# DCU vLLM 部署坑点手册（全部实战踩过）

## A. 平台探测类

### A1. `RuntimeError: Failed to infer device type`
- **原因**：DCU 无 pynvml（NVIDIA）也无 amdsmi（AMD），vLLM 平台探测回落
  UnspecifiedPlatform，DeviceConfig 初始化直接抛错。
- **解法**：`dcu_platform_patch.py` 覆盖 `builtin_platform_plugins["cuda"]`。
  默认 RocmPlatform（跳过 NVIDIA cutlass 的 FP8 路径，走 HIP/lmslim）；
  实测有问题用 `VLLM_DCU_PLATFORM=cuda` 切回 CUDA 兼容模式。

### A2. 主进程打了补丁还是崩（EngineCore 阶段）
- **原因**：vLLM V1 的 EngineCore worker 是 **multiprocessing spawn** 出的独立进程，
  反序列化 import 阶段就触发平台惰性探测，主进程的 patch 传不过去。
- **解法**：补丁同时放 `sitecustomize.py` 进 PYTHONPATH 目录——site 机制保证
  每个解释器（含子进程）启动即生效。注意：
  - sitecustomize 绝不能抛异常（未装 vLLM 的解释器会全局炸）
  - 不能用环境变量做「已 patch」标记——spawn 子进程会继承环境变量导致跳过

### A3. `AttributeError: '_OpNamespace' '_C' object has no attribute 'cutlass_scaled_mm_supports_fp8'`
- **现象**：服务还没起来就崩，栈底是
  `quantization/utils/w8a8_utils.py:94  CUTLASS_FP8_SUPPORTED = cutlass_fp8_supported()`
  → `_custom_ops.py:723  torch.ops._C.cutlass_scaled_mm_supports_fp8(capability)`。
- **根因**：das 版 vLLM 的 `_C.abi3.so` **没有编译 NVIDIA CUTLASS 算子**（DCU 无 FP8
  单元，移植时省掉了）。实测 `import vllm._C` 后
  `[n for n in dir(torch.ops._C) if 'cutlass' in n.lower()]` 仍为 `[]`。
  而 `w8a8_utils.py` 在**模块级**（import 期）就做能力探测：
  ```python
  def cutlass_fp8_supported():
      if not current_platform.is_cuda():
          return False          # rocm 平台安全
      return ops.cutlass_scaled_mm_supports_fp8(capability)   # is_cuda() 为真 → 崩
  ```
- **解法（双保险，两件都要做）**：
  1. `dcu_cutlass_patch.py`：给 `torch._ops._OpNamespace.__getattr__` 打兜底，
     缺失的 `cutlass*` 且名字含 `support`（能力探测类）返回恒 False 桩。
     语义正确（DCU 确实不支持），且**刻意不兜底真计算算子**，避免藏错。
  2. 平台锁 `RocmPlatform`，让 `is_cuda()` 为 False，根本不进这条路径。
- **排查提示**：日志里 `Automatically detected platform cuda.` 有欺骗性——vLLM
  打印的是**插件键名**（我们把 builtin 的 `cuda` 键改写了），不是平台类名。
  要确认真实平台，打印 `type(current_platform).__name__` 与 `is_cuda()`。
  实测：键 cuda → `RocmPlatform`(is_cuda False)；`VLLM_DCU_PLATFORM=cuda` →
  `NonNvmlCudaPlatform`(is_cuda True，正是踩坑那个)。

### A4. `ValueError: Free memory on device (X/Y GiB) ... less than desired GPU memory utilization`
- **根因**：同一张卡上**已经有另一个 vLLM 实例**在跑（常见于重复执行 start.sh，
  或 SSH 里 `&` 后台启动后又用 nohup 起了一个）。1.2B 模型 + `gpu_util=0.9`
  在 64GiB 卡上要 57.59GiB，第二个实例只剩 5.73GiB → 必然失败。
- **解法**：`ps -eo pid,ppid,etime,cmd | grep '[v]llm.entrypoints'` 确认实例数，
  只保留一个。清理时注意 **`pkill -f 'VLLM::EngineCore'` 会匹配到 pkill 自己所在的
  shell 命令行并把它杀掉**（自杀导致后续命令全不执行），用
  `pkill -f '[V]LLM::EngineCore'` 的括号写法规避。

## B. DTK 环境类

### B1. `libgalaxyhip.so.5: cannot open shared object file`
- 没执行 `source /opt/dtk/env.sh`。启动脚本第一行必须 source。

### B2. start.sh 用了 `set -u` 静默中止
- DTK env.sh 引用未定义变量（CMAKE_PREFIX_PATH 等），`-u` 下直接中止。
- 用 `set -eo pipefail`，**永远不要 `set -u`**。

### B3. vLLM 装不上：`cupy==12.3.0` 编译失败
- 该 wheel 硬依赖 cupy sdist，需要 CUDA toolkit 编译，DCU 没有。
- 解法：找海光 DTK 版 cupy wheel，或 `pip install --no-deps` 硬装再补依赖。
- 有的节点干脆装不上 vLLM（如只跑 ONNX 的 11368 节点）——先 check_env.sh 确认。

## C. vLLM 参数类

### C1. `--limit-mm-per-prompt` 解析失败
- 必须写 JSON：`--limit-mm-per-prompt '{"image":2}'`；写 `image=2` 会解析失败。

### C2. 日志出现「bw1000 不支持 fp8」
- **只是告警，不阻断**：BW1000 无 FP8 单元，但 bf16 推理根本不调 FP8 kernel。
  真跑量化（w8a8/GPTQ/AWQ）才需要海光官方 lmslim（awqlite/gptqlite）。

### C3. CUDA graph 捕获失败
- 加 `ENFORCE_EAGER=1`（--enforce-eager）降级，吞吐损失换稳定。

## D. 模型加载类

### D4. 大模型 mmap 报 ENOMEM（SGLang/vLLM 大权重）
- 容器非 privileged 时 `sysctl vm.overcommit_memory` 改不了（只读 fs）。
- 缓解：`--weight-loader-disable-mmap`，或 LD_PRELOAD 钩子给 mmap 强制
  MAP_NORESERVE（需要 gcc 现编 .so）。
- 注意：Committed_AS 是**宿主机全局**的，同节点其他租户波动会让「昨天能起今天起不来」。

### D5. config.json 无 `architectures` / 无 tokenizer
- 不是 vLLM 可加载架构（如 PP-OCRv6 的 ONNX 导出）。
- 改走 onnxruntime + FastAPI：ocr 推理实测 `POST /ocr/file`（multipart）、
  `/ocr/base64`、`/v1/ocr`（OpenAI 风格别名）都可行。ONNX 推理注意：
  先确认输出是否已 softmax、CTC 输出按「末维==类别数」判断转置。

## E. 运行时 / 网络类

### E5. 运行时报外网连接失败
- 服务运行时不需要外网；start.sh 里 `unset http_proxy https_proxy` 避免代理依赖。

### E6. 大 base64 走 curl 被截断（Incorrect padding）
- `bash -c`/heredoc 多层转义会截大 payload。写 JSON 文件 `-d @file.json`，
  或 Python urllib 客户端（零依赖）。

### E7. CRD notebook 容器被回收，/workspace 全丢
- SSH 入口可能在会话间隙被重路由到**新实例**（hostname 变了就是证据）。
- 模型、日志、结论一律写 `/root/private_data`（PVC 持久化）；`/workspace` 只放可重建物。

## F. 流程纪律

- **先算显存再下载**（plan_deployment.py 在下载前完成评估）
- 下载目录固定 `/root/private_data/models/<模型名>`
- 部署目录固定一套（如 `/root/vllm_deploy`），补丁+启动脚本+日志集中管理
- 服务起来后确认 PPID=1（nohup & 后被容器 init 收养，断 SSH 不掉）
- 验收三层：连通（/v1/models + /health）→ 功能（真实样本）→ 性能（usage/metrics）
