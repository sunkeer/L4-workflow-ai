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
