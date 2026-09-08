---
name: dcu-vllm-deploy
description: 在 Hygon DCU（DTK 26.x / HIP，如 BW1000/K100 系列）上把任意 ModelScope/HuggingFace 架构的 LLM/多模态模型通过 vLLM 部署成 OpenAI 兼容推理服务的通用流程。当用户提供模型名称并要求部署到 DCU 机器、或需要先评估「显存是否放得下」再决定是否下载权重时使用。流程固定为三步：①显存匹配计算（模型权重 + KV cache vs 本机 DCU 总显存，给出推荐 TP/dtype）；②按模型名自动在 ModelScope 查找并下载到 /root/private_data/models/<模型名>；③部署（DTK 环境体检、平台补丁、参数化启动脚本、端到端验收）。不绑定具体模型，已用 MinerU2.5-Pro（vLLM 路线）与 PP-OCRv6（非 LLM，ONNX 路线判定）两个案例验证过流程边界。
---

# dcu-vllm-deploy —— DCU 通用 vLLM 推理服务部署

把「用户提供模型名 → 显存评估 → 自动下载 → 部署 → 验收」固化为可复用流程。
**模型无关**：任何 vLLM 可加载的架构（config.json 里有 `architectures` + tokenizer）都走本流程；纯 ONNX/非自回归模型（如 PP-OCRv6）vLLM 加载不了，会在第 0 步被识别并改走 onnxruntime + FastAPI 路线（本 skill 只做判定提示，不展开）。

## 输入要求（开始前向用户确认）

| 输入 | 必需 | 说明 |
|---|---|---|
| 模型名称 | ✅ | 短名（如 `MinerU2.5-Pro`，自动搜索 ModelScope）或完整 id（如 `OpenGVLab/MinerU2.5-Pro`） |
| DCU 节点访问方式 | ✅ | SSH 命令（如 `ssh -p 11360 root@ssh.zzai.scnet.cn`）或已有的 remote_ssh 辅助脚本 |
| 服务端口 | 可选 | 默认 8000 |
| 部署 dtype / TP | 可选 | 默认 bf16（BW1000 无 FP8 单元）、TP=1；第 1 步计算会给推荐值 |

## 步骤 0：同步脚本到节点 + 前置判定

把本 skill 的 `scripts/` 目录同步到节点（如 `/root/vllm_deploy/`）：

```bash
scp -P <PORT> scripts/*.sh scripts/*.py root@<HOST>:/root/vllm_deploy/
```

在节点上运行体检：

```bash
bash /root/vllm_deploy/check_env.sh
```

确认四件事：① `source /opt/dtk/env.sh` 后 `torch.cuda.is_available() == True`；② vLLM 是否已装（**DTK 版 vLLM 无法在线安装，缺了要向用户要 wheel**，形如 `vllm-0.11.0+das.opt1.dtk2604.torch290-*.whl`）；③ 每卡显存容量与已占用；④ `/root/private_data` 磁盘余量（模型都放这里）。

> ⚠️ 前置判定：拿到模型 config.json 后（第 1 步会自动拉），若无 `architectures` 或无 tokenizer 文件 → 不是 vLLM 可加载架构 → 停止 vLLM 流程，提示用户改走 ONNX/其他路线（参考 PP-OCRv6 案例：onnxruntime + FastAPI）。

## 步骤 1：显存匹配计算（下载之前先算，避免白下几十 GB）

在节点上运行：

```bash
python3 /root/vllm_deploy/plan_deployment.py <模型名>
```

脚本会自动完成：解析 ModelScope 模型（短名自动搜索）→ 拉文件清单估算权重大小 + config.json 取层数/头数 → 用 torch 读本机 DCU 每卡显存 → 对 TP=1..N 逐一计算并输出：

- 权重占显存（按部署 dtype 换算，量化模型按文件实际大小）
- KV cache 每 token 字节数、剩余 KV 预算、满长并发估算
- **可行 TP 列表与推荐值**；放不下时直接给出结论（换节点/换量化版本）

判定口径（与 vLLM 实际行为一致）：
```
每卡需求 = 权重/TP + 开销(≈2GiB + 5%碎片) + KV cache 预算
每卡可用 = 总显存 × gpu-memory-utilization(默认0.90) − 已被其他进程占用
可行 ⟺ 每卡需求(不含KV) ≤ 每卡可用，且 KV 预算 ≥ 每卡显存的 20%（太小说明吞吐会很差）
```

**只有 verdict 是 OK 才继续第 2 步**；放不下时把计算结果原样反馈给用户，让用户决定（换量化版/换节点/减小 max_model_len）。

## 步骤 2：ModelScope 查找 + 下载

```bash
python3 /root/vllm_deploy/download_model.py <模型名或完整id>
# 默认下载到 /root/private_data/models/<模型名>；已存在且非空会跳过（--force 强制重下）
```

脚本优先用 modelscope SDK（`snapshot_download`，支持断点续传），未安装会自动 `pip install modelscope`（节点需外网或代理）。下载完校验目录里确有 `.safetensors`/`.bin`/`.onnx`。

> 关键路径约定：**模型一律放 `/root/private_data/models/<模型名>`**。`/root/private_data` 是持久化挂载（PVC），`/workspace` 随 CRD 容器回收会丢。

## 步骤 3：部署（已验证的流程，按序执行）

1. **软件安装**（离线节点：wheel 由用户提供，`pip install --no-deps` 视情况）：
   - torch（DTK 版，通常预装）：`2.9.0+das.opt1.dtk2604` 之类
   - vLLM（DTK 版 wheel）
   - **lmslim**（海光量化包）：只要走量化（FP8/INT8/w8a8）就必须装；bf16 推理不调用，但 vLLM 某些 import 路径会碰它，装上最稳
   - torchvision（DTK 版）：多模态模型（有 image processor）需要
2. **两个平台补丁**（已随 scripts/ 同步到节点，放 `/root/vllm_deploy/`）：
   - `dcu_platform_patch.py`：DCU 无 pynvml/amdsmi，vLLM 平台探测失败会抛 `RuntimeError: Failed to infer device type`；补丁强制走 RocmPlatform（`VLLM_DCU_PLATFORM=cuda` 可切回）
   - `sitecustomize.py`：vLLM V1 引擎 EngineCore worker 是 spawn 子进程，主进程 patch 传不过去；site 机制让每个解释器启动即生效。**放 PYTHONPATH 目录里即可**
3. **渲染启动脚本**：复制 `start_template.sh`，设 `MODEL_PATH/SERVED_NAME/PORT/TP_SIZE`（多模态模型再加 `LIMIT_MM='{"image":2}'`），后台启动：
   ```bash
   cd /root/vllm_deploy
   MODEL_PATH=/root/private_data/models/<模型名> SERVED_NAME=<服务名> nohup bash start_template.sh > serve.log 2>&1 &
   ```
   启动后盯日志：权重加载进度 → CUDA graph 捕获 → `Uvicorn running on 0.0.0.0:<PORT>`。用 `ps -o ppid= -p <PID>` 确认 PPID=1（容器 init 收养，断 SSH 不掉）。
4. **端到端验收**（三条都要过）：
   ```bash
   curl -s http://127.0.0.1:<PORT>/v1/models                 # 返回模型名
   curl -s http://127.0.0.1:<PORT>/health                     # 200
   # 发一条真实 chat 请求（模型有输入模态就带真实样本，多模态用 data:image/png;base64,...）
   ```
   完整测试命令集（专项识别/多图/并发压测/异常回归/性能基准）见 [references/deploy_flow.md](references/deploy_flow.md) §验收。

## 坑点速查（部署失败先对照这个）

| 症状 | 原因与解法 |
|---|---|
| `RuntimeError: Failed to infer device type` | 平台探测失败 → 确认 PYTHONPATH 里有 sitecustomize.py / 启动命令带 `import dcu_platform_patch` |
| `libgalaxyhip.so.5: cannot open` | 没 source `/opt/dtk/env.sh` |
| start.sh 中途静默退出 | 用了 `set -u`（DTK env.sh 引用未定义变量）→ 删掉 -u |
| `--limit-mm-per-prompt image=2` 报错 | 必须写 JSON：`{"image":2}` |
| 日志有「bw1000 不支持 fp8」 | 只是告警不阻断（bf16 路径不触发）；量化模型才需要真正装 lmslim |
| vllm 装不上：`cupy==12.3.0` sdist 编译失败 | 该 wheel 依赖 CUDA toolkit 编译，DCU 无 → 找 DTK 版 cupy 或 `--no-deps` |
| 加载大模型 mmap ENOMEM | 见 troubleshooting.md「overcommit」节，可试 `--weight-loader-disable-mmap` |

完整坑点手册：[references/troubleshooting.md](references/troubleshooting.md)
显存估算公式与经验数值：[references/vram_estimation.md](references/vram_estimation.md)

## 边界与不适用

- 非 vLLM 架构（config.json 无 `architectures`，如 PP-OCRv6 的 ONNX 模型）→ 判定后走 onnxruntime + FastAPI，不在本 skill 范围
- SGLang / PD 分离部署 → 另行处理，本 skill 只覆盖 vLLM 单机（TP 可多卡）
- Windows 本地不能跑 scripts/，全部在 DCU 节点上执行；本地只负责 SSH 与文件同步
