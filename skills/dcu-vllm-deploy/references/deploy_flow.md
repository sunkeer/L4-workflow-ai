# DCU vLLM 部署完整流程（含验收命令集）

> 由 MinerU2.5-Pro（vLLM 路线）与 PP-OCRv6（路线判定）两次实战沉淀，模型无关。
> 环境锚点：Hygon BW1000 64G/卡、DTK 26.04、torch 2.9.0+das.opt1.dtk2604、vLLM 0.11.0+das。

## 一、总流程

```
用户提供模型名
  → 步骤0  同步 scripts/ 到节点 + check_env.sh 体检
  → 步骤1  plan_deployment.py 显存匹配计算（放不下就停）
  → 步骤2  download_model.py 从 ModelScope 下载到 /root/private_data/models/<名>
  → 步骤3  装软件（DTK wheel）+ 放平台补丁
  → 步骤4  start_template.sh 启动
  → 步骤5  验收（连通 → 功能 → 性能）
  → 步骤6  （可选）写入飞书文档沉淀
```

## 二、软件安装

DTK 版 wheel 无法在线装（cupy==12.3.0 硬依赖 CUDA toolkit 编译，DCU 无），
**找用户要 wheel**，形如：

```bash
pip install torch-2.9.0+das.opt1.dtk2604-*.whl \
            torchvision-0.24.0+das.opt1.dtk2604.torch290-*.whl \
            vllm-0.11.0+das.opt1.dtk2604.torch290-*.whl \
            lmslim-*.whl
# lmslim：量化推理必装；bf16 也建议装（某些 import 路径会碰）
```

## 三、平台补丁（必放，不放起不来）

两个文件放部署目录（如 `/root/vllm_deploy/`），启动脚本已自动 `PYTHONPATH` 注入：

1. **`dcu_platform_patch.py`** —— DCU 无 pynvml/amdsmi，vLLM 平台探测回落
   UnspecifiedPlatform → `RuntimeError: Failed to infer device type`。
   补丁把 `vllm.platforms.builtin_platform_plugins["cuda"]` 换成直接返回 RocmPlatform。
2. **`sitecustomize.py`** —— vLLM V1 的 EngineCore worker 是 spawn 子进程，
   反序列化 import 阶段就触发平台惰性探测，主进程 patch 传不过去；site 机制
   让每个解释器（含子进程）启动即生效。绝不能拖垮解释器（内部 try/except）。

## 四、启动

```bash
cd /root/vllm_deploy
MODEL_PATH=/root/private_data/models/<模型名> \
SERVED_NAME=<服务名> PORT=8000 TP_SIZE=1 \
nohup bash start_template.sh > serve.log 2>&1 &

tail -f serve.log     # 盯三个节点：权重加载 → CUDA graph 捕获 → Uvicorn running on 0.0.0.0:8000
ps -o ppid= -p $(pgrep -f "vllm.*serve" | head -1)   # PPID=1 → 容器 init 收养，断 SSH 不掉
```

多模态模型：加 `LIMIT_MM='{"image":2}'`（**必须 JSON 写法**）。
graph 捕获失败：加 `ENFORCE_EAGER=1`。

## 五、验收命令集（按深度递进）

### 5.1 连通性
```bash
export BASE=http://127.0.0.1:8000   # 远端换节点 IP（服务监听 0.0.0.0）
curl -s $BASE/v1/models | python3 -m json.tool
curl -s $BASE/metrics | grep -E '^vllm:' | head -30
curl -s -w '\nHTTP=%{http_code}\n' $BASE/health
```

### 5.2 单请求（纯文本）
```bash
curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "<SERVED_NAME>",
  "messages": [{"role": "user", "content": "你好，请自我介绍。"}],
  "max_tokens": 256, "temperature": 0
}' | python3 -m json.tool
```

### 5.3 多模态（图片 → 输出）
```bash
export IMG_B64=$(base64 -w 0 /path/to/test.png)
curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" -d @- <<JSON
{
  "model": "<SERVED_NAME>",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,$IMG_B64"}},
    {"type": "text", "text": "描述这张图片的内容。"}
  ]}],
  "max_tokens": 4096, "temperature": 0
}
JSON
```
> 大 base64 别用 `bash -c`/heredoc 套 curl（多层转义会截断，报 Incorrect padding），
> 大负载一律写 JSON 文件再 `-d @file.json` 或用 Python 客户端。

### 5.4 并发压测（10 路）
```bash
for i in $(seq 1 10); do
  curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"<SERVED_NAME>","messages":[{"role":"user","content":"写100字介绍"}],"max_tokens":512}' \
    > /tmp/r$i.json &
done; wait
grep -l '"finish_reason"' /tmp/r*.json | wc -l   # 期望 10
curl -s $BASE/metrics | grep -E 'num_requests_(running|waiting)'
```

### 5.5 性能基准（看 usage）
```bash
curl -s $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d '{...}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
# vLLM 日志里看 prompt/decode tokens per second
```

### 5.6 异常回归（应优雅报 4xx，不崩服务）
```bash
curl -s -o /dev/null -w '%{http_code}\n' $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"not-exist","messages":[{"role":"user","content":"x"}]}'
# 期望 404；发超 max_model_len 的输入期望 400；超 limit-mm-per-prompt 图数期望 400
```

### 5.7 重启 / 常驻运维
```bash
pkill -f "vllm.*serve" && sleep 5
cd /root/vllm_deploy && MODEL_PATH=... nohup bash start_template.sh > serve.log 2>&1 &
# 换卡：CUDA_VISIBLE_DEVICES=1
```

## 六、验收通过后（可选）

- 关键数据（吞吐、显存、graph 捕获数）回填部署文档
- 用 `feishu-doc-write` skill 把流程沉淀到飞书 wiki
- 日志与结论写入持久化目录（`/root/private_data`，**不要写 /workspace**——CRD 容器回收会丢）
