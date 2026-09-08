#!/bin/bash
# ============================================================
# 通用 vLLM 推理服务启动脚本（Hygon DCU / DTK 26.x）
# 模型无关：任何 vLLM 可加载架构（LLM/多模态）都能用本脚本起服务。
#
# 用法（所有参数用环境变量覆盖）：
#   MODEL_PATH=/root/private_data/models/Qwen2.5-7B-Instruct \
#   SERVED_NAME=Qwen2.5-7B PORT=8000 TP_SIZE=1 \
#   nohup bash start_template.sh > serve.log 2>&1 &
#
# 多模态模型额外加：
#   LIMIT_MM='{"image":2}'        # 注意必须是 JSON 写法，image=2 会解析失败
#
# 可选开关：
#   ENFORCE_EAGER=1               # CUDA graph 捕获失败时用（更慢但更稳）
#   GPU_MEM_UTIL=0.90  MAX_MODEL_LEN=8192
#   CUDA_VISIBLE_DEVICES=0,1      # 指定卡
# ============================================================
# 注意：不要用 set -u —— DTK 的 env.sh 引用未定义变量（CMAKE_PREFIX_PATH 等）
set -eo pipefail

# ---- 1. DTK / DCU 运行时（必须 source：torch 需要 libgalaxyhip.so.5） ----
source /opt/dtk/env.sh

# ---- 2. 参数（全部可用环境变量覆盖） ----
MODEL_PATH="${MODEL_PATH:?必须提供 MODEL_PATH（权重目录，如 /root/private_data/models/<模型名>）}"
HOST_ADDR="${HOST_ADDR:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_PATH")}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DTYPE="${DTYPE:-auto}"

# 多模态：每条 prompt 最多几张图。空 = 纯文本模型，不加该参数
LIMIT_MM="${LIMIT_MM:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# 部署目录（sitecustomize.py / dcu_platform_patch.py 所在处，脚本自动定位）
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${DEPLOY_DIR}${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "[start] ERROR: 权重目录不存在: $MODEL_PATH" >&2
    exit 1
fi

MM_FLAG=""
if [ -n "$LIMIT_MM" ]; then
    MM_FLAG="--limit-mm-per-prompt $LIMIT_MM"
fi
EAGER_FLAG=""
if [ -n "${ENFORCE_EAGER:-}" ]; then
    EAGER_FLAG="--enforce-eager"
fi

echo "[start] model    = $MODEL_PATH"
echo "[start] served   = $SERVED_NAME"
echo "[start] listen   = $HOST_ADDR:$PORT"
echo "[start] tp=$TP_SIZE dtype=$DTYPE max_len=$MAX_MODEL_LEN gpu_util=$GPU_MEM_UTIL"
echo "[start] devices  = CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[start] patchdir = $DEPLOY_DIR (sitecustomize 全局生效)"

# ---- 3. 启动 vLLM OpenAI 兼容服务 ----
# DCU 上 vLLM 平台探测会失败（无 pynvml/amdsmi → UnspecifiedPlatform →
# RuntimeError: Failed to infer device type），
# 必须先 import dcu_platform_patch 强制 RocmPlatform；
# spawn 出来的 EngineCore worker 由同目录 sitecustomize.py 兜底。
exec python -c "import sys; import dcu_platform_patch; from vllm.entrypoints.cli.main import main; sys.exit(main())" \
    serve "$MODEL_PATH" \
    --host "$HOST_ADDR" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    $MM_FLAG \
    $EAGER_FLAG
