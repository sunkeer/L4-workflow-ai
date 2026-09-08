#!/bin/bash
# ============================================================
# DCU vLLM 部署前置体检
# 用途：部署前一次性确认节点状态（DTK/torch/vLLM/lmslim/显存/磁盘/端口）
# 在节点上运行：bash check_env.sh
# ============================================================
# 注意：不要 set -u —— DTK 的 env.sh 引用未定义变量（CMAKE_PREFIX_PATH 等），
# 在 -u 下脚本会直接中止（实测必踩）
set -o pipefail

DTK_ENV="${DTK_ENV:-/opt/dtk/env.sh}"

echo "===== 0. 节点 / OS ====="
hostname; uname -r; nproc

echo
echo "===== 1. DTK 运行时 ====="
if [ -f "$DTK_ENV" ]; then
    # 必须 source：torch 依赖 LD_LIBRARY_PATH 里的 libgalaxyhip.so.5
    # shellcheck disable=SC1090
    source "$DTK_ENV"
    echo "sourced $DTK_ENV (DTK_HOME=$DTK_HOME)"
else
    echo "[WARN] 未找到 $DTK_ENV —— torch 可能起不来"
fi

echo
echo "===== 2. DCU 卡与显存 ====="
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname 2>/dev/null | head -8
    rocm-smi --showmeminfo vram 2>/dev/null | grep -E "Total|Used" | head -10
elif command -v hy-smi >/dev/null 2>&1; then
    hy-smi 2>/dev/null | head -15
else
    echo "[WARN] rocm-smi / hy-smi 都不可用，后面用 torch 探测"
fi

echo
echo "===== 3. torch（DTK 版） ====="
python3 - <<'PY'
try:
    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n:
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"torch {torch.__version__}  devices={n}  "
              f"per-card={total/2**30:.1f}GiB  free={free/2**30:.1f}GiB  name={p.name}")
    else:
        print(f"torch {torch.__version__}  cuda.is_available()=False（没 source DTK env？）")
except Exception as e:
    print(f"[FAIL] import torch 失败: {e}")
PY

echo
echo "===== 4. vLLM / lmslim / torchvision ====="
python3 - <<'PY'
for mod in ("vllm", "lmslim", "torchvision", "modelscope"):
    try:
        m = __import__(mod)
        print(f"{mod:12} {getattr(m, '__version__', '?'):20} {getattr(m, '__file__', '?')}")
    except Exception as e:
        print(f"{mod:12} 未安装/不可导入（{type(e).__name__}）")
PY

echo
echo "===== 5. 磁盘（模型目录） ====="
if [ -d /root/private_data ]; then
    df -h /root/private_data
    ls /root/private_data/models/ 2>/dev/null | head -10
else
    echo "[WARN] /root/private_data 不存在 —— 模型放哪？"
    df -h /root
fi

echo
echo "===== 6. 常用端口占用 ====="
ss -ltn 2>/dev/null | grep -E ':(8000|8001|8111) ' || echo "8000/8001/8111 空闲"

echo
echo "===== 7. 平台探测依赖（预期都缺，正常现象） ====="
python3 - <<'PY'
for mod in ("pynvml", "amdsmi"):
    try:
        __import__(mod)
        print(f"{mod}: 有")
    except ImportError:
        print(f"{mod}: 无（DCU 上正常，需走 dcu_platform_patch）")
PY

echo
echo "体检完成。缺 vLLM/DTK wheel 的，找海光 DTK 版 wheel："
echo "  pip install <用户提供>.whl   # vllm-0.11.0+das.opt1.dtk2604.torch290-*.whl 等"
