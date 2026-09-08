#!/usr/bin/env python3
"""DCU vLLM 部署前置评估：显存匹配计算 + ModelScope 模型解析。

在 DCU 节点上运行（需能访问 modelscope.cn，走已有代理环境变量）：
    python3 plan_deployment.py <模型名或 namespace/name> [--max-len 8192] [--gpu-util 0.90]

输出：
  1. 模型解析结果（完整 id、权重总大小、层数/头数、是否量化、是否 vLLM 可加载架构）
  2. 本机 DCU 显存（每卡容量、已被其他进程占用量）
  3. TP=1..N 逐一评估表：每卡权重 + 开销 + KV 预算 + 满长并发估算 + verdict
  4. 放不下时给出结论与建议；放得下时给出推荐 TP 和启动参数

判定口径（与 vLLM 实际行为一致——vLLM 按 gpu-memory-utilization 预留显存，
权重装完后剩余全部给 KV cache）：
  每卡需求(不含KV) = 权重/TP + 开销(≈2GiB 固定 + 5% 权重碎片)
  每卡可用 = 总显存 × gpu_util − 已被其他进程占用
  OK  ⟺ 需求 ≤ 可用 且 KV 预算 ≥ 每卡显存 × 20%（太小说明吞吐很差）
  TIGHT ⟺ 需求 ≤ 可用 但 KV 预算 < 20%（能跑但并发低，建议减 max-len 或加卡）
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request

MS_BASE = "https://modelscope.cn/api/v1"
GIB = 1024 ** 3
HEADERS = {"User-Agent": "dcu-vllm-deploy/1.0", "Content-Type": "application/json"}
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth")


# ---------------- ModelScope API ----------------

def _get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _post_json(url, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def search_modelscope(keyword, limit=8):
    """按关键词搜索 ModelScope，返回候选列表；失败返回 []。"""
    try:
        d = _post_json(f"{MS_BASE}/dolphin/models",
                       {"Query": keyword, "PageSize": limit, "PageNumber": 1})
        models = (d.get("Data", {}).get("Model", {}) or {}).get("Models") or []
        out = []
        for m in models:
            path, name = m.get("Path"), m.get("Name")
            if path and name:
                out.append({
                    "id": f"{path}/{name}",
                    "downloads": m.get("Downloads") or 0,
                    "chinese_name": m.get("ChineseName") or "",
                })
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[warn] ModelScope 搜索接口失败：{e}", file=sys.stderr)
        return []


def resolve_model(query):
    """短名 → 搜索选最佳候选；带 namespace 直接用。返回完整 id。"""
    if "/" in query:
        return query.strip().strip("/")
    cands = search_modelscope(query)
    if not cands:
        raise SystemExit(f"[ERROR] 无法解析模型 '{query}'。请提供完整 namespace/name（如 "
                         f"OpenGVLab/MinerU2.5-Pro），或确认网络/代理可用。")
    cands.sort(key=lambda c: c["downloads"], reverse=True)
    print(f"[info] '{query}' 未带命名空间，ModelScope 候选（按下载量排序）：")
    for i, c in enumerate(cands[:5]):
        print(f"  [{i}] {c['id']}  downloads={c['downloads']}  {c['chinese_name']}")
    if not sys.stdin.isatty():
        chosen = cands[0]
        print(f"[info] 非交互环境，自动选择 [0] {chosen['id']}")
    else:
        raw = input("选择序号（回车默认 0）: ").strip()
        idx = int(raw) if raw.isdigit() and int(raw) < len(cands) else 0
        chosen = cands[idx]
    return chosen["id"]


def fetch_file_list(model_id):
    d = _get_json(f"{MS_BASE}/models/{model_id}/repo/files?Revision=master")
    return (d.get("Data", {}) or {}).get("Files") or []


def fetch_raw_file(model_id, path):
    url = (f"{MS_BASE}/models/{model_id}/repo?Revision=master"
           f"&FilePath={urllib.parse.quote(path)}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def fetch_config(model_id):
    try:
        return json.loads(fetch_raw_file(model_id, "config.json"))
    except Exception:  # noqa: BLE001
        return {}


# ---------------- 本机 DCU 显存 ----------------

def dcu_vram():
    """返回 (卡数, 每卡总显存bytes, 当前空闲bytes, 来源字符串)。"""
    try:
        import torch  # noqa: PLC0415  需先 source /opt/dtk/env.sh
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            total = torch.cuda.get_device_properties(0).total_memory
            free, _total = torch.cuda.mem_get_info()
            return n, total, free, f"torch {torch.__version__}"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] torch 探测失败（{e}），回退 rocm-smi", file=sys.stderr)
    for smi in ("/opt/dtk/bin/rocm-smi", "rocm-smi"):
        try:
            out = subprocess.run([smi, "--showmeminfo", "vram"], capture_output=True,
                                 text=True, timeout=30).stdout
            totals = [int(x) for x in re.findall(r"Total Memory \(B\):\s*(\d+)", out)]
            used = [int(x) for x in re.findall(r"Used Memory \(B\):\s*(\d+)", out)]
            if totals:
                return (len(totals), totals[0],
                        totals[0] - (used[0] if used else 0), smi)
        except Exception:  # noqa: BLE001
            continue
    raise SystemExit("[ERROR] 无法读取 DCU 显存：torch 与 rocm-smi 都失败。"
                     "是否已 source /opt/dtk/env.sh？")


# ---------------- 估算 ----------------

def analyze(model_id, files, cfg):
    weights = sum(f.get("Size") or 0 for f in files
                  if f.get("Path", "").endswith(WEIGHT_EXT))
    has_tokenizer = any(f.get("Path", "") in
                        ("tokenizer.json", "tokenizer_config.json",
                         "vocab.json", "spiece.model", "merges.txt")
                        for f in files)
    archs = cfg.get("architectures") or []
    quant = bool(cfg.get("quantization_config"))
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layers") or 0
    heads = cfg.get("num_attention_heads") or cfg.get("n_head") or 0
    kv_heads = cfg.get("num_key_value_heads") or heads
    hidden = cfg.get("hidden_size") or cfg.get("n_embd") or 0
    head_dim = (hidden // heads) if heads and hidden and hidden >= heads else 0
    return {
        "id": model_id, "weights": weights, "has_tokenizer": has_tokenizer,
        "archs": archs, "quantized": quant, "layers": layers,
        "kv_per_token": 2 * layers * kv_heads * head_dim * 2 if layers and kv_heads and head_dim else 0,
        "head_dim": head_dim, "heads": heads, "kv_heads": kv_heads,
    }


def main():
    ap = argparse.ArgumentParser(description="DCU vLLM 部署显存匹配计算")
    ap.add_argument("model", help="模型名或 namespace/name")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--tp", type=int, default=0, help="只评估指定 TP（默认全试一遍）")
    args = ap.parse_args()

    model_id = resolve_model(args.model)
    print(f"\n=== 模型：{model_id} ===")
    files = fetch_file_list(model_id)
    if not files:
        raise SystemExit("[ERROR] 拿不到模型文件清单，检查模型 id 是否存在")
    cfg = fetch_config(model_id)
    info = analyze(model_id, files, cfg)

    print(f"权重文件总大小 : {info['weights'] / GIB:.2f} GiB"
          f"{'（含量化，部署按原大小加载）' if info['quantized'] else '（bf16/fp16 存储）'}")
    print(f"architectures  : {info['archs'] or '（无！）'}")
    print(f"tokenizer      : {'有' if info['has_tokenizer'] else '无（vLLM 可能加载失败）'}")
    print(f"层数/头数      : layers={info['layers']} heads={info['heads']} "
          f"kv_heads={info['kv_heads']} head_dim={info['head_dim']}")

    if not info["archs"] or not info["has_tokenizer"]:
        print("\n[判定] 非 vLLM 可加载架构（无 architectures 或无 tokenizer）。"
              "此类模型（如纯 ONNX 的 PP-OCRv6）应走 onnxruntime + FastAPI 路线，"
              "不要继续 vLLM 部署。")
        sys.exit(2)

    n_dev, total, free, src = dcu_vram()
    print(f"\n=== 本机 DCU（{src}）===")
    print(f"卡数={n_dev}  每卡 {total / GIB:.1f} GiB  "
          f"当前空闲 {free / GIB:.1f} GiB（被其他进程占用 {(total - free) / GIB:.1f} GiB）")

    usable = total * args.gpu_util - (total - free)
    weights = info["weights"]
    kv_per_tok = info["kv_per_token"]

    print(f"\n=== TP 评估（gpu_util={args.gpu_util}, max_model_len={args.max_len}）===")
    print(f"{'TP':>3} {'权重/卡':>9} {'开销/卡':>8} {'KV预算/卡':>10} "
          f"{'满长并发':>8}  verdict")

    tps = [args.tp] if args.tp else [t for t in (1, 2, 4, 8) if t <= n_dev]
    best = None
    for tp in tps:
        w_card = weights / tp
        overhead = 2 * GIB + 0.05 * w_card
        kv_budget = usable - w_card - overhead
        need_no_kv = w_card + overhead
        if kv_budget <= 0:
            verdict = "FAIL（权重都放不下）"
        elif kv_budget < total * 0.20:
            verdict = "TIGHT（KV 太小，吞吐差）"
        else:
            verdict = "OK"
            if best is None:
                best = tp
        conc = (kv_budget / kv_per_tok / args.max_len) if kv_per_tok else -1
        conc_str = f"~{conc:.0f}" if conc >= 0 else "N/A"
        print(f"{tp:>3} {w_card / GIB:>8.1f}G {overhead / GIB:>7.1f}G "
              f"{max(kv_budget, 0) / GIB:>9.1f}G {conc_str:>8}  {verdict}")

    print()
    if best:
        print(f"[结论] 推荐部署：TP={best}，dtype={'量化原样' if info['quantized'] else 'bf16'}，"
              f"max_model_len={args.max_len}，gpu_memory_utilization={args.gpu_util}")
        print(f"[下一步] python3 download_model.py {model_id}   "
              f"# 下载到 /root/private_data/models/{model_id.split('/')[-1]}")
    else:
        print("[结论] 当前节点放不下该模型。可选：①换量化版本（权重可小 2~4 倍）"
              "②减小 max_model_len ③换更大显存节点 ④多卡节点上提高 TP。")
        sys.exit(1)


if __name__ == "__main__":
    main()
