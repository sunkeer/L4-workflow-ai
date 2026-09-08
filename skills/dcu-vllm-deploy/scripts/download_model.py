#!/usr/bin/env python3
"""从 ModelScope 下载模型到 /root/private_data/models/<模型名>。

用法（在 DCU 节点上运行）：
    python3 download_model.py <namespace/name 或短名>
    python3 download_model.py OpenGVLab/MinerU2.5-Pro --dest /root/private_data/models
    python3 download_model.py <id> --force        # 已存在也重下

行为：
  1. 短名先调 ModelScope 搜索解析（与 plan_deployment.py 同一套逻辑）
  2. 优先用 modelscope SDK snapshot_download（断点续传、按文件校验）
  3. 未装 SDK 自动 pip install modelscope（节点需外网/代理；代理走既有环境变量）
  4. 目录已存在且含权重文件 → 跳过（--force 强制重下）
  5. 下载完校验目录里确有 .safetensors/.bin/.onnx 之一
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import urllib.request

DEFAULT_DEST = "/root/private_data/models"
MS_BASE = "https://modelscope.cn/api/v1"
HEADERS = {"User-Agent": "dcu-vllm-deploy/1.0", "Content-Type": "application/json"}
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".onnx")


def _post_json(url, payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def search_modelscope(keyword, limit=8):
    try:
        d = _post_json(f"{MS_BASE}/dolphin/models",
                       {"Query": keyword, "PageSize": limit, "PageNumber": 1})
        models = (d.get("Data", {}).get("Model", {}) or {}).get("Models") or []
        out = []
        for m in models:
            path, name = m.get("Path"), m.get("Name")
            if path and name:
                out.append({"id": f"{path}/{name}", "downloads": m.get("Downloads") or 0})
        return out
    except Exception:  # noqa: BLE001
        return []


def resolve_model(query):
    if "/" in query:
        return query.strip().strip("/")
    cands = sorted(search_modelscope(query), key=lambda c: c["downloads"], reverse=True)
    if not cands:
        raise SystemExit(f"[ERROR] 无法解析 '{query}'，请给完整 namespace/name")
    print(f"[info] 自动选择候选 [0]：{cands[0]['id']}（downloads={cands[0]['downloads']}）")
    return cands[0]["id"]


def has_weights(path):
    if not os.path.isdir(path):
        return False
    for _, _, names in os.walk(path):
        if any(n.endswith(WEIGHT_EXT) for n in names):
            return True
    return False


def ensure_sdk():
    try:
        importlib.import_module("modelscope")
        return True
    except ImportError:
        pass
    print("[info] 未安装 modelscope SDK，尝试 pip install modelscope ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modelscope"])
    if r.returncode != 0:
        print("[warn] pip install modelscope 失败（离线节点？），改用 CLI 兜底重试")
    try:
        importlib.import_module("modelscope")
        return True
    except ImportError:
        return False


def download_sdk(model_id, dest, revision):
    from modelscope import snapshot_download  # noqa: PLC0415
    path = snapshot_download(model_id=model_id, local_dir=dest,
                             revision=revision if revision else None)
    print(f"[done] SDK 下载完成：{path}")
    return path


def download_cli(model_id, dest, revision):
    cmd = [sys.executable, "-m", "modelscope", "download",
           "--model", model_id, "--local_dir", dest]
    if revision:
        cmd += ["--revision", revision]
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return dest


def main():
    ap = argparse.ArgumentParser(description="ModelScope 模型下载")
    ap.add_argument("model", help="模型名或 namespace/name")
    ap.add_argument("--dest", default=DEFAULT_DEST,
                    help=f"下载根目录（默认 {DEFAULT_DEST}）")
    ap.add_argument("--revision", default="", help="分支/tag（默认 master）")
    ap.add_argument("--force", action="store_true", help="目录已存在也重下")
    args = ap.parse_args()

    model_id = resolve_model(args.model)
    name = model_id.rstrip("/").split("/")[-1]
    dest = os.path.join(args.dest, name)

    if has_weights(dest) and not args.force:
        print(f"[skip] {dest} 已含权重文件，视为下载完成（--force 可重下）")
        return

    os.makedirs(dest, exist_ok=True)
    print(f"[info] 模型：{model_id}\n[info] 目标：{dest}")
    if ensure_sdk():
        download_sdk(model_id, dest, args.revision)
    else:
        download_cli(model_id, dest, args.revision)

    if not has_weights(dest):
        raise SystemExit(f"[ERROR] {dest} 下完没有权重文件（网络中断？），重跑或 --force")
    print(f"[OK] 权重就位：{dest}")
    print(f"[下一步] MODEL_PATH={dest} SERVED_NAME={name} "
          f"nohup bash start_template.sh > serve.log 2>&1 &")


if __name__ == "__main__":
    main()
