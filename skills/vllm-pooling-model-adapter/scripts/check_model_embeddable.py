#!/usr/bin/env python3
"""判型：这个模型能不能走「vLLM pooling runner + 适配模块」这条路。

用法：
  python3 check_model_embeddable.py <本地模型目录 | ModelScope/HF 的 repo id>

输出四项结论：
  1) architectures 是否已被 vLLM 支持
  2) 是否含自定义建模代码（modeling_*.py）—— 有的话基本不能直接复用内置类
  3) 池化语义线索（pooling / normalize / 指令前缀）
  4) 该走哪条路：直接用 / 加适配模块 / 另起方案

只读不写，不下载权重（在线 id 只取 config.json 等小文件）。
"""
import json
import os
import sys


def load_config(src):
    if os.path.isdir(src):
        p = os.path.join(src, "config.json")
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8")), src
        return None, src
    # repo id：用 ModelScope API 取 config.json
    import urllib.request

    url = f"https://www.modelscope.cn/api/v1/models/{src}/repo?Revision=master&FilePath=config.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode("utf-8")), None
    except Exception as e:
        print(f"[warn] 取不到 {src} 的 config.json: {e}")
        return None, None


def vllm_archs():
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        return set(ModelRegistry.get_supported_archs())
    except Exception as e:
        print(f"[warn] 读不到 vLLM registry（没装 vllm？）: {e}")
        return set()


def pooling_clues(d):
    """从常见的池化配置文件名里找语义线索。"""
    clues = {}
    cands = {
        "mmemb_model.json": ("pooling", "normalize", "drop_lm_head", "hidden_states_from"),
        "sentence_bert_config.json": ("max_seq_length", "do_lower_case"),
        "modules.json": ("type",),
        "1_Pooling/config.json": ("pooling_mode_cls_token", "pooling_mode_lasttoken",
                                  "pooling_mode_mean_tokens"),
    }
    for fn, keys in cands.items():
        p = os.path.join(d, fn)
        if os.path.exists(p):
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            clues[fn] = {k: j[k] for k in keys if k in j}
            if fn == "modules.json":
                clues[fn] = [m.get("type") for m in j]
    return clues


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    cfg, local = load_config(src)
    if cfg is None:
        sys.exit("拿不到 config.json，无法判型")

    archs = cfg.get("architectures") or []
    print(f"architectures: {archs}")
    print(f"model_type:    {cfg.get('model_type')}")
    print(f"dtype:         {cfg.get('dtype') or cfg.get('torch_dtype')}")

    sup = vllm_archs()
    known = [a for a in archs if a in sup]
    print(f"\n[vLLM registry] 共 {len(sup)} 个 arch")
    for a in archs:
        print(f"  {a}: {'✅ 已支持' if a in sup else '❌ 未支持'}")

    has_code = False
    if local:
        code = [f for f in os.listdir(local) if f.startswith("modeling_") and f.endswith(".py")]
        has_code = bool(code)
        print(f"\n自定义建模代码: {'有 → ' + ', '.join(code) if code else '无'}")

    cl = pooling_clues(local) if local else {}
    if cl:
        print("\n池化线索:")
        for fn, v in cl.items():
            print(f"  {fn}: {v}")

    print("\n=== 结论 ===")
    if known and not has_code:
        print("直接用：arch 已被 vLLM 支持，按生成/池化 runner 启动即可，无需适配模块。")
    elif not has_code and archs:
        print("走适配模块：arch 名未被注册，但无自定义建模代码。")
        print(f"  1) 找同家族生成类：config.model_type={cfg.get('model_type')} 对应的 vLLM 实现")
        print("  2) 复制 assets/adapter_template.py，改 ARCH_NAME / 基类 / 池化方式")
        print("  3) sitecustomize.py + PYTHONPATH 让 spawn 子进程也注册")
        print("  4) --runner pooling --pooler-config '{\"seq_pooling_type\": \"LAST\"}'")
    else:
        print("另起方案：含自定义建模代码，vLLM 内置类大概率对不上。")
        print("  可选：逐个模块把 modeling_*.py 的 forward 对齐进 vLLM 实现，或改用 transformers/FastAPI 直出。")


if __name__ == "__main__":
    main()
