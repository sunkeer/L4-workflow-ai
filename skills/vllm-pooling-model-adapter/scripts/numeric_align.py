#!/usr/bin/env python3
"""四层数值对齐：vLLM 池化服务 vs transformers 参考实现。

不要一上来就比最终向量然后瞎猜。按 L1→L4 逐层收敛，每层把问题锁在一个环节：

  L1 模型 & 池化   原始文本（不走 chat template）：vLLM `input` vs reference last-token hidden
  L2 prompt 构造   vLLM /tokenize 的 token id vs reference apply_chat_template 的 token id
  L3 池化位置      reference 侧扫最后 N 个位置，看哪个位置和 vLLM 输出最像
  L4 误差量级      归一化后最大分量差

判据：余弦 >= 0.999 视为数值等价。

用法：
  python3 numeric_align.py --path <模型目录> --base http://127.0.0.1:8000 \
      --model <SERVED_NAME> [--device cuda:1] [--doc-instruction "..."]

前提：服务已起，且 requests 里带了 add_generation_prompt=true（本脚本已带）。
"""
import argparse
import json
import sys
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.load(r)


def l2(v):
    v = [float(x) for x in v]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def cos(a, b):
    a, b = l2(a), l2(b)
    return sum(x * y for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="模型目录（reference 用）")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="服务里的 model 名")
    ap.add_argument("--device", default="cuda:1", help="reference 跑在哪张卡，避免抢服务卡")
    ap.add_argument("--doc-instruction", default="Represent the candidate content for retrieval.")
    ap.add_argument("--text", default="A dog running on the green grass.")
    ap.add_argument("--scan", type=int, default=6, help="L3 扫描最后几个位置")
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoTokenizer

    # 优先用 AutoModel 的生成类当 reference：结构与权重同构，只是多一个 lm_head
    from transformers import AutoModelForCausalLM

    print("== 加载 reference")
    tok = AutoTokenizer.from_pretrained(a.path)
    proc = AutoProcessor.from_pretrained(a.path) if True else None
    mdl = AutoModelForCausalLM.from_pretrained(
        a.path, dtype=torch.bfloat16, device_map=a.device, trust_remote_code=True).eval()
    print(f"   {type(mdl).__name__} on {a.device}")

    msgs = [{"role": "system", "content": a.doc_instruction},
            {"role": "user", "content": a.text}]

    def ref_last(ids):
        with torch.no_grad():
            out = mdl(input_ids=torch.tensor([ids], device=a.device),
                      output_hidden_states=True)
        return out.hidden_states[-1][0, -1].float().cpu().tolist(), \
               out.hidden_states[-1][0].float().cpu()

    # ---------- L1：原始文本，不走 chat template ----------
    print("\n== L1 模型 & 池化（原始文本）")
    raw = f"{a.doc_instruction}\n\n{a.text}"
    ids_raw = tok(raw, return_tensors="pt").input_ids[0].tolist()
    v_ref_raw, _ = ref_last(ids_raw)
    v_vllm_raw = post(f"{a.base}/v1/embeddings",
                      {"model": a.model, "input": raw,
                       "encoding_format": "float"})["data"][0]["embedding"]
    c1 = cos(v_vllm_raw, v_ref_raw)
    print(f"   dim vllm={len(v_vllm_raw)} ref={len(v_ref_raw)}   cosine={c1:.6f}")
    if len(v_vllm_raw) != len(v_ref_raw):
        print("   [FAIL] 维度不同 → 大概率命中了别的 pooler（late-interaction 的 token_embed？）")
    elif c1 < 0.999:
        print("   [FAIL] L1 不过 → 权重加载 / dtype / 池化位置本身有问题，先解决这层")
    else:
        print("   [OK]")

    # ---------- L2：prompt 构造 ----------
    print("\n== L2 prompt 构造（token id 比对）")
    text_t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids_ref = tok(text_t, return_tensors="pt").input_ids[0].tolist()
    try:
        t = post(f"{a.base}/tokenize", {"model": a.model, "messages": msgs})
        ids_vllm = t["tokens"]
        same = ids_vllm == ids_ref
        print(f"   len vllm={len(ids_vllm)} ref={len(ids_ref)}  完全一致={same}")
        if not same:
            for i, (x, y) in enumerate(zip(ids_vllm, ids_ref)):
                if x != y:
                    print(f"   首个分歧 @{i}: vllm={x} ref={y}")
                    print(f"   vllm 片段 {ids_vllm[max(0,i-3):i+3]}")
                    print(f"   ref  片段 {ids_ref[max(0,i-3):i+3]}")
                    break
            print("   → 查 chat template 文件 / add_generation_prompt / 系统消息处理")
    except Exception as e:
        print(f"   /tokenize 不可用: {type(e).__name__} {str(e)[:120]}")
        ids_vllm = ids_ref

    # ---------- L3：池化位置 ----------
    print("\n== L3 池化位置（reference 扫最后几个位置）")
    v_vllm_chat = post(f"{a.base}/v1/embeddings",
                       {"model": a.model, "messages": msgs,
                        "add_generation_prompt": True,
                        "encoding_format": "float"})["data"][0]["embedding"]
    _, hs = ref_last(ids_ref)
    best = (-1, -1.0)
    for off in range(1, a.scan + 1):
        idx = -off
        c = cos(v_vllm_chat, hs[idx].tolist())
        print(f"   pos -{off} (token {ids_ref[idx]:6d})  cosine={c:.4f}")
        if c > best[1]:
            best = (off, c)
    print(f"   → 最匹配位置 -{best[0]}，cosine={best[1]:.4f}")
    if best[0] == 1 and best[1] >= 0.999:
        print("   [OK] 池化在最后一个 token，与 add_generation_prompt=True 一致")
    elif best[1] >= 0.999:
        print(f"   [WARN] 池化落在倒数第 {best[0]} 个 token ——")
        print("          若请求没带 add_generation_prompt=true，这就是默认 False 的表现，加上即可")
    else:
        print("   [FAIL] 任何位置都对不上 → 回到 L1 查模型/权重")

    # ---------- L4：误差量级 ----------
    print("\n== L4 误差量级")
    c = cos(v_vllm_chat, hs[-1].tolist())
    md = max(abs(x - y) for x, y in zip(l2(v_vllm_chat), l2(hs[-1].tolist())))
    print(f"   cosine={c:.6f}  归一化后最大分量差={md:.2e}")
    print("   参考：1e-3 量级 = bf16 正常误差；1e-1 量级 = 结构性差异，别当精度问题放过")

    ok = c1 >= 0.999 and c >= 0.999
    print("\nALIGN", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
