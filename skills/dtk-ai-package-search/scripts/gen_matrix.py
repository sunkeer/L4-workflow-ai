#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_matrix.py —— 由 data/packages.json 生成 references/version_matrix.md。
索引刷新后重跑一次即可同步文档，避免手写矩阵与数据脱节。

用法: python3 gen_matrix.py
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, os.pardir, "data", "packages.json")
OUT = os.path.join(HERE, os.pardir, "references", "version_matrix.md")

DAS_ORDER = ["DAS1.0", "DAS1.1", "DAS1.1.1", "DAS1.2", "DAS1.3",
             "DAS1.5", "DAS1.6", "DAS1.7", "DAS1.8"]


def das_key(s: str) -> list[int]:
    nums = re.findall(r"\d+", s or "")
    return [int(n) for n in nums] or [-1]


def dtk_key(s: str) -> list[int]:
    m = re.search(r"(\d{4})(\d*)", s or "")
    return [int(m.group(1)), int(m.group(2) or 0)] if m else [-1]


def main() -> int:
    with open(INDEX, encoding="utf-8") as f:
        idx = json.load(f)
    files = idx["files"]

    L: list[str] = []
    L.append("# DTK / DAS 版本矩阵")
    L.append("")
    L.append(f"> 本文件由 `scripts/gen_matrix.py` 从 `data/packages.json` 自动生成，请勿手改。")
    L.append(f"> 索引时间：{idx.get('generated_at')}　数据源：{idx.get('source')}")
    L.append(f"> 规模：**{idx['stats']['packages']} 个包 / {idx['stats']['files']} 个文件**")
    L.append("")

    # ---------- 1. DAS -> DTK ----------
    L.append("## 1. DAS 版本 ↔ DTK 版本对应关系")
    L.append("")
    L.append("挑包最关键的一步：先确定节点的 DTK 版本，再反查该用哪个 DAS 目录。")
    L.append("")
    m: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in files:
        if r.get("das") and r.get("dtk"):
            m[r["das"]][str(r["dtk"])] += 1
    L.append("| DAS 版本 | 主要 DTK 版本 | 其他 DTK | 文件数 |")
    L.append("|---|---|---|---|")
    for das in sorted(m, key=das_key):
        items = m[das].most_common()
        main = items[0][0]
        others = ", ".join(k for k, _ in items[1:]) or "—"
        L.append(f"| `{das}` | `{main}` | {others} | {sum(v for _, v in items)} |")
    L.append("")
    L.append("**注意事项**")
    L.append("")
    L.append("- **DAS1.4 不存在**，官方直接从 1.3 跳到 1.5，别在脚本里硬编码连续版本号。")
    L.append("- `DAS1.8.3` / `DAS1.8.4` 不是常规生态包目录，只放 XCompute / XSystems 工具。")
    L.append("- `DAS1.2` 里混有少量 `dtk24041` 文件、`DAS1.6` 里混有少量 `dtk2504` 文件，")
    L.append("  同一 DAS 目录内 DTK 不一定唯一 —— **务必按文件名里的 `dtkXXXX` 二次确认**。")
    L.append("- DTK 版本号写法：`dtk2604` = DTK 26.04，`dtk25041` = DTK 25.04.1（末位是修订号）。")
    L.append("")

    # ---------- 2. DAS -> Python ----------
    L.append("## 2. 各 DAS 版本支持的 Python")
    L.append("")
    p: dict[str, set[str]] = collections.defaultdict(set)
    for r in files:
        if r.get("das") and r.get("python"):
            p[r["das"]].add(str(r["python"]))
    L.append("| DAS 版本 | 支持的 Python |")
    L.append("|---|---|")
    for das in sorted(p, key=das_key):
        vs = sorted(v for v in p[das] if v != "py3")
        extra = "，另有 `py3`(纯 Python 包，任意版本可用)" if "py3" in p[das] else ""
        L.append(f"| `{das}` | {', '.join('`'+v+'`' for v in vs)}{extra} |")
    L.append("")
    L.append("> `cp312` 只在 `DAS1.8` 出现 —— 需要 Python 3.12 的话只能用 DTK 26.04。")
    L.append("")

    # ---------- 3. 包 x DAS 矩阵 ----------
    L.append("## 3. 包 × DAS 版本矩阵")
    L.append("")
    L.append("`✓` = 该 DAS 目录下有该包。")
    L.append("")
    pk: dict[str, set[str]] = collections.defaultdict(set)
    for r in files:
        if r["package"] != "_root" and r.get("das"):
            pk[r["package"]].add(r["das"])
    cols = [d for d in DAS_ORDER if any(d in v for v in pk.values())]
    L.append("| 包名 | " + " | ".join(c.replace("DAS", "") for c in cols) + " |")
    L.append("|---" * (len(cols) + 1) + "|")
    for name in sorted(pk):
        if name in ("DAS安装包", "XCompute", "XSystems"):
            continue
        cells = ["✓" if c in pk[name] else "" for c in cols]
        L.append(f"| `{name}` | " + " | ".join(cells) + " |")
    L.append("")

    # ---------- 4. 无 DAS 目录的包 ----------
    allp = {r["package"] for r in files if r["package"] != "_root"}
    nodas = sorted(allp - set(pk))
    if nodas:
        L.append("## 4. 没有 DAS 目录的包（走 `previous_release/` 或直接平铺）")
        L.append("")
        L.append("这些包不按 DAS 分目录，检索时**不要加 `--das`**，直接用包名搜或加 `--dtk` 过滤：")
        L.append("")
        L.append("".join(f"`{n}` " for n in nodas))
        L.append("")
        L.append("```bash")
        L.append("# 例：colossalai 没有 DAS 目录，直接搜")
        L.append("python3 scripts/search.py colossalai")
        L.append("```")
        L.append("")

    # ---------- 5. 高频包速查 ----------
    L.append("## 5. 高频包在 DAS1.8 (DTK 26.04) 下的可用版本")
    L.append("")
    hot = ["pytorch", "vision", "torchaudio", "deepspeed", "vllm", "sglang", "lmdeploy",
           "flash_attn", "apex", "triton", "xformers", "megatron", "transformer_engine",
           "bitsandbytes", "deepgemm", "deepep", "flash_mla", "mooncake", "onnxruntime"]
    L.append("| 包名 | 版本 | Python | torch |")
    L.append("|---|---|---|---|")
    for name in hot:
        g = [r for r in files if r["package"] == name and r.get("das") == "DAS1.8"]
        if not g:
            continue
        vers = sorted({str(r.get("version", "-")) for r in g})
        pys = sorted({str(r.get("python", "-")) for r in g})
        tor = sorted({str(r.get("torch", "")) for r in g if r.get("torch")})
        L.append(f"| `{name}` | {', '.join(vers)} | {', '.join(pys)} | {', '.join(tor) or '—'} |")
    L.append("")
    L.append("> 完整列表用 `python3 scripts/search.py --info <包名>` 查。")
    L.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(os.path.abspath(OUT), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"已生成 {os.path.abspath(OUT)}  ({len(L)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
