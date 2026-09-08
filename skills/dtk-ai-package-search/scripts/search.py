#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search.py —— 检索海光 DCU (DTK/DAS) AI 生态包，直接产出 whl 直链与 pip 安装命令。

索引数据: ../data/packages.json（由 refresh_index.py 从光源镜像站重建）

常用姿势
--------
# 1) 模糊搜包名
python3 search.py deepspeed

# 2) 锁定环境：DTK 26.04 + Python 3.10（最常用）
python3 search.py deepspeed --dtk 2604 --py 3.10

# 3) 直接读当前机器环境自动过滤（在 DCU 节点上跑）
python3 search.py vllm --from-env

# 4) 只要每个包的最新 DAS 版本
python3 search.py torch --latest

# 5) 出 pip 命令（可直接粘贴执行）
python3 search.py flash_attn --dtk 2604 --py 3.10 -f pip

# 6) 看有哪些包 / 哪些版本
python3 search.py --list-packages
python3 search.py --list-versions
python3 search.py --info deepspeed
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX = os.path.join(HERE, os.pardir, "data", "packages.json")


# --------------------------------------------------------------- 版本排序

def das_key(s: str) -> list[int]:
    """DAS1.8.4 -> [1,8,4]，用于版本排序。无法解析的排最前。"""
    nums = re.findall(r"\d+", s or "")
    return [int(n) for n in nums] or [-1]


def dtk_key(s: str) -> list[int]:
    """dtk25041 -> [25041]。dtk2604 与 dtk25042 之间按数值不可直接比，
    故先比前 4 位（年月），再比余下的修订位。"""
    m = re.search(r"(\d{4})(\d*)", s or "")
    if not m:
        return [-1]
    return [int(m.group(1)), int(m.group(2) or 0)]


# --------------------------------------------------------------- 索引加载

def load_index(path: str) -> dict:
    p = os.path.abspath(path)
    if not os.path.exists(p):
        sys.exit(f"[错误] 索引不存在: {p}\n  先执行: python3 {os.path.join(HERE, 'refresh_index.py')}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------- 本机环境探测

def detect_env() -> dict:
    """探测当前机器的 DTK 版本与 Python 版本，用于 --from-env 自动过滤。"""
    env: dict[str, str] = {}
    env["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"

    # 1) DTK 版本：优先读 /opt/dtk/.info 或 VERSION，再退到环境变量与软链名
    for cand in ("/opt/dtk/.info/version", "/opt/dtk/VERSION", "/opt/dtk/.version"):
        try:
            with open(cand, encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            m = re.search(r"(\d{2})\.(\d{2})(?:\.(\d+))?", txt)
            if m:
                env["dtk"] = "dtk" + m.group(1) + m.group(2) + (m.group(3) or "")
                env["dtk_raw"] = txt.strip()[:120]
                break
        except OSError:
            continue

    if "dtk" not in env:
        # /opt/dtk 常是指向 /opt/dtk-25.04 之类的软链
        try:
            real = os.path.realpath("/opt/dtk")
            m = re.search(r"(\d{2})\.(\d{2})\.?(\d*)", real)
            if m:
                env["dtk"] = "dtk" + m.group(1) + m.group(2) + (m.group(3) or "")
                env["dtk_raw"] = real
        except OSError:
            pass

    if "dtk" not in env:
        for var in ("DTK_VERSION", "ROCM_VERSION", "HIP_VERSION"):
            v = os.environ.get(var)
            if v:
                m = re.search(r"(\d{2})\.?(\d{2})\.?(\d*)", v)
                if m:
                    env["dtk"] = "dtk" + m.group(1) + m.group(2) + (m.group(3) or "")
                    env["dtk_raw"] = f"{var}={v}"
                    break

    # 2) torch 版本（用于 torch_tag 过滤）
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import torch;print(torch.__version__)"],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0 and out.stdout.strip():
            env["torch"] = out.stdout.strip().split("+")[0]
    except Exception:  # noqa: BLE001 - torch 没装是正常情况
        pass

    return env


# --------------------------------------------------------------- 过滤

def norm_dtk(v: str) -> str:
    """'26.04' / '2604' / 'dtk2604' 统一成 'dtk2604'。"""
    v = (v or "").strip().lower()
    if not v:
        return ""
    digits = re.sub(r"[^\d]", "", v)
    return "dtk" + digits if digits else ""


def norm_das(v: str) -> str:
    """'1.8' / 'das1.8' / 'DAS1.8' 统一成 'DAS1.8'。"""
    v = (v or "").strip()
    if not v:
        return ""
    v = re.sub(r"(?i)^das", "", v).strip()
    return "DAS" + v


def norm_py(v: str) -> str:
    """'cp310' / '310' / '3.10' 统一成 '3.10'。"""
    v = (v or "").strip().lower()
    if not v:
        return ""
    if v in ("py3", "any"):
        return v
    d = re.sub(r"[^\d]", "", v)
    if not d:
        return ""
    return f"{d[0]}.{d[1:]}" if len(d) > 1 else d


def filter_files(files: list[dict], args) -> list[dict]:
    out = files

    if args.query:
        # 支持多个关键词（AND），任一命中包名/文件名即可
        terms = [t.lower() for t in args.query]

        def hit(r: dict) -> bool:
            hay = (r["package"] + " " + r["filename"]).lower()
            return all(t in hay for t in terms)

        # 优先精确包名命中，没有再退到模糊
        exact = [r for r in out if r["package"].lower() in terms]
        out = exact if exact else [r for r in out if hit(r)]

    if args.package:
        want = args.package.lower()
        out = [r for r in out if r["package"].lower() == want]

    if args.dtk:
        want = norm_dtk(args.dtk)
        out = [r for r in out if str(r.get("dtk", "")).lower() == want]

    if args.das:
        want = norm_das(args.das).lower()
        out = [r for r in out if str(r.get("das", "")).lower() == want]

    if args.py:
        want = norm_py(args.py)
        # py3(纯 python 包) 对任何 Python 版本都可用，不应被过滤掉
        out = [r for r in out if r.get("python") in (want, "py3")]

    if args.torch:
        want = re.sub(r"[^\d]", "", args.torch)
        out = [r for r in out if want and re.sub(r"[^\d]", "", str(r.get("torch_tag", ""))) == want]

    if args.ext:
        want = args.ext if args.ext.startswith(".") else "." + args.ext
        out = [r for r in out if r.get("ext") == want]

    if args.whl:
        out = [r for r in out if r.get("ext") == ".whl"]

    if args.latest:
        # 每个包只保留 DAS 最高的那一档
        best: dict[str, list[int]] = {}
        for r in out:
            k = r["package"]
            cur = das_key(r.get("das", ""))
            if k not in best or cur > best[k]:
                best[k] = cur
        out = [r for r in out if das_key(r.get("das", "")) == best[r["package"]]]

    return sorted(
        out,
        key=lambda r: (r["package"], das_key(r.get("das", "")), r.get("version", ""), r["filename"]),
    )


# --------------------------------------------------------------- 输出

def print_table(rows: list[dict], limit: int) -> None:
    if not rows:
        print("(无匹配结果)")
        return
    shown = rows[:limit] if limit > 0 else rows
    cols = [
        ("包名", lambda r: r["package"], 20),
        ("DAS", lambda r: r.get("das", "-"), 9),
        ("版本", lambda r: str(r.get("version", "-")), 14),
        ("DTK", lambda r: str(r.get("dtk", "-")), 10),
        ("Py", lambda r: str(r.get("python", "-")), 5),
        ("torch", lambda r: str(r.get("torch", "-")), 7),
        ("大小", lambda r: str(r.get("size", "-")), 10),
    ]
    header = "  ".join(h.ljust(w) for h, _, w in cols)
    print(header)
    print("-" * len(header))
    for r in shown:
        print("  ".join(str(fn(r))[:w].ljust(w) for _, fn, w in cols))
        print(f"    {r['url']}")
    print("-" * len(header))
    total = len(rows)
    if limit > 0 and total > limit:
        print(f"共 {total} 条，已显示前 {limit} 条（--limit 0 显示全部）")
    else:
        print(f"共 {total} 条")


def print_pip(rows: list[dict], limit: int) -> None:
    shown = rows[:limit] if limit > 0 else rows
    if not shown:
        print("# (无匹配结果)")
        return
    print("# 有网环境：直接 pip install 直链")
    for r in shown:
        print(f"# {r['package']} {r.get('version','')} | {r.get('das','-')} | {r.get('dtk','-')} | py{r.get('python','-')} | {r.get('size','')}")
        print(f"pip install {r['url']}")
    if limit > 0 and len(rows) > limit:
        print(f"# ... 另有 {len(rows)-limit} 条未显示")


def print_wget(rows: list[dict], limit: int) -> None:
    shown = rows[:limit] if limit > 0 else rows
    if not shown:
        print("# (无匹配结果)")
        return
    print("# 无网环境：先在有网机器下载，再拷到 DCU 节点 pip install <本地文件>")
    for r in shown:
        print(f"wget --no-check-certificate '{r['url']}'")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="检索海光 DCU (DTK/DAS) AI 生态包直链",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("query", nargs="*", help="搜索关键词（包名或文件名片段，多个词是 AND）")
    ap.add_argument("-p", "--package", help="精确包名")
    ap.add_argument("--dtk", help="DTK 版本，如 2604 / 26.04 / dtk2604")
    ap.add_argument("--das", help="DAS 版本，如 1.8 / DAS1.8")
    ap.add_argument("--py", help="Python 版本，如 3.10 / cp310")
    ap.add_argument("--torch", help="torch 版本，如 2.5.1 / torch251")
    ap.add_argument("--ext", help="文件后缀过滤，如 whl / tar.gz / pdf")
    ap.add_argument("--whl", action="store_true", help="只要 .whl")
    ap.add_argument("--latest", action="store_true", help="每个包只保留最高 DAS 版本")
    ap.add_argument("--from-env", action="store_true", help="读取本机 DTK/Python 自动过滤")
    ap.add_argument("-f", "--format", default="table",
                    choices=["table", "pip", "url", "wget", "json"], help="输出格式")
    ap.add_argument("-n", "--limit", type=int, default=30, help="最多显示条数，0=全部")
    ap.add_argument("--index", default=DEFAULT_INDEX, help="索引文件路径")
    ap.add_argument("--list-packages", action="store_true", help="列出所有包名")
    ap.add_argument("--list-versions", action="store_true", help="列出 DAS/DTK 版本与对应关系")
    ap.add_argument("--info", help="查看某个包的版本矩阵")
    args = ap.parse_args()

    idx = load_index(args.index)
    files = idx["files"]

    # ---- 元信息类查询 ----
    if args.list_packages:
        pkgs = idx["packages"]
        print(f"索引时间: {idx.get('generated_at')}   共 {len(pkgs)} 个包 / {len(files)} 个文件\n")
        for i in range(0, len(pkgs), 4):
            print("  " + "".join(p.ljust(24) for p in pkgs[i:i + 4]))
        return 0

    if args.list_versions:
        import collections
        m: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for r in files:
            if r.get("das") and r.get("dtk"):
                m[r["das"]][str(r["dtk"])] += 1
        print("DAS 版本 → DTK 版本对应关系（括号内为文件数）\n")
        print("  DAS 版本    DTK 版本")
        print("  " + "-" * 52)
        for das in sorted(m, key=das_key):
            main_dtk = ", ".join(f"{k}({v})" for k, v in m[das].most_common())
            print(f"  {das:<11} {main_dtk}")
        print("\n注: DAS1.4 不存在（官方跳版）；DAS1.8.3/1.8.4 仅含 XCompute/XSystems 工具。")
        return 0

    if args.info:
        import collections
        want = args.info.lower()
        rows = [r for r in files if r["package"].lower() == want]
        if not rows:
            cand = [p for p in idx["packages"] if want in p.lower()]
            print(f"未找到包 '{args.info}'。" + (f" 相近: {', '.join(cand[:10])}" if cand else ""))
            return 1
        print(f"包名: {rows[0]['package']}   文件数: {len(rows)}\n")
        grp: dict[str, list[dict]] = collections.defaultdict(list)
        for r in rows:
            grp[r.get("das") or r.get("dir") or "(根目录)"].append(r)
        print(f"  {'DAS/目录':<12} {'版本':<16} {'DTK':<11} {'Python':<22} 文件数")
        print("  " + "-" * 72)
        for k in sorted(grp, key=das_key):
            g = grp[k]
            vers = sorted({str(r.get("version", "-")) for r in g})
            dtks = sorted({str(r.get("dtk", "-")) for r in g}, key=dtk_key)
            pys = sorted({str(r.get("python", "-")) for r in g})
            print(f"  {k:<12} {','.join(vers)[:15]:<16} {','.join(dtks)[:10]:<11} {','.join(pys)[:21]:<22} {len(g)}")
        return 0

    # ---- --from-env: 用本机环境补全过滤条件 ----
    if args.from_env:
        env = detect_env()
        print("[本机环境探测]")
        print(f"  Python : {env.get('python','?')}")
        print(f"  DTK    : {env.get('dtk','未探测到')}" + (f"   ({env['dtk_raw']})" if env.get("dtk_raw") else ""))
        print(f"  torch  : {env.get('torch','未安装')}")
        if not args.py and env.get("python"):
            args.py = env["python"]
        if not args.dtk and env.get("dtk"):
            args.dtk = env["dtk"]
        if not args.dtk:
            print("  [提示] 没探测到 DTK 版本，本次不按 DTK 过滤；可手动加 --dtk 2604")
        print()

    rows = filter_files(files, args)

    # 有过滤条件但零结果时，给出可用的邻近选项，避免用户干瞪眼。
    # 做法：只保留名称匹配、去掉全部版本过滤，再按包分组报告实际可用组合。
    if not rows and (args.query or args.package):
        import collections

        class _NameOnly:
            query = args.query
            package = args.package
            dtk = das = py = torch = ext = None
            whl = latest = False

        same = filter_files(files, _NameOnly())
        if same:
            grp: dict[str, list[dict]] = collections.defaultdict(list)
            for r in same:
                grp[r["package"]].append(r)
            print("(当前过滤条件下无结果) 名称匹配的包实际可用的组合：\n")
            for pkg in sorted(grp)[:8]:
                g = grp[pkg]
                avail_das = sorted({r.get("das", "") for r in g if r.get("das")}, key=das_key)
                avail_dtk = sorted({str(r.get("dtk", "")) for r in g if r.get("dtk")}, key=dtk_key)
                avail_py = sorted({str(r.get("python", "")) for r in g if r.get("python")})
                print(f"  {pkg}")
                print(f"    DAS    : {', '.join(avail_das) or '-'}")
                print(f"    DTK    : {', '.join(avail_dtk) or '-'}")
                print(f"    Python : {', '.join(avail_py) or '-'}")
            if len(grp) > 8:
                print(f"  ... 另有 {len(grp)-8} 个匹配包未显示")
            return 1

    if args.format == "json":
        print(json.dumps(rows if args.limit <= 0 else rows[:args.limit], ensure_ascii=False, indent=2))
    elif args.format == "url":
        for r in (rows if args.limit <= 0 else rows[:args.limit]):
            print(r["url"])
    elif args.format == "pip":
        print_pip(rows, args.limit)
    elif args.format == "wget":
        print_wget(rows, args.limit)
    else:
        print_table(rows, args.limit)

    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
