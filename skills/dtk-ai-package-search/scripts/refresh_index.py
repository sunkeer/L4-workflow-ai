#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refresh_index.py —— 从光源(SourceFind)镜像站重建 DTK AI 生态包索引。

数据源: https://download.sourcefind.cn:65024/4/main/
真实列表 API (逆向所得):
    GET /api-static/file/ListFile?CategoryID=4&Path=<路径>
    注意: 参数是大写 P 的 `Path`，小写 `path` 会被服务端静默忽略并总是返回根目录。
返回条目字段: Name / IsDir("yes"/"no") / Path / DownloadPath / Size / ModTime
直链 = https://download.sourcefind.cn:65024 + DownloadPath

用法:
    python3 refresh_index.py                      # 全量重建 -> ../data/packages.json
    python3 refresh_index.py -o /tmp/idx.json     # 指定输出
    python3 refresh_index.py --only deepspeed,vllm # 只刷新部分包(调试用)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

ORIGIN = "https://download.sourcefind.cn:65024"
API = ORIGIN + "/api-static/file/ListFile"
CATEGORY_ID = "4"

# 该站证书链本身是完整的（已实测默认校验可通过），所以默认走正常校验。
# 只有在老旧离线节点 CA 根证书过期时才需要 --insecure 降级。
_SSL_CTX: ssl.SSLContext | None = None


def _ctx(insecure: bool) -> ssl.SSLContext:
    global _SSL_CTX
    if _SSL_CTX is None:
        _SSL_CTX = ssl.create_default_context()
        if insecure:
            _SSL_CTX.check_hostname = False
            _SSL_CTX.verify_mode = ssl.CERT_NONE
    return _SSL_CTX

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, os.pardir, "data", "packages.json")


def list_dir(path: str, retries: int = 3, insecure: bool = False) -> list[dict]:
    """列出镜像站某个目录。path 形如 '/' 或 '/deepspeed/DAS1.8'。"""
    qs = urllib.parse.urlencode({"CategoryID": CATEGORY_ID, "Path": path})
    url = f"{API}?{qs}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40, context=_ctx(insecure)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            code = payload.get("code")
            # 该站成功时返回 code="success"，也兼容 0/200 这类常见约定
            if code not in (0, 200, "0", "200", "success", "ok", None):
                raise RuntimeError(f"API code={code} msg={payload.get('msg')}")
            return payload.get("data") or []
        except Exception as exc:  # noqa: BLE001 - 网络抖动统一重试
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"list_dir({path}) 失败: {last}")


# ---------------------------------------------------------------- 文件名解析

# 例: deepspeed-0.18.2+das.opt1.dtk2604.torch290-cp312-cp312-manylinux_2_28_x86_64.whl
RE_DTK = re.compile(r"dtk(\d{4,6})", re.I)
RE_TORCH = re.compile(r"torch(\d{2,4})", re.I)
RE_PY = re.compile(r"(?:^|[-_.])(cp\d{2,3}|py3)(?=[-_.]|$)", re.I)
RE_DASOPT = re.compile(r"das(?:\.(opt\d+))?", re.I)
RE_VER = re.compile(r"^[A-Za-z0-9_.]+?-(\d[\w.]*?)(?:\+|-)")


def parse_filename(name: str) -> dict:
    """从 whl/tar.gz/deb 文件名里抽取版本、dtk、torch、python 等元数据。"""
    meta: dict[str, object] = {}

    m = RE_VER.match(name)
    if m:
        meta["version"] = m.group(1)

    m = RE_DTK.search(name)
    if m:
        meta["dtk"] = "dtk" + m.group(1)

    m = RE_TORCH.search(name)
    if m:
        raw = m.group(1)
        meta["torch_tag"] = "torch" + raw
        # torch290 -> 2.9.0 ; torch271 -> 2.7.1 ; torch21 -> 2.1
        meta["torch"] = ".".join(raw) if len(raw) <= 3 else f"{raw[0]}.{raw[1:-1]}.{raw[-1]}"

    pys = [t.lower() for t in RE_PY.findall(name)]
    if pys:
        uniq: list[str] = []
        for t in pys:
            if t not in uniq:
                uniq.append(t)
        meta["python_tags"] = uniq
        cps = [t for t in uniq if t.startswith("cp")]
        if cps:
            cp = cps[0][2:]
            meta["python"] = f"{cp[0]}.{cp[1:]}"
        elif "py3" in uniq:
            meta["python"] = "py3"

    m = RE_DASOPT.search(name)
    if m and m.group(1):
        meta["das_opt"] = m.group(1).lower()

    low = name.lower()
    for ext in (".whl", ".tar.gz", ".tgz", ".deb", ".rpm", ".zip", ".pdf", ".run", ".tar"):
        if low.endswith(ext):
            meta["ext"] = ext
            break
    else:
        meta["ext"] = os.path.splitext(name)[1] or ""

    if "manylinux" in low:
        meta["platform"] = "manylinux"
    elif "none-any" in low:
        meta["platform"] = "any"

    return meta


def walk(path: str, depth: int, max_depth: int, out: list[dict], seen: set[str],
         quiet: bool, insecure: bool = False):
    """递归遍历目录树，把文件收集进 out。"""
    if depth > max_depth or path in seen:
        return
    seen.add(path)
    try:
        entries = list_dir(path, insecure=insecure)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] {path}: {exc}", file=sys.stderr)
        return

    for e in entries:
        name = e.get("Name") or ""
        if not name:
            continue
        is_dir = str(e.get("IsDir", "")).lower() in ("yes", "true", "1")
        child = (path.rstrip("/") + "/" + name) if path != "/" else "/" + name
        if is_dir:
            walk(child, depth + 1, max_depth, out, seen, quiet, insecure)
        else:
            dl = e.get("DownloadPath") or ""
            if not dl:
                continue
            rel = child.lstrip("/").split("/")
            rec = {
                "package": rel[0] if rel else "",
                "dir": "/".join(rel[1:-1]),
                "filename": name,
                "path": child,
                "url": ORIGIN + dl if dl.startswith("/") else dl,
                "size": e.get("Size") or e.get("SizeStr") or "",
            }
            das = next((p for p in rel[1:-1] if p.upper().startswith("DAS")), "")
            rec["das"] = das
            rec.update(parse_filename(name))
            out.append(rec)
            if not quiet:
                print(f"    + {child}")


def main() -> int:
    ap = argparse.ArgumentParser(description="重建 DTK AI 生态包索引")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="输出 JSON 路径")
    ap.add_argument("--only", default="", help="逗号分隔，只刷新这些包（调试用）")
    ap.add_argument("--max-depth", type=int, default=4, help="最大递归深度")
    ap.add_argument("-q", "--quiet", action="store_true", help="不逐条打印文件")
    ap.add_argument("--insecure", action="store_true",
                    help="跳过 TLS 证书校验（仅在老旧离线节点 CA 过期时使用）")
    args = ap.parse_args()

    print(f"[1/3] 列出根目录 {ORIGIN}/4/main/ ...")
    root = list_dir("/", insecure=args.insecure)
    top_dirs = [e["Name"] for e in root if str(e.get("IsDir", "")).lower() in ("yes", "true", "1")]
    root_files = [e for e in root if str(e.get("IsDir", "")).lower() not in ("yes", "true", "1")]
    print(f"      顶层目录 {len(top_dirs)} 个，根文件 {len(root_files)} 个")

    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        top_dirs = [d for d in top_dirs if d in want]
        print(f"      --only 过滤后: {top_dirs}")

    records: list[dict] = []
    seen: set[str] = set()

    # 根目录下的散装文件（技术文档 PDF 等）
    for e in root_files:
        dl = e.get("DownloadPath") or ""
        if not dl:
            continue
        rec = {
            "package": "_root",
            "dir": "",
            "das": "",
            "filename": e["Name"],
            "path": "/" + e["Name"],
            "url": ORIGIN + dl if dl.startswith("/") else dl,
            "size": e.get("Size") or "",
        }
        rec.update(parse_filename(e["Name"]))
        records.append(rec)

    print(f"[2/3] 递归遍历 {len(top_dirs)} 个包 ...")
    for i, d in enumerate(top_dirs, 1):
        print(f"  ({i}/{len(top_dirs)}) {d}")
        walk("/" + d, 1, args.max_depth, records, seen, args.quiet, args.insecure)

    pkgs = sorted({r["package"] for r in records if r["package"] != "_root"})
    dases = sorted({r["das"] for r in records if r.get("das")})
    dtks = sorted({str(r["dtk"]) for r in records if r.get("dtk")})

    index = {
        "source": ORIGIN + "/4/main/",
        "api": API + "?CategoryID=4&Path=<PATH>",
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "stats": {
            "packages": len(pkgs),
            "files": len(records),
            "das_versions": dases,
            "dtk_versions": dtks,
        },
        "packages": pkgs,
        "files": sorted(records, key=lambda r: (r["package"], r.get("das", ""), r["filename"])),
    }

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    print(f"[3/3] 已写入 {out}")
    print(f"      包 {len(pkgs)} 个 / 文件 {len(records)} 条")
    print(f"      DAS: {', '.join(dases)}")
    print(f"      DTK: {', '.join(dtks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
