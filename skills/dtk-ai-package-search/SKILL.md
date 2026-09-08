---
name: dtk-ai-package-search
description: 检索海光 DCU（DTK / DAS）AI 生态包的官方 whl 直链并直接产出 pip 安装命令。当用户要在 DCU/曙光/海光节点上安装 pytorch、vllm、sglang、deepspeed、flash_attn、apex、triton、transformer_engine、megatron、vision、torchaudio、lmdeploy、bitsandbytes 等 AI 生态包，或问「DTK 26.04 该装哪个版本」「某包有没有 DAS1.8 版本」「DAS 和 DTK 版本怎么对应」「生态包下载地址在哪」，或遇到装完 import 报 undefined symbol / torch.cuda.is_available() 为 False 这类版本错配问题时使用。本地索引 67 个包 / 1268 个文件，离线可查，支持读取本机环境自动匹配。
---

# dtk-ai-package-search —— DCU AI 生态包检索

把光源（SourceFind）镜像站的 AI 生态包做成**本地可离线检索的索引**，输入包名直接拿到匹配当前
DTK / Python / torch 版本的 whl 直链与 `pip install` 命令。

- 数据源：<https://download.sourcefind.cn:65024/4/main/>
- 索引规模：**67 个包 / 1268 个文件**，覆盖 DAS1.0 ~ DAS1.8（含 `previous_release/` 历史版本）
- 配套飞书文档（人工浏览用）：<https://lhui08qsvi.feishu.cn/wiki/ApPYwC4VCieN70kGpr2cqkQCnOe>

## 核心认知：三个版本维度必须同时对上

DCU 生态包**不能按名字装**，必须同时匹配三项，否则典型症状是「装得上、`import` 就崩」：

| 维度 | 文件名体现 | 本机怎么查 |
|---|---|---|
| DTK | `dtk2604` = DTK 26.04 | `cat /opt/dtk/.info/version` 或 `ls -l /opt/dtk` |
| Python | `cp310` = Python 3.10 | `python3 -V` |
| torch | `torch271` = torch 2.7.1 | `python3 -c "import torch;print(torch.__version__)"` |

DAS 目录号是这三者的**打包批次编号**，与 DTK 一一对应（`DAS1.8` ↔ `dtk2604`，`DAS1.7` ↔ `dtk25042`……
完整表见 `references/version_matrix.md`）。**注意 DAS1.4 不存在**，官方从 1.3 直接跳到 1.5。

## 使用流程

### 第 1 步：确定目标环境

如果用户已经说了 DTK / Python 版本（如「DTK 26.04 + py3.10」），直接进第 2 步。

如果不知道，且**能在 DCU 节点上执行命令**，让脚本自己读：

```bash
python3 scripts/search.py <包名> --from-env -f pip
```

`--from-env` 会依次尝试 `/opt/dtk/.info/version` → `/opt/dtk` 软链 → `DTK_VERSION` 环境变量，
并读取当前 Python 与已装 torch 版本，自动套用为过滤条件。探测不到 DTK 时会明确提示并跳过该项过滤，
不会静默给出错误结果。

### 第 2 步：检索

```bash
# 锁定环境精确查（最常用）
python3 scripts/search.py deepspeed --dtk 2604 --py 3.10

# 版本写法都认：2604 / 26.04 / dtk2604 ；1.8 / DAS1.8 ；3.10 / cp310
python3 scripts/search.py vllm --das 1.8 --py cp312

# 同时约束 torch 版本（多 torch 分支的包必加，如 flash_attn / vision）
python3 scripts/search.py vision --dtk 2604 --py 3.10 --torch 2.7.1

# 只看每个包的最新 DAS 档
python3 scripts/search.py sglang --latest

# 摸不准包名时模糊搜（多关键词是 AND）
python3 scripts/search.py flash attn
```

**查不到时不要臆造链接**。脚本在零结果时会打印该包实际可用的 DAS / DTK / Python 组合，
按提示换档位重查即可。若该包确实没有目标 DAS 版本（如 `torch_scatter` 只到 DAS1.3），
如实告知用户并建议降档或源码编译。

### 第 3 步：出安装命令

```bash
# 有网：直接可粘贴执行的 pip 命令
python3 scripts/search.py deepspeed --dtk 2604 --py 3.10 -f pip

# 无网：先在有网机器下载，再拷到节点
python3 scripts/search.py deepspeed --dtk 2604 --py 3.10 -f wget

# 裸链接（喂给 aria2c / 写进别的脚本）
python3 scripts/search.py deepspeed --dtk 2604 --py 3.10 -f url
```

给用户安装命令时**务必带上 `--no-deps`** 或至少提示风险：不加的话 pip 会顺着依赖去 PyPI
拉 NVIDIA 版 torch，把节点上装好的 DCU 版覆盖掉，这是最高频的翻车点。

### 第 4 步：提示验证

安装后让用户确认三件事齐全，否则等于没装对：

```bash
python3 -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count())"
```

## 命令速查

| 目的 | 命令 |
|---|---|
| 列出全部包名 | `python3 scripts/search.py --list-packages` |
| DAS ↔ DTK 对应关系 | `python3 scripts/search.py --list-versions` |
| 某包的完整版本矩阵 | `python3 scripts/search.py --info deepspeed` |
| 精确包名（避免模糊命中） | `python3 scripts/search.py -p apex --dtk 2604` |
| 只要 whl | 加 `--whl`；要文档 PDF 加 `--ext pdf` |
| 输出全部结果 | 加 `-n 0`（默认只显示前 30 条） |
| 机器可读 | 加 `-f json` |

## 索引维护

索引是快照，不是实时的。检索结果与官网对不上时先刷新：

```bash
python3 scripts/refresh_index.py     # 全量重建（约 1~2 分钟）
python3 scripts/gen_matrix.py        # 同步重新生成 references/version_matrix.md
```

索引文件 `data/packages.json` 头部的 `generated_at` 就是数据时效。
老旧离线节点 CA 证书过期导致 TLS 失败时，加 `--insecure` 降级。

**镜像站真实接口**（逆向所得，重要）：

```
GET https://download.sourcefind.cn:65024/api-static/file/ListFile?CategoryID=4&Path=<路径>
```

- 参数是**大写 P 的 `Path`**；写成小写 `path` 会被服务端**静默忽略**并永远返回根目录列表 ——
  这个坑会让人误判成「API 不支持子目录」。
- 成功时 `code` 字段返回字符串 `"success"`，不是 `0`。
- 直链 = `https://download.sourcefind.cn:65024` + 返回条目的 `DownloadPath`。

## 参考文档

- `references/version_matrix.md` —— DAS↔DTK 对应表、各 DAS 支持的 Python、包×DAS 矩阵、
  高频包在 DAS1.8 下的可用版本。**由 `gen_matrix.py` 自动生成，不要手改。**
- `references/install_guide.md` —— 有网/无网安装、安装顺序、pytorch 与 vision/torchaudio
  配套版本表、装完验证脚本、7 类常见坑及处理。

## 文件结构

```
dtk-ai-package-search/
├── SKILL.md
├── data/
│   └── packages.json          # 索引（67 包 / 1268 文件，约 0.65MB）
├── scripts/
│   ├── search.py              # 检索主入口
│   ├── refresh_index.py       # 从镜像站全量重建索引
│   └── gen_matrix.py          # 由索引生成版本矩阵文档
└── references/
    ├── version_matrix.md      # 版本矩阵（自动生成）
    └── install_guide.md       # 安装指南与坑点手册
```
