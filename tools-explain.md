# L4-workflow-ai 工具详解

本文件是仓库 `L4-workflow-ai` 各 skill 的**详细文档**：工具作用、上线时间、复用方式、目录结构与坑点。
Skill 索引与快速跳转见仓库根目录 [README.md](README.md)。

---

## 上线工具：feishu-doc-write（飞书文档程序化写入）

### 工具作用
把结构化内容（复现步骤、Bug 记录、性能对比表、实验报告）**可靠地写入飞书（Lark）wiki / 文档**。
核心解决两类痛点：

1. **长内容写入飞书总是被 shell 转义搞崩** —— 本工具固定「内容写进 `.xml` 文件 → 用 `lark-cli --content @file` 追加」这条最稳路径，规避中文引号 / `&` / `<` / 反斜杠 / heredoc 带来的转义灾难。
2. **团队知识沉淀自动化** —— 复现跑完、Bug 修完、对比出结果，直接用脚本把结论追加进指定飞书文档，无需手工粘贴；并支持「先写框架、跑完回填结果」的分阶段写法。

典型用例：把一次模型性能复现（配置、步骤、遇到的坑、对比表）落盘成团队可查阅的飞书 wiki。

### 上线时间
- **2026-09-08**（首次提交，含 `feishu-doc-write` skill 与本文档）

### 复用方式

#### 方式 A：作为 WorkBuddy skill 使用（推荐）
1. 把本仓库 `skills/feishu-doc-write/` 放入 WorkBuddy 的 skill 目录（用户级 `~/.workbuddy/skills/` 或项目级 `.workbuddy/skills/`）。
2. 在对话中让 Agent 调用该 skill，例如：
   > 「把这次 DiT4DiT 复现的步骤和性能对比写进飞书文档 <URL>」
3. skill 会自动走：读取文档结构 → 用飞书 XML 块格式生成内容文件 → `lark-cli` 追加/覆盖 → 回填校验。

#### 方式 B：直接命令行使用（无需 skill 框架）
前置：WorkBuddy 已连接飞书连接器，`lark-cli` 可用（若不在 PATH，用绝对路径
`$HOME/.workbuddy/binaries/node/cli-connector-packages/lark-cli`）。

```bash
LARK="$(command -v lark-cli || echo "$HOME/.workbuddy/binaries/node/cli-connector-packages/lark-cli")"

# 1) 读取现有文档结构（可选）
"$LARK" docs +fetch --doc "<DOC_URL>" --doc-format markdown --detail simple

# 2) 把要写的内容按飞书 XML 块格式存成文件（参考 skills/feishu-doc-write/assets/template_append.xml）
#    —— 追加时不要带 <title>，从 <h1> 起

# 3) 追加到文档末尾
"$LARK" docs +update --doc "<DOC_URL>" --command append --content @./content.xml

# 4) 校验
"$LARK" docs +fetch --doc "<DOC_URL>" --doc-format markdown --detail simple | head -40
```

> 关键点：`--content` 后面用 `@./content.xml`（读文件），**不要**内联 `--content "..."`。
> 块格式速查见 `skills/feishu-doc-write/references/block_format.md`。

---

## 上线工具二：dcu-vllm-deploy（Hygon DCU 通用 vLLM 推理服务部署）

### 工具作用
在 Hygon DCU（DTK 26.x，如 BW1000）上把**任意 ModelScope/HuggingFace 架构**的 LLM / 多模态模型
通过 vLLM 部署成 OpenAI 兼容推理服务，**模型无关、流程固化**。核心解决三类痛点：

1. **下错模型白费几十 GB 下载** —— 先用 ModelScope 元数据（文件清单 + config.json）与本机 DCU
   显存做匹配计算（权重/TP + 开销 + KV 预算 vs 每卡可用），放不下在下载前就拦住，并给出推荐
   TP / dtype / 满长并发估算。
2. **模型名不完整 / 手工下载易错** —— 短名自动搜索 ModelScope 解析，SDK 断点续传下载到
   `/root/private_data/models/<模型名>`（持久化 PVC，避开容器回收丢数据的坑）。
3. **DCU 上 vLLM 起不来** —— 内置两个必装平台补丁（无 pynvml/amdsmi 导致的平台探测失败 +
   spawn worker 补丁传递不到）与参数化启动模板，规避 `set -u`、`--limit-mm-per-prompt` JSON
   写法等 DCU 特有坑；配套完整坑点手册。

典型用例：`「把 MinerU2.5-Pro 部署到 ssh -p 11360 的 DCU 机器上」` —— skill 自动走
显存评估 → ModelScope 下载 → 补丁 → 启动 → 三层验收（连通/功能/性能）。

### 上线时间
- **2026-09-08**（由 MinerU2.5-Pro vLLM 部署 + PP-OCRv6 路线判定两次实战沉淀抽象）

### 复用方式
```bash
# 1) 同步脚本到 DCU 节点
scp -P <PORT> skills/dcu-vllm-deploy/scripts/* root@<HOST>:/root/vllm_deploy/

# 2) 体检 → 显存评估（下载前先算）→ 下载 → 启动 → 验收
bash /root/vllm_deploy/check_env.sh
python3 /root/vllm_deploy/plan_deployment.py <模型名>
python3 /root/vllm_deploy/download_model.py <模型名>
cd /root/vllm_deploy && MODEL_PATH=/root/private_data/models/<模型名> \
  SERVED_NAME=<服务名> nohup bash start_template.sh > serve.log 2>&1 &
```
> 完整流程与验收命令集见 `skills/dcu-vllm-deploy/references/deploy_flow.md`。

---

## 上线工具三：dtk-ai-package-search（DCU AI 生态包检索）

### 工具作用
把光源（SourceFind）镜像站的海光 DCU AI 生态包做成**本地可离线检索的索引**，
输入包名直接拿到匹配当前 DTK / Python / torch 版本的 whl 直链与 `pip install` 命令。
核心解决三类痛点：

1. **官网只能一层层点，翻不动** —— 镜像站是 JS 驱动的文件服务，67 个包 × 多 DAS 目录，
   找一个包要点五六次还得手动「生成直链」。本工具把整棵树抓成 `packages.json`（**67 个包 /
   1268 个文件**，覆盖 DAS1.0~DAS1.8 及 `previous_release/` 历史版本），一条命令出结果。
2. **版本错配导致装完就崩** —— DCU 生态包必须同时对上 DTK（`dtk2604`）、Python（`cp310`）、
   torch（`torch271`）三个维度，装错的典型症状不是安装失败而是 `import` 报 `undefined symbol`
   或 `torch.cuda.is_available()` 返回 `False`。工具支持 `--from-env` 直接读节点上的
   `/opt/dtk/.info/version` 与 Python/torch 版本自动过滤，并在零结果时列出该包实际可用的组合，
   不让人瞎猜。
3. **DAS↔DTK 对应关系没有权威表** —— 从真实数据反推生成矩阵（`DAS1.8`↔`dtk2604`、
   `DAS1.7`↔`dtk25042`……），并标注了 **DAS1.4 不存在**、`DAS1.2` 内混有 `dtk24041` 文件
   这类容易踩的例外。

典型用例：`「DTK 26.04 + Python 3.10 要装 vllm，给我命令」` → 一条命令出可直接粘贴的
`pip install` 直链。

### 上线时间
- **2026-09-08**（由镜像站直链整理任务沉淀；索引与飞书文档
  <https://lhui08qsvi.feishu.cn/wiki/ApPYwC4VCieN70kGpr2cqkQCnOe> 同源）

### 复用方式
```bash
# 放入 skill 目录后即可让 Agent 调用；也可纯命令行使用：
cd skills/dtk-ai-package-search

python3 scripts/search.py --list-versions              # DAS ↔ DTK 对应关系
python3 scripts/search.py --info deepspeed             # 某包的完整版本矩阵
python3 scripts/search.py vllm --dtk 2604 --py 3.10 -f pip   # 出 pip 命令
python3 scripts/search.py deepspeed --from-env -f pip  # 在 DCU 节点上自动匹配本机环境
python3 scripts/search.py flash_attn --dtk 2604 -f wget      # 无网环境：先下载再拷贝

python3 scripts/refresh_index.py && python3 scripts/gen_matrix.py  # 刷新索引 + 同步矩阵文档
```
> 版本矩阵见 `references/version_matrix.md`，安装顺序与坑点手册见 `references/install_guide.md`。

> **逆向要点**（镜像站没有公开 API 文档）：目录列表接口是
> `GET /api-static/file/ListFile?CategoryID=4&Path=<路径>`，参数是**大写 P 的 `Path`** ——
> 写成小写 `path` 会被服务端静默忽略并永远返回根目录，极易误判成「接口不支持子目录」；
> 成功时 `code` 返回字符串 `"success"` 而非 `0`。

---

## 上线工具四：vllm-pooling-model-adapter（embedding 模型接入 vLLM 池化 runner）

### 工具作用
把任意 HuggingFace 的 **embedding / 向量检索 / 多模态检索**模型接进 vLLM 的 `pooling`
runner，跑成 OpenAI 兼容的 `/v1/embeddings` 服务。核心解决两类痛点：

1. **模型认不出来** —— HF 上大量 `xxxForEmbedding` 自定义类名不在 vLLM registry 里，
   `vllm serve` 直接报「Model architectures not supported」。但这些模型往往是
   「同家族生成模型 + 一个池化头」，权重完全同构。本工具给出**不改权重、不改结构**的
   适配法：子类化 + `is_pooling_model` 标记 + `DispatchPooler.for_embedding` + registry 注册，
   再用 `sitecustomize.py` 让 EngineCore 的 spawn 子进程也生效。
2. **向量算出来不对，但服务不报错** —— 池化位置、prompt 构造、归一化任何一处对不上，
   服务照常返回 200，精度悄悄掉。本工具给一套**四层数值对齐法**
   （模型&池化 → prompt 构造 → 池化位置 → 误差量级），每层把问题锁在一个环节，
   不用对着最终向量瞎猜。

典型用例：`「Trident-Qwen3VL-2B 用 vLLM 部署成 embedding 服务」` —— 适配模块 + 启动参数
+ 对拍脚本一条龙。

### 上线时间
- **2026-10-27**（由国芯国模黑客松 T02 · Trident-Qwen3VL-2B 在 BW1000 + vLLM 0.29 适配沉淀）

### 复用方式
```bash
# 0) 判型：能不能走这条路
python3 scripts/check_model_embeddable.py <模型目录>

# 1) 复制模板改三处（基类 / ARCH_NAME / 池化方式）
cp assets/adapter_template.py adapter/my_pooling_adapter.py
cp assets/sitecustomize.py   adapter/
export PYTHONPATH=/path/to/adapter:$PYTHONPATH

# 2) 起服务
vllm serve <MODEL> --runner pooling \
  --pooler-config '{"seq_pooling_type": "LAST"}' \
  --max-model-len 32768 --no-trust-remote-code

# 3) 四层对齐
python3 scripts/numeric_align.py --path <MODEL> --model <SERVED_NAME>
```

> **头号坑**：vLLM 的 embedding 请求 `add_generation_prompt` **默认 False**，
> 池化会落在用户轮结束符上。多数 embedding 模型要求 True。
> 实测同一 prompt：不开余弦 0.764，开了 **0.9999**。请求里必须显式带上。
>
> **次坑**：`/tokenize` 接口默认是 True，所以「tokenize 结果和参考一致」**不能**证明
> 真实推理时也一致——L2 和 L3 两层都要做。

---

## 上线工具五：dcu-vllm-backend-triage（DCU 上 vLLM attention 后端选型）

### 工具作用
诊断「DCU 上 vLLM 加载完权重、一跑推理就崩在 attention」这一类问题，一次定位到该用哪个
`--attention-backend`。DTK 版 vLLM 常常缺几个编译期扩展，导致默认选中的后端不可用：

| 现象 | 根因 |
| --- | --- |
| `AttributeError: '_OpNamespace' '_rocm_C' object has no attribute 'paged_attention'` | 缺 `vllm._rocm_C` → ROCM_ATTN 不可用 |
| `AssertionError: FlashAttention version not detected` | 缺 `vllm_flash_attn`（**不是**系统里的 `flash_attn`，两个包） |
| 两者都缺 | 用 `TRITON_ATTN`（纯 Triton，不依赖编译扩展） |

### 上线时间
- **2026-10-27**（BW1000 / DTK 26.04 / vLLM 0.29 实测：ROCM_ATTN ❌、FLASH_ATTN ❌、TRITON_ATTN ✅）

### 复用方式
```bash
python3 scripts/probe_attention_backends.py            # 体检 + 给建议
vllm serve <MODEL> ... --attention-backend TRITON_ATTN
```

> 注意与 `dcu-vllm-deploy` 里的 `dcu_cutlass_patch.py` 区分：那个是 **import 期**的
> `cutlass_*_supported()` 能力探测崩（量化路径），本工具是**推理期**的 attention 后端崩。
> 两者报错长得像（都是 `'_OpNamespace' '_C'`），但时机、算子、位置完全不同。

---

## 目录结构
```
L4-workflow-ai/
├── README.md            # Skill 索引（点击跳转各 SKILL.md）
├── tools-explain.md     # 本文件：各 skill 详细作用 / 上线时间 / 复用方式 / 坑点
└── skills/
    ├── feishu-doc-write/
    │   ├── SKILL.md                  # skill 定义与完整流程
    │   ├── references/
    │   │   └── block_format.md        # 飞书 DocxXML 块格式速查
    │   └── assets/
    │       └── template_append.xml    # 追加片段模板（改占位符即用）
    ├── dcu-vllm-deploy/
    │   ├── SKILL.md                  # skill 定义（三步流程：算显存→下载→部署）
    │   ├── scripts/                   # 全部在 DCU 节点上执行
    │   │   ├── check_env.sh           # 节点体检（DTK/torch/vLLM/显存/磁盘/端口）
    │   │   ├── plan_deployment.py     # 显存匹配计算（下载前的可行性判定）
    │   │   ├── download_model.py      # ModelScope 查找 + 下载（断点续传）
    │   │   ├── dcu_platform_patch.py  # 平台探测补丁（pynvml/amdsmi 缺失）
    │   │   ├── sitecustomize.py       # spawn worker 补丁全局生效
    │   │   └── start_template.sh      # 参数化 vLLM 启动脚本（模型无关）
    │   └── references/
    │       ├── vram_estimation.md     # 显存估算公式与经验数值
    │       ├── deploy_flow.md         # 完整流程 + 验收命令集
    │       └── troubleshooting.md      # 坑点手册（A~F 六类，全实战）
    ├── dtk-ai-package-search/
    │   ├── SKILL.md                  # skill 定义（对版本 → 检索 → 出命令 → 验证）
    │   ├── data/
    │   │   └── packages.json          # 生态包索引（67 包 / 1268 文件，约 750KB）
    │   ├── scripts/
    │   │   ├── search.py              # 检索主入口（含 --from-env 本机环境探测）
    │   │   ├── refresh_index.py       # 从镜像站全量重建索引
    │   │   └── gen_matrix.py          # 由索引生成版本矩阵文档
    │   └── references/
    │       ├── version_matrix.md      # DAS↔DTK↔Python 矩阵（自动生成，勿手改）
    │       └── install_guide.md       # 安装顺序、验证脚本、7 类常见坑
    ├── vllm-pooling-model-adapter/
    │   ├── SKILL.md                  # 判型 → 选基类 → 写适配 → 注册 → 四层对齐
    │   ├── assets/
    │   │   ├── adapter_template.py    # 适配模块模板（改三处即用）
    │   │   └── sitecustomize.py       # 让 spawn 子进程也注册到 vLLM registry
    │   └── scripts/
    │       ├── check_model_embeddable.py  # 判型：该不该走这条路
    │       └── numeric_align.py           # L1~L4 数值对齐
    └── dcu-vllm-backend-triage/
        ├── SKILL.md                  # 判定树 + 后端对照表 + 与 cutlass 坑的区分
        └── scripts/
            └── probe_attention_backends.py  # 扩展体检 + 后端建议
```

## 说明
- `feishu-doc-write` 是飞书连接器 `lark-doc` skill 的「高层封装」，固定了最稳的写入路径；更底层的块格式与全量参数见飞书连接器自带 `lark-doc` skill。
- `dcu-vllm-deploy` 沉淀自 2026-09-08 的 MinerU2.5-Pro DCU 部署（vLLM 路线）与 PP-OCRv6（非 vLLM 架构判定为 ONNX 路线），流程边界经过两个案例验证。
- `dtk-ai-package-search` 的 `data/packages.json` 是**快照而非实时数据**，头部 `generated_at`
  即时效；与官网对不上时先跑 `refresh_index.py`。`version_matrix.md` 由脚本生成，改数据后
  记得重跑 `gen_matrix.py` 同步。
- 三个 skill 的关系：`dtk-ai-package-search` 负责「装什么版本」，`dcu-vllm-deploy` 负责
  「怎么把服务跑起来」，`feishu-doc-write` 负责「把结论沉淀成团队文档」。
