# L4-workflow-ai

AI 工作流工具集：把重复的「调研 / 复现 / 记录」动作沉淀为可复用 skill，便于在 WorkBuddy 中一键调用或二次开发。

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

## 目录结构
```
L4-workflow-ai/
├── README.md
└── skills/
    ├── feishu-doc-write/
    │   ├── SKILL.md                  # skill 定义与完整流程
    │   ├── references/
    │   │   └── block_format.md        # 飞书 DocxXML 块格式速查
    │   └── assets/
    │       └── template_append.xml    # 追加片段模板（改占位符即用）
    └── dcu-vllm-deploy/
        ├── SKILL.md                  # skill 定义（三步流程：算显存→下载→部署）
        ├── scripts/                   # 全部在 DCU 节点上执行
        │   ├── check_env.sh           # 节点体检（DTK/torch/vLLM/显存/磁盘/端口）
        │   ├── plan_deployment.py     # 显存匹配计算（下载前的可行性判定）
        │   ├── download_model.py      # ModelScope 查找 + 下载（断点续传）
        │   ├── dcu_platform_patch.py  # 平台探测补丁（pynvml/amdsmi 缺失）
        │   ├── sitecustomize.py       # spawn worker 补丁全局生效
        │   └── start_template.sh      # 参数化 vLLM 启动脚本（模型无关）
        └── references/
            ├── vram_estimation.md     # 显存估算公式与经验数值
            ├── deploy_flow.md         # 完整流程 + 验收命令集
            └── troubleshooting.md     # 坑点手册（A~F 六类，全实战）
```

## 说明
- `feishu-doc-write` 是飞书连接器 `lark-doc` skill 的「高层封装」，固定了最稳的写入路径；更底层的块格式与全量参数见飞书连接器自带 `lark-doc` skill。
- `dcu-vllm-deploy` 沉淀自 2026-09-08 的 MinerU2.5-Pro DCU 部署（vLLM 路线）与 PP-OCRv6（非 vLLM 架构判定为 ONNX 路线），流程边界经过两个案例验证。
