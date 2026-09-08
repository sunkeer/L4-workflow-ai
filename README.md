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

## 目录结构
```
L4-workflow-ai/
├── README.md
└── skills/
    └── feishu-doc-write/
        ├── SKILL.md                  # skill 定义与完整流程
        ├── references/
        │   └── block_format.md        # 飞书 DocxXML 块格式速查
        └── assets/
            └── template_append.xml    # 追加片段模板（改占位符即用）
```

## 说明
- `feishu-doc-write` 是飞书连接器 `lark-doc` skill 的「高层封装」，固定了最稳的写入路径；更底层的块格式与全量参数见飞书连接器自带 `lark-doc` skill。
- 仓库初始为空，本提交为首次内容。
