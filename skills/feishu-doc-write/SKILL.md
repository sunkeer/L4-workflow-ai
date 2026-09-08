---
name: feishu-doc-write
description: 把结构化内容（复现步骤、Bug 记录、性能对比表、实验报告）程序化写入飞书（Lark）wiki / 文档。当用户要把 Markdown 或报告写入飞书文档、向已有文档批量追加章节、或自动化把实验结果/日志结论记录到飞书时使用。封装了「fetch 现有结构 → 用飞书 XML 块格式写文件 → lark-cli 追加/覆盖」的可靠流程，规避 shell 转义与长内容内联导致的失败。
---

# feishu-doc-write —— 飞书文档程序化写入

把任意结构化内容可靠地写进飞书 wiki / 文档。核心经验：**内容永远写进 `.xml` 文件再用 `@file` 传入，绝不内联**，否则长内容 + 中文 + 引号会毁掉 shell 转义。

## 适用场景
- 把一份「复现步骤 / 排障记录 / 性能对比」报告写进飞书 wiki，作为团队可查阅的文档。
- 向已有飞书文档追加新章节（如先写框架、跑完后再回填「结果」一节）。
- 自动化工作流里需要落盘一份结构化记录到飞书（而非只留在本地）。

## 前置条件
1. WorkBuddy 已连接飞书连接器（`lark-cli` 可用）。连接器连上时 `lark-cli` 通常在 PATH；若不在，用绝对路径：
   ```bash
   LARK="$(command -v lark-cli || echo "$HOME/.workbuddy/binaries/node/cli-connector-packages/lark-cli")"
   "$LARK" --version   # 期望 >= 1.0.0
   ```
2. 目标文档已存在（wiki 或 docx）。拿到它的 **URL**（形如 `https://<tenant>.feishu.cn/wiki/<token>` 或 `https://<tenant>.feishu.cn/docx/<token>`）。

## 流程（推荐 4 步）

### 1. 读取现有文档结构（可选，但首次写/追加前建议做）
```bash
"$LARK" docs +fetch --doc "<DOC_URL>" --doc-format markdown --detail simple
```
确认文档当前有哪些章节、是否已有 `<title>`，决定是「新建（带 title）」还是「追加（只给块片段、不带 title）」。

### 2. 用飞书 XML 块格式写内容文件
新建文档时以唯一 `<title>` 开头；**向已有文档追加时只写块片段（h1-h9 / p / callout / table …），不要带 `<title>`**。
把内容写到本地 `.xml` 文件（例如 `content.xml`）。块格式速查见 [references/block_format.md](references/block_format.md)，可复制 [assets/template_append.xml](assets/template_append.xml) 改。

标题自动编号示例：`<h1 seq="auto">章节</h1>` → 渲染为 `1 章节`、`1.1 子章节`。

### 3. 写入文档
- **追加**到文档末尾：
  ```bash
  "$LARK" docs +update --doc "<DOC_URL>" --command append --content @./content.xml
  ```
- **整体覆盖**（会丢弃原文档其它富内容，慎用）：把 `--command append` 换成 `--command overwrite`。
- 也可用更精细的 `block_insert_after` / `block_replace` / `str_replace` 做定点编辑（先 `lark-cli docs +fetch --detail with-ids` 拿块 ID）。

> 关键：`--content` 后面用 `@./content.xml`（读文件），不要用 `--content "..."` 内联。多行内容用 `@file` 或 `-`（stdin）。

### 4. 回填后校验
```bash
"$LARK" docs +fetch --doc "<DOC_URL>" --doc-format markdown --detail simple | head -40
```
确认新章节已渲染、表格/代码块/高亮块正常。

## 踩过的坑（务必避开）
- **长内容内联 `--content "..."` 必炸**：中文引号、`&`、`<`、`>`、反斜杠、heredoc 都会让 shell 转义混乱，远端 SSH 尤其严重。一律写文件 + `@file`。
- **追加时不要带 `<title>`**：已有文档的 title 已存在，再写一个会被当成重复/嵌套出错。追加片段从 `<h1>` 起即可。
- **文本转义**：块内文本里的 `<` → `&lt;`、`>` → `&gt;`、`&` → `&amp;`；换行用 `<br/>`。不要转义标签本身。
- **层级连续**：`<h1>` 后不能直接 `<h3>`，先 `<h2>`。需要编号用 `seq="auto"`。
- **callout 内只允许 p/ol/ul/checkbox/行内标签**，不能放 table/img/pre/hr/grid。
- **代码必须放在 `<code>` 内**：`<pre lang="go"><code>...</code></pre>`，不要直接把文本放 `<pre>` 下。
- **表格单元格用 `<p>` 包裹**：`<td><p>内容</p></td>`，否则渲染异常。

## 与其它 skill 的关系
本 skill 是「写飞书」的高层封装。更底层的块格式、fetch/update 全量参数见飞书连接器自带 `lark-doc` skill（`lark-cli skills read lark-doc` 及其 `references/lark-doc-xml.md`）。本 skill 的价值在于固定了「写文件 + @file 追加」这条最稳的路径。

## 最小可用示例
```bash
LARK="$(command -v lark-cli || echo "$HOME/.workbuddy/binaries/node/cli-connector-packages/lark-cli")"
cat > content.xml <<'XML'
<h1 seq="auto">本次复现结论</h1>
<p>优化分支稳态约 0.97 s/step，较原 BW1000 基线 5.19 s/step 快约 5×。</p>
<callout emoji="⚠️" background-color="light-red"><p>本机 diffusers 为 0.37.0.dev0，README 要求 0.38.0，离线无法安装；已打一行最小补丁。</p></callout>
XML
"$LARK" docs +update --doc "<DOC_URL>" --command append --content @./content.xml
```
