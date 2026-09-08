# 飞书 DocxXML 块格式速查

> 整理自 WorkBuddy 飞书连接器 `lark-doc` skill 的 `references/lark-doc-xml.md`。属性必须写成 `name="value"`，禁止省略引号。默认宽度约 820px（宽版约 1020px）。

## 顶层结构
- **新建文档**：以唯一 `<title>文档标题</title>` 开头，之后是正文块。
- **追加章节**：只写块片段（从 `<h1>` 起），**不要带 `<title>`**。

## 常用块标签
- 文本/标题：`p, h1-h9, blockquote, hr`
- 行内富文本：`b`（粗）、`em`（斜）、`u`（下划线）、`del`（删除线）、`span`、`<br/>`（换行）
- 链接：`<a type="url-preview" href="URL">链接标题</a>`
- 行内公式：`<latex>E = mc^2</latex>`
- 列表：
  ```xml
  <ol><li>第一项<ul><li>子项</li></ul></li><li>第二项</li></ol>
  ```
  子列表放 `<li>` 内；有序列表默认 `seq="auto"`，可从指定数字开始如 `seq="3"`。
- 代码：
  ```xml
  <pre lang="go" caption="示例"><code>fmt.Println("hello")</code></pre>
  ```
  代码必须放在 `<code>` 内，禁止直接放 `<pre>` 下；`caption` 可省略。
- 图片：`<img path="@./photo.png"/>`（本地）或 `<img href="URL"/>`（网络，限 PNG/JPEG/GIF/WebP，单图 ≤20MiB）或 `<img src="token"/>`（复制已有）。可选 `width/height/caption/name`。
- 附件：`<source path="@./report.pdf" name="报告.pdf"/>`，可独立用、行内放 `<p>` 内、或 `<figure view-type="Card|Preview"><source/></figure>`。
- 待办：`<checkbox done="true|false">todo</checkbox>`
- 对齐：`p, h1-h9, li, checkbox, title` 支持 `align="left|center|right"`。

## 标题与编号
- 层级须连续，不跳级（`<h1>` 后先 `<h2>` 再 `<h3>`）。
- 自动编号：`seq="auto"` → 一级 `1`、二级 `1.1`。

## 表格
```xml
<table>
  <colgroup><col width="360"/><col width="360"/></colgroup>
  <thead><tr><th background-color="light-gray"><p>指标</p></th><th background-color="light-gray"><p>值</p></th></tr></thead>
  <tbody>
    <tr><td><p>s/step</p></td><td><p>0.97</p></td></tr>
  </tbody>
</table>
```
- `<colgroup><col width=".."/>` 紧跟 `<table>` 定义列宽；`width` 列宽，`span` 连续作用列数。
- `<th>/<td>` 支持 `background-color, vertical-align(top|middle|bottom), colspan, rowspan`。
- 表头优先 `light-gray` / `medium-gray`；彩色单元格只表达状态/分类。被合并单元格不再写入。

## 高亮块 callout
```xml
<callout emoji="💡" background-color="light-blue" border-color="blue"><p>高亮内容</p></callout>
```
子块仅支持 `p, ol, ul, checkbox, 行内标签`；**禁止** table/img/pre/hr/grid/whiteboard 等资源块。可选 `text-color`。

## 颜色（表达语义，保持一致）
- 色相：`red, orange, yellow, green, blue, purple, gray`。
- `text-color/border-color` 用基础色相；`<span>/<th>/<td>/<button>` 背景支持基础色相、`light-{色相}`、`medium-gray`；callout 背景支持 `gray`、`light-{色相}`、`medium-{色相}`。

## 转义规则
禁止转义标签本身；只转义标签内文本：
- `<` → `&lt;`，`>` → `&gt;`，`&` → `&amp;`，换行 → `<br/>`
- 正确：`<p>A &amp; B 的对比：1 &lt; 2</p>`
- 错误：把 `<p>` 本身写成 `&lt;p&gt;`
