---
name: vllm-pooling-model-adapter
description: 把任意 HuggingFace 的 embedding / 向量检索 / 多模态检索模型接进 vLLM 的 pooling runner 跑成 OpenAI 兼容服务。当遇到「config.json 的 architectures 不在 vLLM registry」「报 Model architectures [...] are not supported」「Qwen3VLForEmbedding / …ForEmbedding 类名不存在」「要把 BGE/Qwen3-VL/CLIP 系模型做成 /v1/embeddings 服务」「vLLM --task embed / --runner pooling 起不来」，或需要判断一个模型到底该用生成 runner 还是池化 runner 时使用。核心产出：一个不改权重、不改结构的适配模块 + 一套四层数值对齐排查法。
agent_created: true
---

# vllm-pooling-model-adapter —— 把 embedding 模型接进 vLLM pooling runner

vLLM 的生成模型走 `generate` runner，向量检索类模型要走 `pooling` runner。
很多 HF 上的 embedding 模型（尤其是 `xxxForEmbedding` 这种自定义类名）**vLLM 不认**，
但它们往往是同家族生成模型 + 一个池化头，权重完全同构。

本 skill 的结论：**不改权重、不改结构、不重训**，只在服务层补 pooling 头 + 注册 arch 名。

## 第 0 步：判型（先判断该不该走这条路）

```bash
python3 scripts/check_model_embeddable.py <模型目录或 HF id>
```

脚本输出三件事：

| 判据 | 含义 |
| --- | --- |
| `architectures` 是否在 vLLM registry | 在 → 可能什么都不用做；不在 → 需要适配 |
| 是否有 `modeling_*.py` | 有 → 是 remote code，**不能**直接复用 vLLM 内置类，得逐个模块对齐 |
| 池化语义线索 | `mmemb_model.json` / `1_Pooling/config.json` / `modules.json` / `sentence_bert_config.json` 里的 pooling、normalize |

**能走本 skill 的前提**：无自定义建模代码（或自定义代码只是加了 pooler），且同家族生成模型
vLLM 已支持。反之（真自定义结构、ONNX、非 transformer 架构）不在范围内。

## 第 1 步：找 vLLM 里可复用的基类

不要从零写模型。vLLM 自带一批「某 VLM/LLM + pooler」的现成实现，照着抄：

```bash
SP=$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
grep -n "is_pooling_model = True" -B6 $SP/model_executor/models/*.py | head -40
grep -n "DispatchPooler.for_embedding" -B10 $SP/model_executor/models/*.py | head -30
```

最值得看的两个样板：

- `vllm/model_executor/models/nemotron_vl.py` → `LlamaNemotronVLForEmbedding`（VLM + 池化）
- `vllm/model_executor/models/bert.py` / `colqwen3.py`（纯文本 / late-interaction）

## 第 2 步：写适配模块

模板见 [assets/adapter_template.py](assets/adapter_template.py)，核心就三件事：

```python
class XxxForEmbedding(XxxForConditionalGeneration, VllmModelForPooling):
    is_pooling_model = True                      # ① 让 registry 认它是池化模型
    default_seq_pooling_type = "LAST"            # ② 默认池化方式（CLS / LAST / MEAN）
    default_tok_pooling_type = "ALL"
    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pooler = DispatchPooler.for_embedding(   # ③ 挂池化头
            vllm_config.model_config.pooler_config)
```

要点：

- `is_pooling_model` 只是**类属性**（`getattr(cls, "is_pooling_model", False)`），不需要实现接口方法。
- `prefix` 默认值要和父类保持一致（有的模型是 `""`，有的是 `"model"`），**抄父类签名**。
- 池化方式必须和模型的训练口径一致。搞错了精度会掉但不会报错——这是最隐蔽的坑。

## 第 3 步：注册（关键是 spawn 子进程）

```python
ModelRegistry.register_model("XxxForEmbedding", "adapter_module:XxxForEmbedding")
```

**只在主进程注册是不够的**：vLLM V1 的 EngineCore 是 spawn 出来的子进程，会重新 import，
主进程的 monkeypatch 传不过去。必须让每个解释器启动就注册，最稳的做法是
`sitecustomize.py`（随 [assets/sitecustomize.py](assets/sitecustomize.py) 分发）：

```bash
export PYTHONPATH=/path/to/adapter:$PYTHONPATH   # 目录里有 sitecustomize.py
```

`sitcustomize` 里一定要 try/except 兜住——抛异常会让该机器上所有 python 进程都起不来。

## 第 4 步：启动

```bash
vllm serve <MODEL_PATH> \
  --served-model-name <NAME> \
  --runner pooling \
  --pooler-config '{"seq_pooling_type": "LAST"}' \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --no-trust-remote-code
```

## 第 5 步：四层数值对齐（本 skill 的核心）

**不要**一上来就对比最终向量然后猜。按下面四层逐层收敛，每层都能把问题锁死在一个环节：

| 层 | 做法 | 对不上说明什么 |
| --- | --- | --- |
| L1 模型 & 池化 | 用**原始文本**（不走 chat template）打 `/v1/embeddings` 的 `input`，和 reference 的 last-token hidden state 比 | 对不上 → 权重加载 / 池化位置 / dtype 有问题 |
| L2 prompt 构造 | 用 `/tokenize` 拿 vLLM 实际 token 序列，和 reference 的 `apply_chat_template` 逐 id 比 | 长度一致但内容不同 → chat template 或 `add_generation_prompt` 不一致 |
| L3 池化位置 | reference 侧扫最后 N 个位置，看哪个位置和 vLLM 输出最像 | 在中间某个位置峰值 → vLLM 池化在别处（见下方大坑） |
| L4 误差量级 | 归一化后最大分量差 | 1e-3 量级 = bf16 正常；1e-1 量级 = 结构性差异 |

脚本：[scripts/numeric_align.py](scripts/numeric_align.py)。判据：**余弦 ≥ 0.999 视为等价**。

### 头号大坑：`add_generation_prompt` 默认 False

vLLM 的池化/embedding 请求里 `add_generation_prompt` **默认是 False**，
于是池化落在**用户轮结束符**上；而绝大多数 embedding 模型的训练口径是
`add_generation_prompt=True`（池化在 `<|im_start|>assistant\n` 之后）。

实测差距：同一模型同一 prompt，不开 0.764、开了 **0.9999**。
请求里必须显式加：

```json
{"model": "...", "messages": [...], "add_generation_prompt": true}
```

注意 `/tokenize` 接口默认是 True，所以「tokenize 结果和 reference 完全一致」**不能**
证明真正推理时也一致——L2 和 L3 都要做。

### 其他常见偏差

| 现象 | 原因 |
| --- | --- |
| 维度不是 hidden_size | 命中了别家的 pooler（如 late-interaction 的 ColQwen3 走 token_embed） |
| 范数恰好 = 1.0 | vLLM 输出前做了 L2 归一化；若模型要求不归一化需自行还原 |
| 不同图片输出几乎相同 | vision tower 没生效（被 `--language-model-only` 关掉 / mm 参数没传） |
| 批量 vs 逐条有 1e-4 差异 | bf16 batch 内归约顺序不同，正常 |
| 图文请求报 500 | `--limit-mm-per-prompt` 必须是 JSON：`'{"image": 4}'` |

## 边界与不适用

- 真自定义结构（有 `modeling_*.py` 且与内置实现差异大）→ 必须写完整模型实现，不在本 skill 范围
- 非 transformer / ONNX / 纯 CPU 模型 → 走 onnxruntime + FastAPI
- rerank / cross-encoder：用 `DispatchPooler.for_seq_cls`，套路同构但池化头不同
- late-interaction（ColBERT/ColQwen）：走 `token_embed`，vLLM 已有现成 arch，通常不用适配

## 实战来源

2026-10-27，Trident-Qwen3VL-2B（`architectures: ["Qwen3VLForEmbedding"]`）在 BW1000 + vLLM 0.29 上适配：
纯文本余弦 0.99990、图文 0.99959；`add_generation_prompt` 那个坑让对齐卡了半小时。
