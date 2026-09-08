# DCU vLLM 部署显存估算

## 为什么下载前先算

模型动辄几十 GiB，DCU 节点带宽/磁盘有限，白下一次代价高。`plan_deployment.py` 在**下载之前**就
通过 ModelScope API 拿到文件清单（权重大小）和 config.json（结构），先算再下。

## vLLM 显存行为（关键前提）

vLLM 启动时按 `--gpu-memory-utilization`（默认 0.90）**预留**该比例显存：先加载权重，剩余全部
划给 KV cache block pool。所以：

- 「服务占了卡 89% 显存」不代表权重有 57G，多数是 KV 预留（MinerU 1.2B 实测：权重 ~2.4G，
  HCU0 占用 89%×64G）。
- 判定「放不放得下」只看**权重 + 开销 ≤ 每卡可用**，KV 是自动填充的余量。

## 公式

```
每卡需求(不含KV) = 权重总量 / TP + 开销
开销            ≈ 2 GiB（HIP context/图/碎片兜底）+ 5% × 权重/TP

每卡可用 = 每卡总显存 × gpu_util − 其他进程已占用

KV 预算/卡 = 每卡可用 − 每卡需求
KV 每 token 字节 = 2(K和V) × num_hidden_layers × num_key_value_heads × head_dim × 2(bf16)
    注意 GQA 模型用 num_key_value_heads（常远小于 num_attention_heads）
满长并发 ≈ KV预算 / KV每token字节 / max_model_len
```

## 判定标准

| verdict | 条件 | 动作 |
|---|---|---|
| OK | 需求 ≤ 可用 且 KV 预算 ≥ 每卡显存×20% | 正常部署 |
| TIGHT | 需求 ≤ 可用 但 KV 预算 < 20% | 能跑但并发低；减 max_model_len / 加卡 / 换量化 |
| FAIL | 需求 > 可用 | 放不下；换量化版（权重小2~4倍）/ 换节点 / 加 TP |

## dtype 换算

- 权重文件大小 ≈ 存储精度下的体积（社区模型绝大多数 bf16/fp16）
- bf16/fp16 部署：大小即所需，无需换算
- 量化模型（GPTQ/AWQ/HQQ，config.json 有 `quantization_config`）：文件多大占多大，
  **BW1000 无 FP8 单元**——fp8 量化模型在 BW1000 上只能当告警放着（bf16 路径不触发），
  真跑量化要走海光 lmslim（awqlite/gptqlite）
- int8/int4 量化权重 ≈ bf16 的 1/2 / 1/4

## 经验数值（实测锚点）

| 模型 | 权重 | 部署形态 | 实测 |
|---|---|---|---|
| MinerU2.5-Pro (Qwen2-VL 1.2B) | ~2.4 GiB | TP=1 bf16, 64G/卡 | HCU0 占用 89%（多为 KV 预留），CUDA graph 67/67，prompt 240 tok/s |
| 7B bf16 | ~14 GiB | TP=1 需 ≥ 24G 卡 | 预估：14 + 2 + KV |
| 70B bf16 | ~140 GiB | TP=4×64G 可行 | 35G/卡 + 开销，KV 偏紧建议 TP=8 或量化 |
| DeepSeek V4-Flash-INT8 | ~70 GiB 级 | 4×A800-80G TP8/EP8 | 另一栈（A卡），仅参考量级 |

## 注意事项

- **先查已被占用的显存**：节点上常驻其他服务时，`torch.cuda.mem_get_info()` 的空闲值才是真可用
- 多模态模型的 vision encoder 权重也在总大小里（文件清单已含），但运行时 activation 峰值更高，
  开销按 2 GiB 已偏保守
- `max_model_len` 拉大只影响 KV 消耗（并发换上下文），不影响权重
- RDU/多机 TP 超过 8 卡时，NCCL/HCCL 通信开销上升，TP 不是越大越好（优先考虑量化或 EP）
