# L4-workflow-ai

AI 工作流工具集：把重复的「调研 / 复现 / 记录」动作沉淀为可复用 skill，便于在 WorkBuddy 中一键调用或二次开发。

> 各 skill 的详细作用、上线时间、复用方式与坑点见 [tools-explain.md](tools-explain.md)。

## Skill 索引

| Skill | 一句话说明 | 入口（SKILL.md） |
|-------|-----------|------------------|
| [feishu-doc-write](skills/feishu-doc-write/SKILL.md) | 飞书文档程序化写入：把复现步骤 / Bug 记录 / 性能对比可靠落盘 | `skills/feishu-doc-write/SKILL.md` |
| [dcu-vllm-deploy](skills/dcu-vllm-deploy/SKILL.md) | Hygon DCU 通用 vLLM 推理服务部署：模型无关、流程固化 | `skills/dcu-vllm-deploy/SKILL.md` |
| [dtk-ai-package-search](skills/dtk-ai-package-search/SKILL.md) | DCU AI 生态包离线检索：对 DTK/Python/torch 版本出 pip 直链 | `skills/dtk-ai-package-search/SKILL.md` |
| [vllm-pooling-model-adapter](skills/vllm-pooling-model-adapter/SKILL.md) | 把 HF 的 embedding / 多模态检索模型接进 vLLM pooling runner，附四层数值对齐法 | `skills/vllm-pooling-model-adapter/SKILL.md` |
| [dcu-vllm-backend-triage](skills/dcu-vllm-backend-triage/SKILL.md) | DCU 上 vLLM attention 后端选型：`_rocm_C` / `vllm_flash_attn` 缺失时该用哪个后端 | `skills/dcu-vllm-backend-triage/SKILL.md` |

## 三者关系
- `dtk-ai-package-search` 负责「装什么版本」
- `dcu-vllm-deploy` 负责「怎么把服务跑起来」
- `feishu-doc-write` 负责「把结论沉淀成团队文档」

## 两条正交的排障线
- `dcu-vllm-backend-triage` 解决「服务根本没起来」——后端/编译扩展缺失
- `vllm-pooling-model-adapter` 解决「服务起来了但向量不对」——池化语义没对齐
