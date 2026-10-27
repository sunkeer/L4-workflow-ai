"""vLLM 池化模型适配模块模板。

复制本文件，按注释替换「<>」处的内容。核心只有三步：
  1) 继承 vLLM 已支持的**同家族生成模型**；
  2) 打上 is_pooling_model 标记并声明默认池化方式；
  3) 注册 arch 名，让 config.json 里的 <%= %> 类名能被 resolve。

父类怎么选：
  SP=$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
  grep -n "is_pooling_model = True" -B6 $SP/model_executor/models/*.py
  # 样板：nemotron_vl.py::LlamaNemotronVLForEmbedding（VLM+池化）
  #       bert.py / colqwen3.py（纯文本 / late-interaction）

池化方式必须和模型训练口径一致，写错不报错、只掉精度：
  LAST  —— 取最后一个 token（多数 decoder-only embedding 模型）
  CLS   —— 取首 token
  MEAN  —— 全局平均
"""

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.interfaces_base import VllmModelForPooling

# <改：换成目标模型的同家族生成类，模块与类名照抄 vLLM 里的实现>
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.registry import ModelRegistry

# <改：config.json 里 architectures 声明的类名>
ARCH_NAME = "Qwen3VLForEmbedding"
# <改：模块文件名，不带 .py>
MODULE_NAME = "my_pooling_adapter"


class AdaptedForEmbedding(Qwen3VLForConditionalGeneration, VllmModelForPooling):
    """<改：写清这个类服务哪几个模型、池化口径从哪来>"""

    # ① 让 ModelRegistry.is_pooling_model() 认它（实现上就是 getattr 读这个类属性）
    is_pooling_model = True

    # ② 默认池化方式，与模型训练口径对齐
    default_seq_pooling_type = "LAST"
    default_tok_pooling_type = "ALL"

    # ③ 签名照抄父类（prefix 默认值可能是 "" 或 "model"，别写错）
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None, "池化模式必须提供 pooler_config（加 --runner pooling）"
        self.pooler = DispatchPooler.for_embedding(pooler_config)


def register() -> None:
    """用字符串形式注册，避免 import 期初始化 CUDA。"""
    ModelRegistry.register_model(ARCH_NAME, f"{MODULE_NAME}:AdaptedForEmbedding")


register()
