# 让适配模块在「主进程 + EngineCore spawn 子进程」里都生效。
# 生效条件：本文件所在目录在 PYTHONPATH 里（python 启动时 site 模块会自动 import sitecustomize）。
#
# 注意：这里千万不要让异常冒出去，否则这台机器上**所有** python 进程都会起不来。
try:
    import my_pooling_adapter  # noqa: F401  <改：换成你的适配模块名，import 即完成注册>
except Exception as _e:
    import sys

    print(f"[sitecustomize] 池化适配模块加载失败，vLLM 会认不出自定义 arch: {_e!r}",
          file=sys.stderr)
