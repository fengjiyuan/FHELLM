"""模型适配层。

加载 HuggingFace 模型，将标准层替换为 HE 兼容层，
并处理权重/激活量化以适配密态运算。
"""

from fhellm.models.plaintext import GenerateResult, PlaintextLLM

__all__ = ["PlaintextLLM", "GenerateResult"]
