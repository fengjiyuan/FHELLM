"""FHELLM: 大模型密态推理框架（基于同态加密的 LLM 推理）。

项目目标:
    在不暴露用户输入与模型权重的前提下完成大模型推理。
    核心技术路线为同态加密（Homomorphic Encryption, HE），
    支持对密文直接进行矩阵乘、非线性近似等运算。

模块组织:
    - fhellm.he         : HE 原语（CKKS 上下文、编解码、密文运算）
    - fhellm.models     : 模型适配（加载 HF 模型、HE 兼容层、量化）
    - fhellm.engine     : 推理引擎（编排加密→密态计算→解密流水线）
    - fhellm.protocols  : 部署协议（client-server 交互模式）
    - fhellm.utils      : 通用工具（配置加载、张量处理、日志）

快速上手:
    >>> import fhellm
    >>> fhellm.__version__
    '0.1.0'
"""

__version__ = "0.1.0"
