"""同态加密原语层。

本模块封装底层 HE 库（TenSEAL / Microsoft SEAL）的细节，
对上层提供统一的加密、解密、密文运算接口。

支持的 HE 方案:
    - CKKS: 适用于近似计算（浮点向量），是大模型密态推理的首选方案。
            支持加法、乘法、旋转（rotation）等运算，但有累积噪声，
            需通过 rescale / bootstrapping 控制精度衰减。
    - BFV : 适用于精确整数运算，可用于 token id 等离散值。

核心抽象（待实现）:
    - HEContext       : HE 上下文（密钥、参数、evaluator）
    - PlainTensor     : 明文编码/解码（向量 → 多项式 slot）
    - CiphertextTensor: 密文张量，支持 +、*、rot 等运算

设计原则:
    对上层只暴露与 torch.Tensor 语义接近的接口，
    屏蔽 poly_modulus_degree / coeff_modulus 等密码学参数。
"""
