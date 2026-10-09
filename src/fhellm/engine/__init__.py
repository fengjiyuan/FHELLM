"""推理引擎。

负责编排一次完整的密态推理流水线:

    明文 prompt
        ↓ tokenize
    token id 序列
        ↓ embed + 加密（client 侧）
    密文激活
        ↓ 逐层密态前向传播（server 侧，不接触明文）
    密文 logits
        ↓ 返回 client 侧解密 + argmax + 解码
    明文输出

引擎职责:
    - 管理 HEContext 的生命周期（密钥生成、参数选择）
    - 按模型结构调度各层的密态算子（Linear / LayerNorm / GELU 近似等）
    - 控制噪声预算（noise budget），在必要时插入 rescale / relinearize
    - 与 fhellm.protocols 配合，完成 client-server 消息往返
"""
