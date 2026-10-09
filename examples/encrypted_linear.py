"""示例：对单个线性层做密态计算。

本示例是密态推理的最小验证单元：
    先跑通「加密向量 × 编码权重 → 密文输出 → 解密」，
    再与明文矩阵乘的结果做误差比对，
    确认 CKKS 精度满足要求后，才逐步扩展到完整 Transformer 层。

演示目标（待实现）:
    1. 用 fhellm.he 创建 CKKS 上下文并加密输入向量
    2. 用 fhellm.models 把线性层权重编码为 HE 兼容形式（明文编码）
    3. 做密文-明文矩阵乘（HE linear），得到密文输出
    4. 解密并与 torch.nn.Linear 的明文结果对照

运行: ``python examples/encrypted_linear.py``
"""

from fhellm.utils import load_config


def main():
    # 加载配置确认 HE 参数可用，后续步骤依赖这些参数
    cfg = load_config("configs/default.yaml")
    print(f"[example] scheme={cfg['he']['scheme']}, scale={cfg['he']['global_scale']}")
    # TODO: 接入 fhellm.he + fhellm.models.layers


if __name__ == "__main__":
    main()
