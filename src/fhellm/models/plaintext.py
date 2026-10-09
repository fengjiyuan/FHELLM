"""明文大模型推理基线。

本模块是密态推理项目的对照基准（baseline）：
    先跑通标准明文 HuggingFace 模型，记录其输出与耗时；
    后续每把一个层改成密态计算，都用本基线的输出做数值比对，
    确保加密 / 近似引入的误差在可接受范围内。

使用方式:
    >>> from fhellm.models import PlaintextLLM
    >>> llm = PlaintextLLM("gpt2-medium")
    >>> result = llm.generate("The capital of China is")
    >>> result.output
    ' the capital of China.'
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerateResult:
    """单次生成结果，用于后续与密态推理输出做比对。

    Attributes:
        prompt:    原始输入文本。
        output:    仅模型新生成的部分（不含 prompt），便于逐 token 比对。
        full_text: prompt + output 拼接，便于人工阅读。
        elapsed_s: 从 model.generate() 调用到返回的墙钟耗时（秒），
                   含 GPU kernel 执行但不含模型加载。
        device:    实际执行设备（'cuda' / 'cpu'）。
    """

    prompt: str
    output: str          # 仅模型新生成的部分
    full_text: str       # prompt + output
    elapsed_s: float
    device: str


class PlaintextLLM:
    """标准明文因果语言模型封装。

    封装 HuggingFace ``AutoModelForCausalLM`` 的加载与生成流程，
    对外提供统一的 ``generate()`` 接口，返回 :class:`GenerateResult`。

    设计要点:
        - GPU 优先：默认在 CUDA 上以 float16 运行以节省显存与加速；
          无 GPU 时退回 CPU + float32（HE 计算通常也在 CPU）。
        - 推理模式：加载后立即 ``model.eval()`` 关闭 dropout，
          生成时用 ``@torch.inference_mode()`` 关闭梯度计算。
        - 作为基线：不改动模型结构，不做量化，保持参考输出的纯净性。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-0.5B",
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """初始化模型与 tokenizer。

        Args:
            model_name: HuggingFace Hub 上的模型标识，如 ``"gpt2-medium"``。
            device:     指定设备；None 时自动选择（有 GPU 用 cuda）。
            dtype:      模型权重精度；GPU 默认 float16 节省显存，
                        CPU 强制 float32（float16 在 CPU 上算子支持不全）。
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # CPU 上 float16 算子覆盖不全，退回 float32 保证正确性
        self.dtype = dtype if self.device == "cuda" else torch.float32

        print(f"[plain] loading {model_name!r} on {self.device} ...")
        # trust_remote_code=True 允许加载含自定义代码的模型（如 Qwen）
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()  # 切换到推理模式，关闭 dropout
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[plain] ready, {n_params:.1f}M params, dtype={self.dtype}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
    ) -> GenerateResult:
        """对给定 prompt 执行自回归生成。

        Args:
            prompt:        输入文本。
            max_new_tokens: 最多生成的新 token 数（不含 prompt）。
            do_sample:      False=贪心解码（确定性，适合基线比对）；
                            True=采样解码（需先设 temperature/top_p）。

        Returns:
            :class:`GenerateResult`，含输出文本与耗时。

        Note:
            - 贪心解码（do_sample=False）适合做基线比对，因为输出确定性可复现。
            - pad_token_id 设为 eos_token_id：部分模型（如 GPT-2）无 pad token，
              用 eos 代替避免 generate() 报 warning。
        """
        # tokenize 并移到目标设备；return_tensors="pt" 返回 torch.Tensor
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # 记录 prompt 长度，用于后续从输出中切出"新生成"部分
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        # model.generate() 自回归生成；out_ids 形状 [batch, prompt_len + new_len]
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - t0

        # 切出新生成的 token（去掉 prompt 部分），解码为文本
        new_ids = out_ids[0][prompt_len:]
        output = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        full_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return GenerateResult(prompt, output, full_text, elapsed, self.device)
