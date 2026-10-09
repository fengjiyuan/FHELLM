"""明文大模型推理基线脚本。

用法（已激活 venv）:
    # 直连 HuggingFace
    python scripts/run_inference.py --prompt "你好，请用一句话介绍自己"

    # 大陆网络走镜像（避免下载超时）
    HF_ENDPOINT=https://hf-mirror.com python scripts/run_inference.py --prompt "你好"
"""

import argparse

from fhellm.models import PlaintextLLM
from fhellm.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(description="FHELLM 明文推理基线")
    p.add_argument("--prompt", required=True, help="输入文本")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default=None, help="覆盖配置中的模型名")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = args.model or cfg["model"]["name"]
    llm = PlaintextLLM(model_name=model_name)
    result = llm.generate(
        args.prompt,
        max_new_tokens=cfg["infer"]["max_new_tokens"],
        do_sample=cfg["infer"]["do_sample"],
    )

    print("\n" + "=" * 60)
    print("PROMPT :", result.prompt)
    print("OUTPUT :", result.output)
    print("-" * 60)
    print(f"device={result.device}, elapsed={result.elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
