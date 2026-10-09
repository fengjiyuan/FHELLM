"""命令行入口。

安装后可通过 ``fhellm`` 命令调用（见 pyproject.toml 的 [project.scripts]）。
当前为骨架，后续接入推理 / 密钥生成 / 加密等子命令。

用法:
    fhellm --version        # 查看版本
    fhellm run              # 运行密态推理（待实现）
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。

    Args:
        argv: 参数列表，None 时取 sys.argv[1:]（argparse 默认行为）。

    Returns:
        进程退出码，0 表示正常。
    """
    parser = argparse.ArgumentParser(
        prog="fhellm", description="FHELLM: 大模型密态推理框架"
    )
    parser.add_argument("-V", "--version", action="version", version="fhellm 0.1.0")
    sub = parser.add_subparsers(dest="cmd")

    # 占位：后续接入推理 / 密钥生成 / 加密等子命令
    sub.add_parser("run", help="运行密态推理")

    args = parser.parse_args(argv)
    # 未给出子命令时打印帮助，而非静默退出
    if args.cmd is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
