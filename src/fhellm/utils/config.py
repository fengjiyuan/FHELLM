"""配置加载工具。

项目配置统一用 YAML 管理（见 configs/default.yaml），
运行时通过本模块读取为嵌套字典，供 engine / models / he 各层使用。
"""

import os
from pathlib import Path

import yaml


def load_config(path: str | os.PathLike) -> dict:
    """加载 YAML 配置文件并返回字典。

    Args:
        path: YAML 文件路径，相对路径以当前工作目录解析。

    Returns:
        嵌套字典，顶层 key 为 ``he`` / ``model`` / ``engine`` / ``infer``。

    Raises:
        FileNotFoundError: 文件不存在时抛出。

    Example:
        >>> cfg = load_config("configs/default.yaml")
        >>> cfg["he"]["scheme"]
        'ckks'
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    # 用 safe_load 避免 YAML 中的任意代码执行（如 !!python/object）
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
