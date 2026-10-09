"""通用工具：配置加载、张量处理、日志等。

对外暴露:
    - load_config: 从 YAML 文件加载配置字典
"""

from fhellm.utils.config import load_config

__all__ = ["load_config"]
