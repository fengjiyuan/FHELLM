"""冒烟测试。

只验证最基础的可导入性（importability）与版本号，
不涉及模型加载或 HE 运算——这些在后续模块实现后再加专项测试。
运行: ``pytest -q``
"""


def test_import():
    """顶层包可导入且版本号正确。"""
    import fhellm

    assert fhellm.__version__ == "0.1.0"


def test_subpackages_import():
    """五个子包均可无副作用导入。"""
    import fhellm.engine  # noqa: F401
    import fhellm.he  # noqa: F401
    import fhellm.models  # noqa: F401
    import fhellm.protocols  # noqa: F401
    import fhellm.utils  # noqa: F401
