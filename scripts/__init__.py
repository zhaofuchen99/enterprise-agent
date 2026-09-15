"""独立脚本集合。

作为包导入是为了让测试能直接 `from scripts.check_layering import ...`，
而不是用 importlib 手工加载文件（后者不会把模块注册进 `sys.modules`，
`@dataclass` 在解析注解时会因此失败）。
"""
