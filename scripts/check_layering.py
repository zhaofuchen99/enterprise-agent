#!/usr/bin/env python3
"""分层约束检查 —— CI 门禁（开发流程 5.2 / 7.3）。

为什么用脚本而不是人工 review：这两条约束一旦破坏，影响要到运行时才暴露
（API 进程启动变慢、内存翻倍、Worker 无法独立扩缩容），review 靠不住。

检查项：
  L1  app/main.py 与 app/api/** 不得 import agent.graph / agent.nodes.* / tools.*
  L2  app/worker.py 与 app/agent/**、app/tools/** 不得 import fastapi
  L3  app/domain/** 不得依赖 fastapi 或具体数据库客户端（保持领域纯净）

退出码 0 表示通过，1 表示存在违规。
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent / "app"

#: L3 中视为「具体数据库/外部客户端」的模块前缀
DB_CLIENTS: tuple[str, ...] = (
    "sqlalchemy",
    "asyncmy",
    "pymysql",
    "aiomysql",
    "redis",
    "pymilvus",
    "aioboto3",
    "boto3",
)

FASTAPI: tuple[str, ...] = ("fastapi", "starlette")
AGENT_ORCHESTRATION: tuple[str, ...] = ("app.agent.graph", "app.agent.nodes", "app.tools")


@dataclass(frozen=True)
class Violation:
    rule: str
    path: Path
    lineno: int
    imported: str
    reason: str

    def render(self) -> str:
        try:
            location = str(self.path.relative_to(APP_ROOT.parent))
        except ValueError:
            location = str(self.path)
        return (
            f"  [{self.rule}] {location}:{self.lineno}  import {self.imported}"
            f"\n        {self.reason}"
        )


def _module_name(path: Path) -> str:
    """app/api/chat.py -> app.api.chat"""
    rel = path.relative_to(APP_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(module: str, level: int, name: str | None, *, is_package: bool) -> str:
    """把相对 import 还原成绝对模块名。

    层级语义取决于当前文件是不是包：
      - 包 `app/api/__init__.py`（module=`app.api`）中，`from . import x`（level=1）指向 `app.api`；
      - 模块 `app/api/chat.py`（module=`app.api.chat`）中，`from . import x` 指向 `app.api`。
    少算一层会让违规模块名对不上前缀，检查器被静默绕过。
    """
    parts = module.split(".")
    depth = len(parts) - level + (1 if is_package else 0)
    base = parts[:depth] if depth > 0 else []
    if name:
        base = [*base, name]
    return ".".join(base)


def _iter_imports(tree: ast.AST, module: str, *, is_package: bool) -> list[tuple[int, str]]:
    """返回 (行号, 绝对模块名) 列表。

    `from X import Y` 同时产出 `X.Y` 与 `X`：前者能命中 `app.agent.graph` 这类
    深层禁用模块，后者保证 `import` 整体也被检查到。
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.append(
                    (
                        node.lineno,
                        _resolve_relative(module, node.level, node.module, is_package=is_package),
                    )
                )
            elif node.module:
                found.append((node.lineno, node.module))
                for alias in node.names:
                    found.append((node.lineno, f"{node.module}.{alias.name}"))
    return found


#: 规则编号 -> 违规说明。命中的是第一条匹配的规则（同一次 import 只报一条）。
RULES: dict[str, str] = {
    "L1": "API 进程不得加载 Agent 编排实现与 Tool，否则启动变慢、内存翻倍",
    "L2": "Worker 进程不得依赖 Web 框架，否则无法独立扩缩容",
    "L3": "domain 层必须保持纯净，不得依赖 Web 框架或具体数据库客户端",
}


def _matches(imported: str, prefixes: tuple[str, ...]) -> bool:
    return any(imported == p or imported.startswith(f"{p}.") for p in prefixes)


def _hit_rule(
    imported: str, is_api_side: bool, is_worker_side: bool, is_domain: bool
) -> str | None:
    """返回命中的规则编号；无违规返回 None。"""
    if is_api_side and _matches(imported, AGENT_ORCHESTRATION):
        return "L1"
    if is_worker_side and _matches(imported, FASTAPI):
        return "L2"
    if is_domain and (_matches(imported, FASTAPI) or _matches(imported, DB_CLIENTS)):
        return "L3"
    return None


def check_file(path: Path) -> list[Violation]:
    return check_source(
        path.read_text(encoding="utf-8"),
        _module_name(path),
        display_path=path,
        is_package=path.name == "__init__.py",
    )


def check_source(
    source: str,
    module: str,
    display_path: Path | None = None,
    *,
    is_package: bool = False,
) -> list[Violation]:
    """检查一段源码。`module` 是绝对模块名，决定适用哪些规则。"""
    path = display_path or Path(f"<{module}>")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Violation("SYNTAX", path, exc.lineno or 0, "", f"无法解析：{exc.msg}")]

    violations: list[Violation] = []
    seen: set[tuple[str, int]] = set()
    is_api_side = module == "app.main" or module.startswith("app.api")
    is_worker_side = (
        module == "app.worker" or module.startswith("app.agent") or module.startswith("app.tools")
    )
    is_domain = module.startswith("app.domain")

    for lineno, imported in _iter_imports(tree, module, is_package=is_package):
        # `from fastapi import Depends` 会同时产出 fastapi.Depends 与 fastapi，
        # 两条都命中同一条规则；按 (规则, 行号) 去重，避免同一行报两次。
        if (rule := _hit_rule(imported, is_api_side, is_worker_side, is_domain)) is None:
            continue
        if (rule, lineno) in seen:
            continue
        seen.add((rule, lineno))

        reason = RULES[rule]
        violations.append(Violation(rule, path, lineno, imported, reason))
    return violations


def main() -> int:
    if not APP_ROOT.is_dir():
        print(f"找不到 app 目录：{APP_ROOT}", file=sys.stderr)
        return 1

    violations: list[Violation] = []
    checked = 0
    for path in sorted(APP_ROOT.rglob("*.py")):
        checked += 1
        violations.extend(check_file(path))

    if violations:
        print(f"分层约束检查未通过（扫描 {checked} 个文件，{len(violations)} 处违规）：")
        for violation in violations:
            print(violation.render())
        return 1

    print(f"分层约束检查通过（扫描 {checked} 个文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
