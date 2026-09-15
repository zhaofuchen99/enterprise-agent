"""分层约束检查器自身的测试。

检查器是 CI 门禁，它自己失效比被检查的代码违规更危险——所以既验证
「真实代码树通过」，也验证「合成违规确实会被抓到」。
"""

from __future__ import annotations

import pytest

from scripts import check_layering as checker


def test_real_tree_passes() -> None:
    assert checker.main() == 0


@pytest.mark.parametrize(
    ("rule", "module", "source"),
    [
        ("L1", "app.api.chat", "from app.agent.graph import build_graph\n"),
        ("L1", "app.api.chat", "import app.tools.sql.executor\n"),
        ("L1", "app.main", "from app.agent.nodes import supervisor\n"),
        ("L2", "app.worker", "import fastapi\n"),
        ("L2", "app.agent.nodes.planner", "from fastapi import Depends\n"),
        ("L2", "app.tools.sql.executor", "from fastapi import HTTPException\n"),
        ("L3", "app.domain.task", "import sqlalchemy\n"),
        ("L3", "app.domain.evidence", "from fastapi import APIRouter\n"),
    ],
)
def test_violation_is_detected(rule: str, module: str, source: str) -> None:
    violations = checker.check_source(source, module)
    assert [v.rule for v in violations] == [rule]


@pytest.mark.parametrize(
    ("module", "source"),
    [
        # API 侧引用 agent 的纯数据结构（下方用相对导入验证边界不误伤）
        ("app.api.chat", "from app.services.task_runner import run_task\n"),
        ("app.api.chat", "from app.core.errors import AgentError\n"),
        # Worker 侧不得依赖 fastapi，但 pydantic 是允许的
        ("app.agent.nodes.planner", "from pydantic import BaseModel\n"),
        ("app.worker", "from app.tools.sql.validator import validate\n"),
        # 命中前缀但并非禁用模块，不应误报
        ("app.api.chat", "import app.agentless_helper\n"),
    ],
)
def test_allowed_imports_do_not_trigger(module: str, source: str) -> None:
    assert checker.check_source(source, module) == []


def test_relative_import_is_resolved() -> None:
    """相对导入必须还原成绝对模块名，否则边界检查可以被绕过。"""
    # app/api/chat.py 中的 `from ..agent.graph import ...` -> app.agent.graph
    violations = checker.check_source("from ..agent.graph import build_graph\n", "app.api.chat")
    assert [v.rule for v in violations] == ["L1"]


def test_relative_import_inside_package_counts_one_level_up() -> None:
    """包自身的 __init__ 里 level=1 指向该包，而非其父包。

    少算这一层会让模块名解析错位，违规模块名对不上前缀，检查器被静默绕过。
    """
    # app/domain/__init__.py 中的 `from fastapi import ...`（绝对导入）必须被 L3 抓到
    violations = checker.check_source(
        "from fastapi import APIRouter\n", "app.domain", is_package=True
    )
    assert [v.rule for v in violations] == ["L3"]

    # app/api/__init__.py 中的 `from .chat import router` -> app.api.chat，不违规
    assert checker.check_source("from .chat import router\n", "app.api", is_package=True) == []


def test_duplicate_import_targets_report_once() -> None:
    """`from X import Y` 同时产出 X 与 X.Y，同一行只能报一条。"""
    violations = checker.check_source("from fastapi import Depends, HTTPException\n", "app.worker")
    assert len(violations) == 1
