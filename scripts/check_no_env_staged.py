#!/usr/bin/env python3
"""禁止提交 .env 类密钥文件 —— pre-commit 门禁。

`.gitignore` 拦得住常规路径，但拦不住两件事：
  1. `git add -f .env`（-f 强制绕过忽略规则）
  2. 手动改 .gitignore 后误提交

所以这里从 **git 索引**直接读取暂存文件，而不是依赖 pre-commit 传入的文件名——
被 `.gitignore` 忽略的文件本来就不会被传给钩子，只有 `git add -f` 过的才会出现，
而那恰恰是需要拦下的情况。

内容层面的密钥（sk-、AKIA、ghp_ 等）由 gitleaks 钩子负责，本脚本只管文件名。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath

#: 允许提交的例外——它是模板，值为空或占位符
ALLOWED: frozenset[str] = frozenset({".env.example"})

#: 禁止提交的文件名前缀
FORBIDDEN_PREFIX = ".env"


def staged_files() -> list[str]:
    """列出已暂存的新增/修改/重命名文件。"""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def is_forbidden(path: str) -> bool:
    name = PurePosixPath(path).name
    if name in ALLOWED:
        return False
    return name == FORBIDDEN_PREFIX or name.startswith(f"{FORBIDDEN_PREFIX}.")


def main() -> int:
    try:
        files = staged_files()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 不在 git 仓库里（例如被单独调用）——不该因此阻塞提交
        return 0

    offenders = [path for path in files if is_forbidden(path)]
    if not offenders:
        return 0

    print("检测到暂存区里有 .env 类密钥文件，已阻止提交：", file=sys.stderr)
    for path in offenders:
        print(f"  - {path}", file=sys.stderr)
    print(
        "\n这些文件用于存放 API key，不得进入版本库。\n"
        "如果确认要提交（几乎不可能），用 git commit --no-verify 绕过。\n"
        "如果只是误暂存，执行：git reset <文件路径>",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
