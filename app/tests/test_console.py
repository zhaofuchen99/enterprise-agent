"""演示控制台（`app/static/index.html`）的静态页与它那点 JS。

## 为什么这里会出现一个"跑 node"的用例

页面里的 JS **没有任何其它自动化覆盖**，而它恰好出过一个真缺陷：
`miniMarkdown` 只给标题与列表做了行内替换，于是答案段落里的 `**加粗**`
原样带着星号显示出来。这类问题在浏览器里一眼可见，但没有浏览器自动化时
它只能靠这种"把脚本抽出来、用 node 跑断言"的土办法挡住。

**node 不在时跳过，不让 `make check` 因此变红**：这台开发机与 CI 都不保证有它，
而"页面的 JS 没被验证"与"后端坏了"是两件事，混在一起会让门禁失去意义。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.main import WEB_DIR

_PAGE = WEB_DIR / "index.html"
#: 从页面里抠出 `<script>` 正文。**用正则而不是 HTML 解析器**：
#: 这是本项目自己的单文件页面，形状由本仓库控制，引一个解析器不值当。
_SCRIPT = re.compile(r"<script>\n(.*?)\n</script>", re.S)

#: 在 node 里跑的最小断言集：**先转义再渲染**这条顺序是它唯一真正要守的东西
#: （反过来就是给自己开一个注入口子），其余几项防的是"渲染器整个不工作"。
_NODE_CHECKS = """
const assert = (ok, msg) => { if (!ok) { console.error("失败：" + msg); process.exit(1); } };
assert(escapeHtml('<img src=x onerror=alert(1)>').includes("&lt;img"), "转义没生效");
assert(miniMarkdown("**粗体**").includes("<strong>粗体</strong>"), "段落里的加粗没渲染");
assert(miniMarkdown("# 标题").includes("<h3>标题</h3>"), "标题没渲染");
assert(miniMarkdown("- 甲\\n- 乙").includes("<ul>"), "列表没渲染");
assert(miniMarkdown("`code`").includes("<code>code</code>"), "行内代码没渲染");
const xss = miniMarkdown("<script>alert(1)</script> 与 **粗体**");
assert(!xss.includes("<script>"), "注入口子：原始标签进了 HTML");
assert(xss.includes("<strong>粗体</strong>"), "转义把 Markdown 一起吃掉了");
console.log("ok");
"""


@pytest.fixture
def page() -> str:
    assert _PAGE.is_file(), f"演示控制台缺失：{_PAGE}"
    return _PAGE.read_text(encoding="utf-8")


def _node(args: list[str]) -> subprocess.CompletedProcess[str]:
    """跑一次 node。**命令与参数都是本文件拼的**，没有外部输入。"""
    node = shutil.which("node") or "node"  # 调用方已跳过"没有 node"的情形
    return subprocess.run(  # noqa: S603 - 参数由本文件构造，无外部输入
        [node, *args], capture_output=True, text=True, timeout=30, check=False
    )


def _syntax_check(script: str, tmp_dir: Path) -> subprocess.CompletedProcess[str]:
    """语法检查必须**落成文件再 `node --check`**。

    直接 `node -e <脚本>` 是**执行**而不是检查——页面脚本在顶层就访问
    `document`，于是"语法没问题"会以 `ReferenceError: document is not defined`
    的形式失败。实测踩到：第一版就是这么写的，报错指向的却是 DOM 桩没打好。
    """
    path = tmp_dir / "console.js"
    path.write_text(script, encoding="utf-8")
    return _node(["--check", str(path)])


def _stub_dom(script: str) -> str:
    """把页面脚本套上 DOM 桩，导出两个纯函数供断言使用。"""
    return (
        """
const stub = () => ({ appendChild(){}, addEventListener(){}, scrollTop: 0,
                      scrollHeight: 0, style: {} });
globalThis.document = { getElementById: () => stub(), createElement: () => stub() };
globalThis.performance = { now: () => 0 };
globalThis.fetch = async () => ({ ok: true, json: async () => ({ code: "OK", data: {} }) });
globalThis.EventSource = class { addEventListener() {} close() {} };
globalThis.alert = () => {};
"""
        + script
        + "\n"
        + _NODE_CHECKS
    )


async def test_root_redirects_to_the_console(client: AsyncClient) -> None:
    """根路径进控制台：面试演示时少记一个地址。

    **它是 307 而不是 200**：控制台是被挂载的静态目录（`/ui/`），
    根路径只是一个人口，两者不是同一个资源。
    """
    response = await client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


async def test_console_page_is_served(client: AsyncClient) -> None:
    """页面本体可达，且是 `text/html`。

    挂载写错（路径、`html=True` 漏了）时的症状是 404，而 404 很容易被
    当成"页面还没做"——这条用例把它钉成"装配坏了"。
    """
    response = await client.get("/ui/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "企业智能数据分析" in response.text


def test_console_page_has_no_external_dependencies(page: str) -> None:
    """**零外部依赖**：不引 CDN、不引框架。

    引一个 CDN 的代价是"演示时如果没网，页面白屏"——而面试现场的网络
    不归我们管。页面要展示的东西全都是接口现成的，没有理由再拉一个库。
    """
    assert "http://" not in page.replace("http://www.w3.org", "")  # 只放过 SVG 命名空间
    assert "https://" not in page
    assert "<script src=" not in page
    assert "<link" not in page


@pytest.mark.skipif(shutil.which("node") is None, reason="本机没有 node，跳过页面 JS 的检查")
def test_console_script_matches_what_the_browser_would_run(page: str, tmp_path: Path) -> None:
    """抽出 `<script>` 的正文，用 node 跑一遍：**语法 + 那几条渲染断言**。

    故意**不引 jsdom/playwright**：为两个纯函数装一套浏览器环境不值得。
    这里验的是它们的行为，浏览器里的布局与事件绑定仍未自动化——如实写在
    模块 docstring 里，不假装覆盖到了。
    """
    match = _SCRIPT.search(page)
    assert match is not None, "页面里没有找到 <script> 块"

    syntax = _syntax_check(match.group(1), tmp_path)
    assert syntax.returncode == 0, f"页面 JS 语法错误：{syntax.stderr[:400]}"

    behaviour = _node(["--input-type=module", "-e", _stub_dom(match.group(1))])
    assert behaviour.returncode == 0, f"页面 JS 断言失败：{behaviour.stderr[:400]}"
    assert behaviour.stdout.strip() == "ok"


def test_the_static_dir_constant_points_at_a_real_file() -> None:
    """`WEB_DIR` 由 `__file__` 推出来（见 `main.py` 的说明），不是工作目录相对路径。

    写成相对路径的话，换一种启动方式就会 404——而那时看起来像"页面没做"。
    """
    assert WEB_DIR.is_dir()
    assert _PAGE.is_file()
