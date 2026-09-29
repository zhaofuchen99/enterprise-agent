"""演示与评测共用的**调用层**与**判据**。

## 为什么要有这个模块

`make demo`（现场演示）与 `make eval-agent`（回归评测）跑的是同一条链路、
同一套 `expect_*` 词汇，区别只在**数据集的规模与输出形态**：

| | `demo.py` | `eval_agent.py` |
|---|---|---|
| 数据集 | 9 条固化演示题 | 20 条评测题（分类） |
| 输出 | 打印答案正文，让人看得见 | 按类别报通过率 |
| 口令 | 一次性问题，现场看 | 回归门禁，看退步 |

**判据只能有一份**。两份判据的漂移方式是"演示判对、评测判错"（或反过来），
而同一条问题在两个入口给出不同结论时，没有人能说出哪个才对——
这正是 CLAUDE.md 反复记的那类"两处各说各话"。

## 判定与"读"分开

- `login` / `post` / `get` / `wait_for_task`：**怎么跟 Agent API 说话**；
- `amounts` / `close` / `missing_amounts`：**怎么从答案里读出一个数**；
- `evaluate`：**这条用例算不算通过**。

`--ask` 探针模式（见两个 runner）用的也是这三层，所以它打印出来的事实
与判定用的是同一份解析——**否则"先跑一遍再固化"会先猜一遍**。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from app.repositories.user_repo import DEMO_ACCOUNTS

#: 演示账号。**从 `DEMO_ACCOUNTS` 取而不是在这里再抄一遍口令**：
#: 抄一遍就有两处会漂移，而漂移的表现是"脚本登不上"——
#: 那时要先怀疑是不是口令改了，而它其实只是改了另一边。
#:
#: **默认用 admin**：它的数据范围不限（TBC-03），于是不会因为
#: "华东之外的数据被权限挡掉"而出现看不懂的空结果——那不是系统坏了，
#: 但解释它要花掉半分钟。受限账号的行为由用例自己的 `account` 字段选。
DEMO_USERNAME = "admin"
#: `{username: password}`，同样从 `DEMO_ACCOUNTS` 取（见上）
_PASSWORDS = {username: password for username, password, _, _ in DEMO_ACCOUNTS}

#: 单条任务最多等多久。图的正常耗时在 30–90 秒（含两次模型调用与一次检索），
#: 给 180 秒是留足余量又能在真的卡住时很快报出来。
TASK_TIMEOUT_SECONDS = 180
POLL_INTERVAL_SECONDS = 3

#: 用 `httpx`（项目已有的依赖）而不是 `urllib`：后者的错误类型零散
#: （`URLError` / `HTTPError` / `socket.error`），而这里只需要"连不上"与
#: "通了但报错"两类。脚本的异常处理越短越好——它出问题时
#: 正是在有人看着的时候。
_TIMEOUT = httpx.Timeout(30.0)

#: `{username: token}`。**按账号缓存**：一条用例可以选择用它自己的账号跑
#: （见用例的 `account` 字段），而反复登录既慢又没必要。
_TOKENS: dict[str, str] = {}


# ------------------------------------------------------------------ 调用层
def login(base: str, username: str) -> str:
    """登录并缓存 token。口令从 `DEMO_ACCOUNTS` 取，不在这里再抄一份。"""
    if username not in _PASSWORDS:
        # 账号写错时报出**有哪些账号**，而不是一个裸的 `KeyError`：
        # 这条路径只有改 YAML 时才会走到，而那时人正盯着 YAML 找拼写。
        raise SystemExit(f"账号 {username!r} 不存在，只有：{sorted(_PASSWORDS)}")
    if username not in _TOKENS:
        password = _PASSWORDS[username]
        _TOKENS[username] = post(
            f"{base}/api/auth/login", {"username": username, "password": password}
        )["data"]["access_token"]
    return _TOKENS[username]


def post(url: str, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
    response = httpx.post(url, json=payload, headers=_headers(token), timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def get(url: str, *, token: str) -> dict[str, Any]:
    response = httpx.get(url, headers=_headers(token), timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def wait_for_task(token: str, task_id: str, *, base: str) -> dict[str, Any]:
    """轮询到任务进终态，返回任务详情。

    终态是 `SUCCEEDED` / `FAILED` / `CANCELLED` 三个——`QUEUED` 与 `RUNNING`
    都还要等。**不区分是谁的错**：调用方拿到详情后自己判（演示要能看见失败，
    评测要把失败与"行为不符"分开报，见 `evaluate`）。
    """
    deadline = time.monotonic() + TASK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        detail: dict[str, Any] = get(f"{base}/api/agent/tasks/{task_id}", token=token)["data"]
        if detail.get("status") in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return detail
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"任务 {task_id} 在 {TASK_TIMEOUT_SECONDS} 秒内没有结束")


def ask(base: str, token: str, question: str) -> dict[str, Any]:
    """提交一个问题并等它跑完，返回任务详情。"""
    created = post(f"{base}/api/agent/chat", {"message": question}, token=token)["data"]
    return wait_for_task(token, created["task_id"], base=base)


def ask_turns(base: str, token: str, questions: list[str]) -> list[dict[str, Any]]:
    """按顺序问一串问题，**共用同一个会话**，返回每轮的任务详情。

    `conversation_id` 从**第一轮的响应**里取（由服务端生成），之后逐轮带上——
    这正是浏览器/前端要做的事，脚本里不自己造一个 ID：造 ID 就绕开了
    "服务端会不会返回它、下一轮认不认它"这两件事，而它们才是这条链路。

    ⚠️ **一轮失败不中断**：第二轮的失败往往正是"记忆没生效"的第一个症状
    （代词没被解析 → 缺前提 → 任务失败），把它掐掉会让演示只留下一句
    "连不上"之类的假象。每轮都等到终态，判据逐轮跑。
    """
    details: list[dict[str, Any]] = []
    conversation_id: str | None = None
    for question in questions:
        payload: dict[str, Any] = {"message": question}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        created = post(f"{base}/api/agent/chat", payload, token=token)["data"]
        conversation_id = str(created["conversation_id"])
        details.append(wait_for_task(token, created["task_id"], base=base))
    return details


# ------------------------------------------------------- 多轮用例的展开与执行
def case_turns(case: dict[str, Any]) -> list[dict[str, Any]]:
    """把一条用例展开成**逐轮的子用例**。

    单轮用例（绝大多数）只写 `question`；多轮用例写 `turns: [...]`，
    每一项可以覆写任意 `expect_*`——这是必须的，因为**该断言什么逐轮不同**：
    首轮没有上文可继承，中间轮要钉"口径继承对了"，末轮才钉答案里的数。

    展开成一串**结构相同的子用例**而不是另立一套多轮判据，是为了让
    `evaluate` 原样复用：多一份判据就多一处会与演示各说各话的地方（约定 87）。
    """
    turns = case.get("turns")
    if not turns:
        return [case]
    shared = {key: value for key, value in case.items() if key != "turns"}
    return [{**shared, **turn} for turn in turns]


def run_case(base: str, token: str, case: dict[str, Any]) -> list[dict[str, Any]]:
    """跑一条用例，返回**逐轮**的任务详情（单轮用例就是长度为 1 的列表）。"""
    turns = case_turns(case)
    if len(turns) == 1:
        return [ask(base, token, str(turns[0]["question"]))]
    return ask_turns(base, token, [str(turn["question"]) for turn in turns])


# ------------------------------------------------------------------ 读答案里的数
#: 数值断言里的数字。**带单位**——语料与答案里「万元」「亿元」是常态，
#: 不认单位会让一条正确的「11,196.70 万元」判成错的。
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(亿元|亿|万元|万|千元|千)?")

#: 中文数词的单位倍率
_UNITS: dict[str, Decimal] = {
    "亿": Decimal(10**8),
    "亿元": Decimal(10**8),
    "万": Decimal(10**4),
    "万元": Decimal(10**4),
    "千": Decimal(10**3),
    "千元": Decimal(10**3),
}

#: 数值断言的**相对**容差。
#:
#: 取 0.1% 是因为**单位换算必然带来舍入**：真实值 111,967,031.73 元
#: 写成「11,196.70 万元」只差 0.0003%。而真正的算错差得远不止这个量级——
#: 实测那次把 8 行明细当成区域合计，差了 46%。
#: 两者之间有五个数量级的空档，阈值取在哪儿都不影响判定，取一个能容下
#: 舍入的值即可。
NUMBER_TOLERANCE = Decimal("0.001")


def amounts(text: str) -> list[Decimal]:
    """答案里出现的所有金额（**按单位还原成基准值**）。

    不还原单位的话，「1.12 亿」与「111,967,031.73」会被当成两个不相关的数，
    而它们说的是同一件事——**断言会因为写法不同而误报**，
    那种红叉比没有断言更糟（它会让人开始忽略断言）。
    """
    found: list[Decimal] = []
    for raw, unit in _NUMBER.findall(text):
        try:
            value = Decimal(raw.replace(",", ""))
        except InvalidOperation:  # pragma: no cover - 正则已保证是数字
            continue
        found.append(value * _UNITS.get(unit, Decimal(1)))
    return found


def close(found: Decimal, expected: Decimal) -> bool:
    if expected == 0:
        return found == 0
    return abs(found - expected) / abs(expected) <= NUMBER_TOLERANCE


def missing_amounts(answer: str, expected: list[str]) -> list[str]:
    """`expect_numbers` 里**没有出现**在答案中的那些。

    配置写错（不是数字）时直接抛——那是**数据集自己的 bug**，
    静默跳过会让一条永远不会生效的断言看起来一直在通过。
    """
    found = amounts(answer)
    missing: list[str] = []
    for raw in expected:
        try:
            want = Decimal(str(raw).replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError(f"expect_numbers 不是数字：{raw!r}") from exc
        if not any(close(value, want) for value in found):
            missing.append(str(raw))
    return missing


# ------------------------------------------------------------------ 判据
@dataclass(frozen=True)
class Verdict:
    """一条用例的判定结果。

    `ok=False` 且 `unjudged=True` 表示**这条用例没有产生可判定的行为**——
    任务失败、或分析生成失败。它与"行为不符"必须分开：把故障说成不符，
    会让人去调 prompt，而真正要修的是别处。
    """

    ok: bool
    note: str
    unjudged: bool = False


def evaluate(case: dict[str, Any], detail: dict[str, Any]) -> Verdict:
    """按用例的 `expect_*` 判定。**每条只给一个结论**——多个失败点会让报告读不出主因。

    **任务本身失败时先报那个**：`MODEL_OUTPUT_INVALID` 会让 `steps` 为空，
    而"期望 rag 实际 []"看起来像"工具选错了"——失败指错了层，
    排查方向会跑到意图判定上去，而真正的原因是一次模型输出抖动。
    这类"报告指向错误的一层"是本项目反复踩到的一类问题。

    判据的取值来源是 `GET /api/agent/tasks/{id}`（17.2 的任务详情）：

    | 断言 | 读的字段 |
    |---|---|
    | `expect_sources` | `steps[].tool`（`status != PENDING`） |
    | `expect_conflicts` / `_count` / `_contain` | `conflicts`（代码写入的描述串） |
    | `expect_rejected` | `refused`（结构化字段，见 `AnalysisResult.refused`） |
    | `expect_clarification` | 步骤为空（澄清不产出步骤） |
    | `expect_numbers` | 答案正文里的金额，按单位还原 |
    | `expect_limitations_contain` | `limitations`（其中"代码生成的"那一部分才该被钉） |
    | `expect_review_*` | `review`（`ReviewResult`，两阶段审查的结论） |
    | `expect_plan_*` | `plan_revision` / `plan_deltas` / `steps[].origin` |
    | `expect_progress_decision` | `progress_decision`（`reflect` 的最终判定） |
    | `expect_resolved_contain` | `resolved_entities`（FR-CHAT-003 的代词解析结果） |
    """
    if detail.get("status") != "SUCCEEDED":
        code = detail.get("error_code")
        message = str(detail.get("error_message") or "")[:60]
        return Verdict(False, f"**任务未成功**：{code} - {message}", unjudged=True)

    steps = detail.get("steps") or []
    sources = {step["tool"] for step in steps if step["status"] != "PENDING"}
    conflicts = detail.get("conflicts") or []
    answer = detail.get("final_answer_md") or ""

    # **代词解析的断言排在最前面**，在任何提前 `return` 之前。
    #
    # 它验的是「上一轮的口径有没有被继承」，与这一轮走了哪几路、有没有拒答
    # **完全正交**。写在下面那些模式分支之后的话，它只在"没拒答也没澄清"
    # 的用例上生效——而"多轮没生效"最典型的表现恰恰是**判成缺前提去澄清**。
    # 那时这条断言永远不会被检查，用例却报"进入澄清 ✓"。
    if expected_entities := case.get("expect_resolved_contain"):
        actual = detail.get("resolved_entities") or {}
        wrong = {key: want for key, want in expected_entities.items() if actual.get(key) != want}
        if wrong:
            return Verdict(
                False,
                f"**上一轮口径没有被继承**：期望 {wrong}，实际解析出 {actual}"
                f"（resolved_question={detail.get('resolved_question')!r}）",
            )

    # 「模式」类断言：拒答与澄清是两条**互斥**的路径，走了它们就不再判
    # 工具选择 / 数值 / 冲突——那些字段在那条路径上没有意义（澄清一步都
    # 不跑，`steps` 必然是空的）。但**审查与模式正交**，所以它排在后面
    # 统一判，见 `_review_or`。
    if case.get("expect_rejected"):
        # **判据读 `refused` 字段，不认答案里的词**。
        #
        # 这里原先是一串字符串匹配（"没有找到"/"未能检索到"/…），它出过一次
        # 典型的漏判：同一条问题、同一份代码，模型说「没有找到」时判对，
        # 说「证据中没有…的条文」时判错——**四次里错一次**。判据在猜词，
        # 而演示现场出现一次红叉，代价远大于它省下的那点改动。
        if detail.get("refused") is True:
            return _review_or(case, detail, Verdict(True, "按 11.8 拒答（refused=true）"))
        # 分析生成失败时 `refused` 是 `False`（故障不是拒答），
        # 但此时报"没有拒答"会把一次故障说成一次编造——**失败指错了层**。
        # 这里认的是分析节点自己写下的那句，是**我们自己的字符串**，
        # 不是模型措辞，所以不含上面那种不确定性。
        if any("分析生成失败" in item for item in detail.get("limitations") or []):
            return Verdict(
                False, "**未能判定**——综合分析生成失败，这条不是拒答行为的结果", unjudged=True
            )
        return Verdict(False, "**没有拒答**——refused=false，而语料里没有这条")

    if case.get("expect_clarification"):
        if not steps:
            return _review_or(
                case, detail, Verdict(True, f"进入澄清：{answer.strip().splitlines()[0][:60]}")
            )
        return Verdict(False, f"**没有进入澄清**——直接跑了 {sorted(sources)}")

    # **缺省与空列表是两件事**：缺省表示"这条用例不关心走了哪几路"，
    # `[]` 才表示"确实不该调任何工具"。合成一个的话，一条只断审查结论的
    # 用例会被判成"工具选择不符"——而它压根没打算断言工具。
    # 缺省值取"不检查"而不是"检查为空"，是因为误报的代价更大：
    # 一条永远红着的用例会让整份评测报告失去可信度。
    if (expect_raw := case.get("expect_sources")) is not None:
        expect = set(expect_raw)
        if expect != sources:
            return Verdict(
                False, f"**工具选择不符**：期望 {sorted(expect)}，实际 {sorted(sources)}"
            )

    if case.get("expect_conflicts") and not conflicts:
        return Verdict(False, "**没有检出冲突**——而这条问题的两个来源数字本就不一致")

    # **条数是精确判据**，用来钉住"不该报的没报"——冲突检测的误报
    # 与漏报代价不同：漏报看不出来（答案照常给出），误报看得出来
    # （读者会问"这两行为什么能放在一起比"）。所以"零条"值得单独立一条用例。
    expected_count = case.get("expect_conflict_count")
    if expected_count is not None and len(conflicts) != expected_count:
        return Verdict(
            False,
            f"**冲突条数不符**：期望 {expected_count}，实际 {len(conflicts)}（{list(conflicts)}）",
        )

    if missing_conflicts := [
        wanted
        for wanted in (case.get("expect_conflict_contain") or [])
        if not any(wanted in item for item in conflicts)
    ]:
        return Verdict(False, f"**冲突描述里缺少**：{missing_conflicts}")

    if missing := missing_amounts(answer, case.get("expect_numbers") or []):
        return Verdict(False, f"**答案里没有出现期望的数值**：{missing}")

    if missing_limits := [
        item
        for item in (case.get("expect_limitations_contain") or [])
        if item not in " ".join(detail.get("limitations") or [])
    ]:
        return Verdict(False, f"**限制清单里缺少**：{missing_limits}")

    if (decision := case.get("expect_progress_decision")) and detail.get(
        "progress_decision"
    ) != decision:
        return Verdict(
            False, f"**演进判定不符**：期望 {decision}，实际 {detail.get('progress_decision')}"
        )

    if verdict := _check_review(case, detail.get("review")):
        return verdict

    if verdict := _check_plan(case, detail):
        return verdict

    return Verdict(
        True,
        f"走了 {sorted(sources)}" + (f"，检出 {len(conflicts)} 条冲突" if conflicts else ""),
    )


def _check_plan(case: dict[str, Any], detail: dict[str, Any]) -> Verdict | None:
    """任务循环的断言（详设 7.5 / 22.10）。**没写 `expect_plan_*` 就跳过**。

    22.10.5 明写循环类问题"至少 15 条，按 `loop_expectation` 校验"。
    本层的这几个字段是它的最小可判定集：

    | 字段 | 读的字段 | 回答的问题 |
    |---|---|---|
    | `expect_plan_revision` | `plan_revision` | **不该**下钻时是 0（精确） |
    | `expect_plan_revision_min` | `plan_revision` | **该**下钻时至少几次（下限） |
    | `expect_plan_delta_count` | `plan_deltas` | 演进了几次（精确，用于 0） |
    | `expect_extended_tools_contain` | `steps[].origin == EXTENDED` | 下钻到了哪一路（**包含**） |
    | `expect_max_steps` | `len(steps)` | 有没有越界 |
    | `expect_trigger_finding_linked` | `plan_deltas[].trigger_finding_id` | 反查得到发现吗 |

    ## 为什么"该演进"钉下限、"不该演进"钉精确

    演进有**两个触发源**：`reflect` 的确定性判定（SQL 空 → 补 RAG）与
    审查第二阶段的 `retry_target=expand`。前者是代码判的，每次必发生；
    **后者是模型判的，同一题两次跑可以不一样**——实测（2026-09-28）
    一条「查不到的期间」的问题跑出**两次**演进：第一步确定性补了 RAG，
    审查又判了一次"缺一整类信息"，模型路再加 3 步。

    所以"反正会演进"的那一类**只能钉下限与包含**；钉精确值等于把
    "模型这次怎么想的"固化成门禁，而随机红的门禁很快就会被忽略。
    反过来，"**不该**演进"那几条是稳的——单源查询与澄清压根没有
    可补的一路，模型再想加也没有由头，所以那里钉精确的 0。

    这与 `expect_sources`（精确）和 `expect_extended_tools_contain`
    （包含）的差别是同一条理由：**能精确的地方精确，模型插得进手的地方
    只钉方向**。

    `expect_trigger_finding_linked` 对应 22.10.3 的「每个 `origin=EXTENDED`
    的步骤都能反查到存在的 `finding_id`」——**它同时查两个方向**：
    有新增步骤的 delta 必须有触发源，而那个源必须真的在推理链里。
    只查前者的话，一个指向不存在 id 的值照样通过（而它看起来完全正常）。
    """
    keys = (
        "expect_plan_revision",
        "expect_plan_revision_min",
        "expect_plan_revision_max",
        "expect_plan_delta_count",
        "expect_extended_tools_contain",
        "expect_max_steps",
        "expect_trigger_finding_linked",
        "expect_plan_records_consistent",
    )
    if all(case.get(key) is None for key in keys):
        return None

    revision = detail.get("plan_revision") or 0
    deltas = detail.get("plan_deltas") or []
    steps = detail.get("steps") or []
    extended = {step["tool"] for step in steps if step.get("origin") == "EXTENDED"}

    if (wanted := case.get("expect_plan_revision")) is not None and revision != wanted:
        return Verdict(False, f"**计划演进次数不符**：期望 {wanted}，实际 {revision}")

    if (floor := case.get("expect_plan_revision_min")) is not None and revision < floor:
        return Verdict(False, f"**该演进却没有演进**：plan_revision={revision} < {floor}")

    # 上限是 22.10.4 那条预算共用的端到端形态：`reflect` 的演进与
    # `retry_target=expand` **共用同一份 `expansions_left`**（14.3 原文），
    # 两者之和不得超过配置值。绕过它只有一条路——某条路自己扣了别的预算，
    # 而"四类预算各自独立"正是 FR-REV-002 业务规则 1 的要求。
    if (ceiling := case.get("expect_plan_revision_max")) is not None and revision > ceiling:
        return Verdict(
            False, f"**演进次数超出预算**：{revision} > {ceiling}（两类演进共用同一份预算）"
        )

    if (wanted := case.get("expect_plan_delta_count")) is not None and len(deltas) != wanted:
        return Verdict(
            False,
            f"**演进记录条数不符**：期望 {wanted}，实际 {len(deltas)}——"
            f"「计划变了」与「变了却没留记录」是两件事",
        )

    if missing := [
        tool for tool in (case.get("expect_extended_tools_contain") or []) if tool not in extended
    ]:
        return Verdict(False, f"**没往这几路下钻**：{missing}（下钻出来的只有 {sorted(extended)}）")

    if (limit := case.get("expect_max_steps")) is not None and len(steps) > limit:
        return Verdict(False, f"**总步数越界**：{len(steps)} > {limit}")

    if case.get("expect_plan_records_consistent"):
        # 22.10.3 的另外两条：「`plan_deltas` 与实际演进次数一致」与
        # 「`revision_no` 连续无跳号」。
        #
        # **它们是"计划变了"与"变了留没留记录"之间的那道校验**：两者不一致时，
        # 复盘看到的是一份**缺了一版的计划变更史**，而答案与步骤都正常——
        # 只有把 revision 与 delta 摆在一起才看得出来。
        # 与 `expect_plan_delta_count`（钉一个具体条数）不同，这条是**不变量**：
        # 不知道这条问题会演进几次时，它照样成立。
        if len(deltas) != revision:
            return Verdict(
                False,
                f"**变更史与版本号对不上**：plan_revision={revision}，"
                f"而 plan_deltas 只有 {len(deltas)} 条——有一版变更没留下记录",
            )
        numbers = [int(delta.get("revision_no") or 0) for delta in deltas]
        if numbers != list(range(1, len(deltas) + 1)):
            return Verdict(
                False,
                f"**revision_no 跳号或乱序**：{numbers}，期望 1..{len(deltas)}"
                f"——跳号意味着某一版被覆盖过，而覆盖之后剩下的那版看起来是对的",
            )

    if case.get("expect_trigger_finding_linked"):
        unlinked = [
            delta.get("revision_no")
            for delta in deltas
            if delta.get("added_step_ids") and not delta.get("trigger_finding_id")
        ]
        if unlinked:
            return Verdict(
                False,
                f"**下钻步骤反查不到催生它的发现**（revision {unlinked}）"
                f"——「因为查空了所以补了一路」这句话在产物上不成立",
            )
        chain = {item.get("finding_id") for item in detail.get("investigation_chain") or []}
        dangling = [
            delta["trigger_finding_id"]
            for delta in deltas
            if delta.get("trigger_finding_id") and delta["trigger_finding_id"] not in chain
        ]
        if dangling:
            return Verdict(False, f"**触发源指向不存在的发现**：{dangling}")

    return None


def _review_or(case: dict[str, Any], detail: dict[str, Any], ok: Verdict) -> Verdict:
    """模式类用例（拒答 / 澄清）通过后，**仍要判审查与计划**。

    深挖一层的原因：这两条路径都会提前 `return`，于是写在后面的
    `expect_review_*` 与 `expect_plan_*` 会**永远不被检查**——写了却从不
    生效的断言比没有断言更糟，因为它看起来是有保障的。这两组与模式正交，
    所以单独走一步。

    **计划那一组同样正交**，而且它的值正是这类用例该钉的：
    拒答与澄清都不该发生演进，`expect_plan_revision: 0` 是最有信息量的
    一条断言——它挡的是"澄清路上莫名其妙多跑了一步"。

    它同时也是一条真断言：**拒答的任务不能被审查拦下**（拒答不是失败）。
    """
    if verdict := _check_review(case, detail.get("review")):
        return verdict
    if verdict := _check_plan(case, detail):
        return verdict
    return ok


def _check_review(case: dict[str, Any], review: Any) -> Verdict | None:
    """Reviewer 的断言。**没写 `expect_review_*` 就完全跳过**，返回 `None`。

    它为什么值得单独断言：Reviewer 是唯一能让一条**看起来正常**的答案
    被拦下的地方（14.3 的一票否决）。而它失效时的症状是"所有任务都通过"——
    与"所有任务都没问题"完全同形，没有断言就永远发现不了。
    """
    expected_status = case.get("expect_review_status")
    expected_reason = case.get("expect_review_reason_code")
    expected_issues = case.get("expect_review_issue_codes")
    if expected_status is None and expected_reason is None and expected_issues is None:
        return None

    if review is None:
        return Verdict(False, "**没有审查结论**——任务没走到 reviewer 那一步")

    if expected_status is not None and review.get("status") != expected_status:
        return Verdict(
            False,
            f"**审查结论不符**：期望 {expected_status}，实际 {review.get('status')}"
            f"（{review.get('reason_code')}）",
        )

    # 六条检查是**顺序执行、每条都可能不产出 issue**，所以"跑过"从结论上
    # 反推：`reason_code` 是命中的第一条的 code，其余检查没有别的痕迹。
    # 能断言的只有"结论形态正确"——这正是本层能力边界（见 `reviewer.py`
    # 的模块 docstring：六条里有三条在本版不产出 issue）。
    if expected_reason is not None and review.get("reason_code") != expected_reason:
        return Verdict(
            False,
            f"**审查归因不符**：期望 {expected_reason}，实际 {review.get('reason_code')}",
        )

    # **集合相等，不是包含**：多记了一条 issue 说明有检查在误报，
    # 而误报正是这一层最该被发现的事——包含关系会让它永远看不出来。
    # 与 `expect_sources` 同一条约定：缺省 = 不检查，`[]` = 确实不该有。
    if expected_issues is not None:
        actual = sorted(str(item.get("code")) for item in review.get("issues") or [])
        if actual != sorted(expected_issues):
            return Verdict(
                False, f"**审查记录的问题不符**：期望 {sorted(expected_issues)}，实际 {actual}"
            )

    # 分数是"审查有多确定"的粗粒度表达（100 − 各 issue 的罚分）。
    # 它**不是**通过判据（status 才是），所以只用来钉住"这一条应当零扣分"。
    floor = case.get("expect_review_score_min")
    if floor is not None and (review.get("score") or 0) < floor:
        return Verdict(False, f"**审查分数低于下限**：{review.get('score')} < {floor}")

    return None


__all__ = [
    "DEMO_USERNAME",
    "NUMBER_TOLERANCE",
    "POLL_INTERVAL_SECONDS",
    "TASK_TIMEOUT_SECONDS",
    "Verdict",
    "amounts",
    "ask",
    "close",
    "evaluate",
    "get",
    "login",
    "missing_amounts",
    "post",
    "wait_for_task",
]
