"""模型输出的边界：**能理解的输入就别判非法**。

这一组的存在理由是一次真实故障（2026-09-20）：模型在"这个问题没有时间"时
把 `time_range` 写成 `{"start": "", "end": ""}`，而我们判它非法 →
`MODEL_OUTPUT_INVALID` → 整条任务失败。

症状极具误导性：**只有不含明确年份的问题会挂**（带「2025年Q3」的问题模型会
填真日期、一切正常），于是看起来像"检索坏了"或"某些问题查不出来"——
排查方向会跑到 RAG 上去，而故障在 Schema 与 JSON 契约示例里。
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from app.agent.schemas.analysis import AnalysisResult
from app.agent.schemas.plan import IntentResult
from app.infrastructure.model_gateway import build_json_contract
from app.tools.rag.schemas import QueryRewrite
from app.tools.sql.schemas import SqlCandidate


def _valid_intent(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "intent": "QUERY",
        "required_sources": ["sql"],
        "confidence": 0.9,
    }
    return {**base, **overrides}


def test_an_empty_time_range_means_no_time_filter() -> None:
    """`{"start": "", "end": ""}` 是"没有时间"，不是"非法时间"。

    空串不携带信息，而本字段的 `None` 本来就表示"不限时点"——两者语义相同。
    判非法会让一条**完全正常的问题**整条任务失败。
    """
    result = IntentResult.model_validate(_valid_intent(time_range={"start": "", "end": ""}))

    assert result.time_range is None


def test_a_partially_empty_time_range_is_still_rejected() -> None:
    """只给一半（`start` 空、`end` 有值）**仍然判非法**。

    它与"没有时间"不同：半个区间是**有信息但说不清**，当成"不限时点"会把
    用户明确说的那个边界悄悄丢掉——那比报错更糟。模型输出这种形状时
    应当重试（网关的 VALIDATION 预算正是干这个的）。
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IntentResult.model_validate(
            _valid_intent(time_range={"start": "", "end": "2025-12-31T00:00:00Z"})
        )


def test_a_real_time_range_is_accepted() -> None:
    result = IntentResult.model_validate(
        _valid_intent(time_range={"start": "2025-07-01T00:00:00Z", "end": "2025-10-01T00:00:00Z"})
    )

    assert result.time_range is not None
    assert result.time_range.start.month == 7


#: 真的会被送进 `invoke_structured` 的那几个 Schema。**示例必须对每一个都自洽**，
#: 否则模型照着示例填出来的东西必然过不了校验（这就是那次故障的机制）。
_REAL_SCHEMAS = (IntentResult, SqlCandidate, AnalysisResult, QueryRewrite)


@pytest.mark.parametrize("schema", _REAL_SCHEMAS, ids=lambda s: s.__name__)
def test_the_json_contract_example_validates_against_its_own_schema(
    schema: type[BaseModel],
) -> None:
    """**JSON 契约里的示例必须能通过它自己的校验**。

    示例是给模型看的引导，模型会照着它的**形状与取值风格**填。给一个
    "这个字段不可能是这个值"的示例（日期字段给空串、必填字段给空串），
    等于把模型往非法输出上引——而失败发生在校验层，
    报的是"模型不听话"，实际是我们给错了示范。

    这条断言是通用的：任何新增/修改 Schema 只要引入新的取值约束
    （`format`、`minLength`…），示例生成也必须跟着能产出合法值。
    """
    contract = build_json_contract(schema)
    start, end = contract.index("{"), contract.rindex("}")
    example = json.loads(contract[start : end + 1])

    schema.model_validate(example)


def test_an_optional_field_is_shown_as_null_in_the_contract() -> None:
    """**可空字段的示例给 `null`，不给具体值**——按"照抄示例的后果"定的规则。

    照抄 `null` 得到的是**正确**语义（这个字段可以没有）；照抄一个具体值
    得到的是**看起来合法、实际错误**的值。这一条是拿两种写法各跑一遍
    真实模型 A/B 出来的（见 `_example_from_schema` 的注释）：
    `time_range` 的示例给日期时，模型会把它抄进"这个问题没有时间"的意图里，
    于是凭空多出一个时间区间去过滤文档生效期，而且不报任何错。
    """
    contract = build_json_contract(IntentResult)
    start, end = contract.index("{"), contract.rindex("}")
    example = json.loads(contract[start : end + 1])

    assert example["time_range"] is None
    # 非空字段仍然给具体示例：它们**必须**有值，示例也必须有内容
    assert example["intent"] == "QUERY"
