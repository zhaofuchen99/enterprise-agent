"""RAG 金标评测集的装载（开发流程 6.7 施工项 12 / 7.4 的「RAG 检索 20 条」）。

放在 `app/tools/rag/` 而不是 `scripts/`：**两份消费方分属两侧**——
`scripts/eval_rag.py` 算 Recall@8，`app/tools/rag/verify.py` 的
「无答案提问」那条断言读同一份 `absent_cases`。装载器留在脚本里的话，
`verify.py` 就得反向 import 一个脚本，而 `scripts/` 不是应用包的一部分。

## 两条纪律，与 `eval_sql_golden.yaml` 同源

1. **锚在语料事实，不锚在实现细节**：`expect_docs` 用 `logical_key@version`，
   不用 `chunk_id`（后者随分块逻辑变化，见 YAML 文件头的说明）。
2. **金标与实现分批产出**：本文件写于 Phase 5 语料冻结之后，
   `expect_docs` 是从**产物**（Qdrant 的 payload）里查出来的，不是照清单猜的。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.domain.user import UserRole

#: 默认路径。与 `configs/eval_sql_golden.yaml` 并列。
GOLDEN_PATH = Path("configs/eval_rag_golden.yaml")


class GoldenCase(BaseModel):
    """一条有答案的金标问题。

    `expect_sections` 是**定位一致**的判据（22.3 的门禁："固定问题可稳定返回
    document/version/section/chunk 定位"）。做成"含任一个"而不是"全含"：
    一个问题常常跨两节都算对（定义在「指标定义」、公式在「计算公式」），
    要求全含会把正确答案判错，而那会让人去调检索参数——方向全错。
    """

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    expect_docs: tuple[str, ...] = Field(min_length=1)
    expect_sections: tuple[str, ...] = ()
    #: 问题所问的时点。给定时命中必须是那一版（21.8 验收 2 的"有效期过滤正确"）。
    #: **不给就是"不限时点"**，那时同名制度的两个版本都可能进候选。
    as_of: date | None = None
    #: 以什么身份检索。默认 `ANALYST`（与 `app_user` 的演示账号一致）。
    #:
    #: **这个字段是踩出来的**：`policy/regional-discount-quota`（区域折扣授权额度表）
    #: 是语料里 2 份 CONFIDENTIAL 之一，`allowed_roles` 只有 ADMIN。
    #: 一条问它的金标以 ANALYST 跑，检索**正确地**把它挡在外面，
    #: 而评测报出来的是"未命中"——看起来像召回缺陷，实际是权限过滤在工作。
    #: 一条用例的失败原因指错方向时，调参的人会去调 embedding。
    role: UserRole = UserRole.ANALYST
    #: 这条用例要证明什么。**不是装饰**：Recall@8 掉下来时，
    #: 第一个要回答的问题是"掉的是哪一类问题"，而问题本身看不出来。
    proves: str = ""


class AbsentCase(BaseModel):
    """一条**应拒答**的问题（16.11.2 第 10 项「涉及但不存在」）。

    `absent_topic` 与 `corpus_manifest.yaml` 的 `absent_policies[].name`
    一一对应——那份清单登记"不该生成的文档"，这里登记"问它的问法"。
    """

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    absent_topic: str
    #: 用来证明"语料里确实没有这件事"的**探针词**：在全部 chunk 正文里
    #: 出现次数为 0 的那几个词（`make verify-corpus` 会复核这一点）。
    #:
    #: **不按片段切、也不用整串匹配**：整串匹配会因为一个虚词就放过；
    #: 按 3 字切则会把「业务管理」「管理办法」这类通用组合算成命中——
    #: 实测第一版就是这么报出"语料里出现了"的假警报。
    #: 探针词必须是**这份制度特有的**，所以逐条手写并注明出处。
    absent_probes: tuple[str, ...] = ()
    proves: str = ""


class GoldenSet(BaseModel):
    """整份金标集。

    `cases` 与 `absent_cases` **分成两组而不是打一个 `absent` 标记**：
    两组的分母与判定标准完全不同（Recall@8 vs 拒答率），
    合成一组就必然要在统计时过滤，而漏过滤的表现是
    「Recall@8 看起来只有 87%」——一个不会指向任何代码的数字。
    """

    model_config = ConfigDict(frozen=True)

    version: str
    cases: tuple[GoldenCase, ...]
    absent_cases: tuple[AbsentCase, ...] = ()


def load_golden(path: str | Path = GOLDEN_PATH) -> GoldenSet:
    """读金标集。**格式错误直接抛**，不吞。

    与语料清单（`gen_corpus.load_manifest`）同样的取舍：金标是**门禁的输入**，
    读出一份缺字段的它，会让门禁以一个看起来正常的比率通过。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return GoldenSet.model_validate(raw)


__all__ = ["GOLDEN_PATH", "AbsentCase", "GoldenCase", "GoldenSet", "load_golden"]
