"""演示语料生成器（详细设计 16.11.1 / 16.11.2）。

按 `configs/corpus_manifest.yaml` 生成 88 篇企业文档，注入 10 类缺陷，
输出 PDF / DOCX / Markdown / TXT 四种格式。

## 为什么语料要自己生成

见清单文件头部的三条理由（版权、与业务库对不上、真实文档不自带缺陷）。
这里只补一条**工程上**的理由：**语料必须能被重建**。

正文不进版本库（`data/` 已 gitignore），进库的是清单与这个脚本。
于是「语料是什么」与「语料怎么来的」都可复审，产物随时可重建。
若把 88 份二进制文档直接提交，改一个缺陷就要重新提交一批二进制，
diff 里什么都看不出来——这正是 `verify-corpus` 想防的那类不可复审状态。

## 数字来自业务库，不手抄

报告的销售额、达成率、同比全部**现查业务库**。手抄进模板会与库漂移，
而「报告与数据库的数字差 1–2%」是这个项目最核心的演示点：
两边都错一点，差异就不是 1.78% 而是别的数，冲突检测的断言随之失去意义。

顺带也证明了只读账号（`DATABASE_URL_BUSINESS_RO`）读得到全部演示数据——
与 `business_seed.py` 的断言走同一条路径。

## 体裁学真的，内容自己填

制度类的「第一章 总则 → 分章规则 → 附则 + 修订历史表」、经营报告的
「摘要 → 分项业绩 → 问题归因 → 下月计划 → 风险提示」、达成率的红绿灯阈值
（≥100% 绿 / 80–100% 黄 / <80% 红），这些都是公开范本里的真实结构。
**学形式不学内容**：既绕开版权，又让文档读起来像真制度而不是模型腔。

## 缺陷注入的落点

`16.11.2` 的 10 类缺陷里，8 类由本脚本按 `defects` 标记注入；
Prompt Injection 的正文来自 `configs/corpus_handwritten/`（**手工编写**，
见该目录的说明）；「涉及但不存在的制度」靠的是**不生成**，
登记在清单的 `absent_policies` 里。

用法：
    uv run python scripts/gen_corpus.py                # 全量
    uv run python scripts/gen_corpus.py --subset 10    # 只出 10 篇样本（先跑通链路用）
    uv run python scripts/gen_corpus.py --only SP-001  # 单篇调试
    uv run python scripts/gen_corpus.py --list         # 只打印清单摘要，不生成
"""

from __future__ import annotations

import argparse
import asyncio
import calendar
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings

# ----------------------------------------------------------------- 常量

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "configs" / "corpus_manifest.yaml"
HANDWRITTEN_DIR = ROOT / "configs" / "corpus_handwritten"
DEFAULT_OUT = ROOT / "data" / "corpus"

#: 公司抬头。演示系统按 TBC-01 是自建演示系统，没有真实企业主体，
#: 但页眉页脚需要一个名字——用「示例」而不是编造一个像真的企业名，
#: 避免语料被误当成某家真实公司的内部文件。
COMPANY = "示例科技股份有限公司"

#: 密级。制度类与经营报告是 INTERNAL，
#: 涉及折扣审批权限与客户口径的标 CONFIDENTIAL（与 11.4 的 classification 对齐）。
CONFIDENTIAL_KEYS = ("policy/regional-discount-quota", "policy/discount-compliance-redline")

#: 报告里「销售额」的口径：含税 − 折扣，**未扣退货冲减**。
#: 与 net_amount（净销售额）的差额恰为退货金额，2025 年实测 1.77%–1.80%，
#: 落在 16.11.2 要求的 1–2% 内。依据与实测见 schema_catalog.yaml 的 net_sales.note。
REPORT_BASIS_NOTE = "本月销售额按含税口径列示，未扣除退货冲减"

#: 达成率红绿灯（公开范本里的通行阈值，见模块 docstring）
ACHIEVE_GREEN, ACHIEVE_YELLOW = Decimal("1.00"), Decimal("0.80")

#: 16.11.2 第 4 项：区域范围口径冲突。文档里写的省份数，
#: 与 `dim_region.province_count` 的真实值故意不一致。
SCOPE_CLAIM_PROVINCES = {"华东": 3}


# ----------------------------------------------------------------- 内容块模型


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Para:
    text: str


@dataclass(frozen=True)
class Table:
    caption: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    #: 跨页时是否重复表头。16.11.2 第 7 项要测的正是这件事，
    #: 所以它是表级开关而不是全局设置——全 False 就测不出还原能力。
    repeat_header: bool = True
    #: 强制分页的语义标记：置 True 时渲染器会在表格前插分页符，
    #: 使表格必然跨越两页（否则一张 8 行的表可能正好塞进当前页，
    #  跨页这条缺陷就注入了个寂寞）。
    force_split: bool = False


@dataclass(frozen=True)
class PageBreak:
    pass


Block = Heading | Para | Table | PageBreak


@dataclass
class DocSpec:
    """清单条目 + 生成时补充的信息。"""

    id: str
    logical_key: str
    title: str
    type: str
    format: str
    version: str
    department: str | None
    source_kind: str
    effective_from: str | None
    effective_to: str | None
    published_at: str | None
    defects: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def confidential(self) -> bool:
        return self.logical_key in CONFIDENTIAL_KEYS

    @property
    def classification(self) -> str:
        return "CONFIDENTIAL" if self.confidential else "INTERNAL"


# ----------------------------------------------------------------- 业务库事实


@dataclass
class Facts:
    """一次取全的业务库真值，供所有渲染函数复用。

    故意**不做按需查询**：88 篇文档若每篇查一次库就是 88 次往返，
    且不同文档可能看到不一致的快照。一次取全既快又保证全体语料同源。
    """

    regions: dict[str, dict[str, Any]]
    channels: dict[str, dict[str, Any]]
    product_lines: dict[str, dict[str, Any]]
    products: dict[str, dict[str, Any]]
    #: (ym, region, channel, line) -> 金额分项，单位「元」
    daily_facts: dict[tuple[str, str, str, str], dict[str, Decimal]]
    #: (ym, region, line) -> 目标净销售额，单位「元」
    targets: dict[tuple[str, str, str], Decimal]


async def load_facts() -> Facts:
    """从业务库读全量真值。用只读账号——与 SQL Tool 的取数路径一致。"""
    settings = get_settings()
    engine = create_async_engine(settings.database_url_business_ro)

    regions: dict[str, dict[str, Any]] = {}
    channels: dict[str, dict[str, Any]] = {}
    lines: dict[str, dict[str, Any]] = {}
    products: dict[str, dict[str, Any]] = {}
    daily: dict[tuple[str, str, str, str], dict[str, Decimal]] = {}
    targets: dict[tuple[str, str, str], Decimal] = {}

    async with engine.connect() as conn:
        for row in (
            await conn.execute(
                text("SELECT region_code, region_name, province_count FROM dim_region")
            )
        ).mappings():
            regions[str(row["region_name"])] = {
                "code": row["region_code"],
                "province_count": int(row["province_count"]),
            }

        for row in (
            await conn.execute(
                text("SELECT channel_code, channel_name, channel_type FROM dim_channel")
            )
        ).mappings():
            channels[str(row["channel_name"])] = {
                "code": row["channel_code"],
                "type": row["channel_type"],
            }

        for row in (
            await conn.execute(
                text("SELECT product_line_code, product_line_name, category FROM dim_product_line")
            )
        ).mappings():
            lines[str(row["product_line_name"])] = {
                "code": row["product_line_code"],
                "category": row["category"],
            }

        for row in (
            await conn.execute(
                text(
                    "SELECT p.product_code, p.product_name, p.list_price, p.launch_date, "
                    "       p.status, ln.product_line_name, ln.category "
                    "FROM dim_product p "
                    "JOIN dim_product_line ln ON ln.product_line_id = p.product_line_id"
                )
            )
        ).mappings():
            products[str(row["product_code"])] = {
                "name": row["product_name"],
                "price": Decimal(str(row["list_price"])),
                "launch_date": row["launch_date"],
                "status": row["status"],
                "line": row["product_line_name"],
                "category": row["category"],
            }

        # 按天取，聚合到月/季/年在 Python 里做。
        # 之所以不直接在 SQL 里按季度聚合：报告要按「统计截止日」截断
        # （如 9/28 而非 9/30），那要求数据保留到**天**这一粒度，
        # 否则 TIME 冲突这类缺陷根本造不出来。
        sql = text(
            "SELECT DATE_FORMAT(f.order_date, '%Y-%m') AS ym, f.order_date AS d, "
            "       r.region_name AS region, c.channel_name AS channel, "
            "       ln.product_line_name AS line_name, "
            "       SUM(f.net_amount) AS net, SUM(f.gross_amount) AS gross, "
            "       SUM(f.discount_amount) AS disc, SUM(f.return_amount) AS ret, "
            "       COUNT(f.order_id) AS cnt "
            "FROM fact_sales_order_item f "
            "JOIN dim_region r ON r.region_id = f.region_id "
            "JOIN dim_channel c ON c.channel_id = f.channel_id "
            "JOIN dim_product p ON p.product_id = f.product_id "
            "JOIN dim_product_line ln ON ln.product_line_id = p.product_line_id "
            "GROUP BY ym, d, region, channel, line_name"
        )
        for row in (await conn.execute(sql)).mappings():
            key = (
                f"{row['d']}",
                str(row["region"]),
                str(row["channel"]),
                str(row["line_name"]),
            )
            daily[key] = {
                "net": Decimal(str(row["net"] or 0)),
                "gross": Decimal(str(row["gross"] or 0)),
                "disc": Decimal(str(row["disc"] or 0)),
                "ret": Decimal(str(row["ret"] or 0)),
                "cnt": Decimal(str(row["cnt"] or 0)),
            }

        for row in (
            await conn.execute(
                text(
                    "SELECT DATE_FORMAT(t.period_month, '%Y-%m') AS ym, "
                    "       r.region_name AS region, ln.product_line_name AS line_name, "
                    "       SUM(t.target_amount) AS tgt "
                    "FROM sales_target t "
                    "JOIN dim_region r ON r.region_id = t.region_id "
                    "JOIN dim_product_line ln ON ln.product_line_id = t.product_line_id "
                    "GROUP BY ym, region, line_name"
                )
            )
        ).mappings():
            targets[(str(row["ym"]), str(row["region"]), str(row["line_name"]))] = Decimal(
                str(row["tgt"] or 0)
            )

    await engine.dispose()
    return Facts(regions, channels, lines, products, daily, targets)


# ----------------------------------------------------------------- 聚合工具


#: 报告期字符串 -> 覆盖的月份列表。月的表达方式跨文档不统一
#: （月度写 `2025-07`、季度写 `2025-Q3`、半年写 `2025-H1`），
#: 在清单里保持业务写法，在这里统一解析。
def period_months(period: str) -> list[str]:
    if "-Q" in period:
        year, q = period.split("-Q")
        start = (int(q) - 1) * 3 + 1
        return [f"{year}-{m:02d}" for m in range(start, start + 3)]
    if "-H" in period:
        year, h = period.split("-H")
        start = 1 if h == "1" else 7
        return [f"{year}-{m:02d}" for m in range(start, start + 6)]
    if len(period) == 4:
        return [f"{period}-{m:02d}" for m in range(1, 13)]
    return [period]


def prior_year_period(period: str) -> str:
    """同比对照期。年度没有上一年数据（2024 只有事实表、没有目标），返回空。"""
    if len(period) == 4:
        return ""
    head, _, tail = period.partition("-")
    return f"{int(head) - 1}-{tail}"


def aggregate(
    facts: Facts,
    months: list[str],
    *,
    region: str | None = None,
    channel: str | None = None,
    line: str | None = None,
    cutoff: str | None = None,
) -> dict[str, Decimal]:
    """把日粒度事实聚合成报告要用的金额分项。

    `cutoff` 是**统计截止日**（形如 `2025-09-28`）：只统计到这一天为止。
    这正是 16.11.2 第 3 项 TIME 冲突的机制——报告写出来的数字偏低，
    原因不是口径而是少统计了几天，两类冲突必须能被分开判定。
    """
    out = {k: Decimal(0) for k in ("net", "gross", "disc", "ret", "cnt")}
    for (day, r, c, ln), vals in facts.daily_facts.items():
        if day[:7] not in months:
            continue
        if cutoff and day > cutoff:
            continue
        if region and r != region:
            continue
        if channel and c != channel:
            continue
        if line and ln != line:
            continue
        for k in out:
            out[k] += vals[k]
    return out


def target_sum(
    facts: Facts, months: list[str], *, region: str | None = None, line: str | None = None
) -> Decimal:
    total = Decimal(0)
    for (ym, r, ln), amt in facts.targets.items():
        if ym not in months:
            continue
        if region and r != region:
            continue
        if line and ln != line:
            continue
        total += amt
    return total


# ----------------------------------------------------------------- 格式化


def wan(amount: Decimal) -> str:
    """元 -> 万元，千分位，两位小数。"""
    return f"{amount / 10000:,.2f}"


def pct(numerator: Decimal, denominator: Decimal) -> str:
    if denominator == 0:
        return "—"
    return f"{numerator / denominator * 100:.1f}%"


def delta(cur: Decimal, prev: Decimal) -> str:
    if prev == 0:
        return "—"
    d = (cur - prev) / prev * 100
    return f"{'+' if d >= 0 else ''}{d:.1f}%"


def weighted(cur: Decimal, prev: Decimal) -> str:
    """同比用「含税 − 折扣」口径比——报告前后两期必须同口径，
    否则同比里混进了口径变化，读出来的趋势是错的。"""
    return delta(cur - Decimal(0), prev)


def achieve(actual: Decimal, target: Decimal) -> tuple[str, str]:
    """达成率与红绿灯。返回值第二项是灯色，写进正文用文字表达。"""
    if target == 0:
        return "—", "无目标"
    ratio = actual / target
    light = "绿灯" if ratio >= ACHIEVE_GREEN else ("黄灯" if ratio >= ACHIEVE_YELLOW else "红灯")
    return f"{ratio * 100:.1f}%", light


def report_amount(agg: dict[str, Decimal]) -> Decimal:
    """报告的「销售额」= 含税 − 折扣（未扣退货）。见 REPORT_BASIS_NOTE。"""
    return agg["gross"] - agg["disc"]


# ----------------------------------------------------------------- 体裁渲染：制度类


def _meta_block(spec: DocSpec, extra: list[str] | None = None) -> list[Block]:
    """文件头信息块。真实制度文档的第一页都有这一块。"""
    rows = [
        ("文件编号", spec.id),
        ("版　　本", spec.version),
        ("责任部门", spec.department or "—"),
        ("密　　级", spec.classification),
    ]
    if spec.effective_from:
        rows.append(("生效日期", spec.effective_from))
    if spec.effective_to:
        rows.append(("失效日期", spec.effective_to))
    for e in extra or []:
        k, _, v = e.partition("：")
        rows.append((k, v))
    return [Table("文件信息", ("项目", "内容"), tuple(rows), repeat_header=False)]


def _revision_history(spec: DocSpec, *, pages: int = 0) -> list[Block]:
    """修订历史表。16.11.2 第 8 项要求「页眉页脚 / 修订历史」清洗后不进 chunk——
    这类结构化冗余若被切进正文，会污染检索（「版本」这种词会命中所有文档）。"""
    hist = [
        (
            spec.version,
            spec.effective_from or "—",
            "首次发布" if spec.version.endswith("1.0") else "修订发布",
        )
    ]
    if spec.version.endswith("1.1"):
        hist.insert(0, ("v1.0", spec.effective_from or "—", "首次发布"))
    if spec.version.endswith("1.2"):
        hist = [
            ("v1.2", spec.effective_from or "—", "修订发布"),
            ("v1.0", spec.effective_from or "—", "首次发布"),
        ]
    if spec.version.endswith("2.0"):
        hist.insert(0, ("v1.0", "—", "首次发布（已废止）"))
    return [
        Heading(1, "附则"),
        Heading(2, "第二十条　解释与修订"),
        Para(
            f"本文件由{spec.department or '相关部门'}负责解释，修订须经制度评审会审议通过后发布。"
            "本文件与本公司其他制度不一致时，以本文件为准；与法律法规不一致时，以法律法规为准。"
        ),
        Heading(2, "第二十一条　施行"),
        Para(
            f"本文件自{spec.effective_from or '发布之日'}起施行"
            + ("，原同期文件同时废止。" if not spec.effective_to else "。")
        ),
        Heading(1, "修订历史"),
        Table(
            "修订记录",
            ("版本", "日期", "修订说明"),
            tuple(hist),
        ),
    ]


def render_channel_discount_policy(spec: DocSpec, facts: Facts) -> list[Block]:
    """区域渠道折扣政策（SP-001 ~ SP-006）。"""
    region = spec.raw.get("region") or _region_from_key(spec.logical_key, facts)
    info = facts.regions.get(region, {"province_count": 3, "code": "R??"})
    provinces = SCOPE_CLAIM_PROVINCES.get(region, info["province_count"])

    ch_rows = []
    for i, name in enumerate(facts.channels):
        base = Decimal(3 + i) + Decimal("0.5") * i
        ch_rows.append(
            (
                name,
                f"{base:.1f}%",
                f"{base + Decimal('2'):.1f}%",
                f"{base + Decimal('5'):.1f}%",
                "区域总监" if i < 2 else "渠道经理",
            )
        )

    return [
        Heading(1, f"{region}区域渠道折扣政策"),
        *_meta_block(spec),
        Heading(1, "第一章　总则"),
        Heading(2, "第一条　目的"),
        Para(
            f"为规范{region}区域各渠道的价格与折扣管理，统一审批口径，防止跨渠道窜价与无序让利，"
            "保障公司整体毛利水平，特制定本政策。"
        ),
        Heading(2, "第二条　适用范围"),
        Para(
            f"本政策适用于{region}区域（下辖 {provinces} 个省份，区域编码 {info['code']}）"
            "范围内所有直营、经销、电商及 KA 渠道的产品销售与折扣审批活动。"
        ),
        Heading(2, "第三条　术语定义"),
        Para(
            "（一）标准折扣率：指按渠道类型与产品线预先核定的基准折扣，未经审批不得突破。\n"
            "（二）特批折扣：指因竞争、清库存或战略客户等原因，突破标准折扣率并履行审批程序的折扣。\n"
            "（三）窜货：指经销商将约定销售区域内的产品销往其他区域，或以低于核定价对外销售的行为。"
        ),
        Heading(1, "第二章　折扣标准"),
        Heading(2, "第四条　分渠道标准折扣率"),
        Para("各渠道标准折扣率按下表执行。表中「下限」为不得超过的最大让利幅度。"),
        Table(
            f"{region}区域分渠道标准折扣率",
            ("渠道", "标准折扣率", "审批下限", "特批上限", "审批人"),
            tuple(ch_rows),
        ),
        Heading(2, "第五条　产品线调整系数"),
        Para(
            "新品上市首年、临期清仓品及战略主推品可在标准折扣率基础上申请调整，"
            "调整系数须在折扣审批单中列明依据，不得以口头方式确认。"
        ),
        Heading(1, "第三章　审批与执行"),
        Heading(2, "第六条　审批权限"),
        Para(
            "折扣幅度在标准范围内的，由区域销售负责人审批；突破标准折扣率但未超过审批下限的，"
            "由渠道经理审批；超过审批下限的，须报销售运营部与财务部会签。"
        ),
        Heading(2, "第七条　执行与监督"),
        Para(
            "折扣一经审批，须在订单系统中如实录入，不得事后补录。"
            "财务部按月抽查折扣执行情况，发现未按审批执行的，暂停该渠道当月折扣权限。"
        ),
        *_revision_history(spec),
    ]


def _region_from_key(logical_key: str, facts: Facts) -> str:
    mapping = {
        "east-china": "华东",
        "south-china": "华南",
        "north-china": "华北",
        "central-china": "华中",
        "southwest-china": "西南",
    }
    for k, v in mapping.items():
        if k in logical_key:
            return v
    return next(iter(facts.regions), "华东")


def render_region_quota(spec: DocSpec, facts: Facts) -> list[Block]:
    """SP-015　区域折扣授权额度表。

    两处缺陷叠加：
      - **SCOPE_CONFLICT**：正文写「华东区域含三省」，而库里 province_count = 4。
        差异可被确定性比对，比对的另一侧是 MD-008 与 `dim_region`。
      - **CROSS_PAGE_TABLE**：额度表行数足够多且强制分页，必然跨页，
        表头在第二页必须重复，否则解析出来的第二页表格没有列名。
    """
    region = "华东"
    claimed = SCOPE_CLAIM_PROVINCES[region]
    #: 明细表按 区域 × 渠道 × 产品线 铺开 = 5 × 4 × 4 = 80 行。
    #:
    #: **为什么是 80 行而不是 16 行**：`force_split` 只是把表挪到新页开头，
    #: 并不强制它跨页——表要真的跨页，行数必须撑破一页。实测（A4 + 20mm 边距 +
    #: 9.5pt 正文）门槛是 **35 行**，16 行的表稳稳地待在一页里，
    #: 于是「标记了 CROSS_PAGE_TABLE」而「实际没跨页」——第一版就是这么错的，
    #: 而且错得很隐蔽：`verify-corpus` 若只断言「这篇标记了该缺陷」就会放行。
    #: 现在由 `check_injection()` 在生成后**回读 PDF 验证表头真的重复了**。
    rows: list[tuple[str, ...]] = []
    for rname in facts.regions:
        for cname in facts.channels:
            for lname in facts.product_lines:
                idx = len(rows)
                rows.append(
                    (
                        rname,
                        cname,
                        lname,
                        f"{Decimal(30 + idx % 17 * 5) * 10000:,.0f}",
                        f"{Decimal(2 + idx % 6):.1f}%",
                    )
                )
    return [
        Heading(1, "区域折扣授权额度表"),
        *_meta_block(spec, [f"适用区域：{region}（含 {claimed} 个省份）"]),
        Heading(1, "一、编制说明"),
        Para(
            f"本表按区域 × 渠道 × 产品线维度核定各销售单元的年度折扣授权额度。"
            f"{region}区域下辖 {claimed} 个省份，各省级销售单元在总额度内自主分配，"
            "不得跨区域调剂。额度按自然年度核算，年末剩余额度不结转。"
        ),
        Para(
            "超出额度部分须走特批流程，单笔超过 50 万元的须报总经理办公会审议。"
            "额度执行情况由财务部按季度通报。"
        ),
        PageBreak(),
        Heading(1, "二、授权额度明细"),
        Para("下表为各区域、各渠道、各产品线的折扣授权额度（单位：元）。"),
        Table(
            "各区域折扣授权额度明细",
            ("区域", "渠道", "产品线", "年度额度", "标准折扣率"),
            tuple(rows),
            force_split=True,
        ),
        Heading(1, "三、附则"),
        Para(
            "本表自发布之日起执行，由销售运营部与财务部共同维护。"
            "额度调整须以书面形式通知各销售单元，口头通知无效。"
        ),
        *_revision_history(spec),
    ]


def render_dealer_assessment(spec: DocSpec, facts: Facts) -> list[Block]:
    """CM-002　经销商年度考核办法。

    CROSS_PAGE_TABLE 第 2 份：考核评分表 14 行 × 5 列，加上表前的分章内容，
    必然跨页。**这张表不能走通用骨架**——通用骨架的「分渠道要求」表只有 4 行，
    塞进当前页绰绰有余，于是「标记了跨页表格」而「实际没跨页」，
    注入形同虚设。第一版就是这么错的，靠逐份抽取文本才发现。
    """
    #: 考核维度与权重。分值与权重相乘得到该项满分，合计 100 分。
    #:
    #: 条目按「维度 / 子项」两层展开，共 40 行——**行数是刻意的**。
    #: 实测 A4 + 20mm 边距下表格要 ≥35 行才会真的跨页（见 `render_region_quota`
    #: 的说明）。一份 40 项的考核表在真实企业里很常见（销售类考核本就琐碎），
    #: 不是为凑行数硬加的：每个子项都有独立的评分规则。
    items = [
        ("销售达成", "年度净销售额达成率", "20%", "100", "≥100% 得满分，每低 1pp 扣 2 分"),
        ("销售达成", "季度达成均衡度", "4%", "100", "按四个季度达成率的标准差折算"),
        ("销售达成", "重点产品线达成率", "3%", "100", "按智能家居产品线达成率折算"),
        ("销售达成", "新客户首单贡献", "3%", "100", "按新客户销售额占比折算"),
        ("销售成长", "净销售额同比增速", "8%", "100", "≥5% 得满分，负增长得 0 分"),
        ("销售成长", "客户数同比净增", "4%", "100", "按净增客户数折算"),
        ("销售成长", "单客户产出提升", "3%", "100", "按客均销售额同比折算"),
        ("回款质量", "货款回收率", "8%", "100", "≥95% 得满分，低于 80% 得 0 分"),
        ("回款质量", "逾期账款占比", "4%", "100", "≤2% 得满分，每超 1pp 扣 20 分"),
        ("回款质量", "账期执行合规", "3%", "100", "每发现一次超账期扣 15 分"),
        ("渠道健康", "库存周转天数", "6%", "100", "≤45 天得满分，每超 5 天扣 10 分"),
        ("渠道健康", "低于安全线周数", "4%", "100", "无低于安全线周次得满分，每周扣 15 分"),
        ("渠道健康", "滞销品清理完成率", "3%", "100", "按季度清理计划完成率折算"),
        ("渠道健康", "库存账实相符率", "2%", "100", "≥99% 得满分，低于 95% 得 0 分"),
        ("价格合规", "折扣执行合规率", "6%", "100", "每发现一次未审批先执行扣 20 分"),
        ("价格合规", "窜货投诉次数", "4%", "100", "零投诉得满分，每次投诉扣 25 分"),
        ("价格合规", "价保政策执行", "3%", "100", "每发现一次违价销售扣 20 分"),
        ("价格合规", "促销备案及时率", "2%", "100", "每次未备案扣 10 分"),
        ("市场建设", "终端网点净增数", "3%", "100", "按年度计划完成率折算"),
        ("市场建设", "品牌活动执行", "2%", "100", "按活动执行评分折算"),
        ("市场建设", "导购培训覆盖", "2%", "100", "按培训人次完成率折算"),
        ("市场建设", "陈列达标率", "2%", "100", "按抽查达标比例折算"),
        ("客户服务", "客户满意度", "2%", "100", "≥90 分得满分"),
        ("客户服务", "投诉处理时效", "1%", "100", "48 小时内闭环得满分"),
        ("客户服务", "售后服务配合度", "1%", "100", "按厂家评价折算"),
        ("客户服务", "客诉复发率", "1%", "100", "≤5% 得满分"),
        ("运营协同", "月度目标分解完成", "2%", "100", "按月提交且经确认得满分"),
        ("运营协同", "促销执行落地率", "2%", "100", "按落地门店比例折算"),
        ("运营协同", "竞品信息反馈", "1%", "100", "按月反馈得满分"),
        ("运营协同", "区域协同配合", "1%", "100", "由区域经理评价"),
        ("数据报送", "报送及时率", "—", "扣分项", "迟报一次扣 5 分，扣完为止"),
        ("数据报送", "数据准确率", "—", "扣分项", "差错一次扣 5 分，扣完为止"),
        ("数据报送", "系统录入规范", "—", "扣分项", "录入不规范每次扣 3 分"),
        ("数据报送", "台账完整性", "—", "扣分项", "台账缺失每次扣 5 分"),
        ("合规红线", "重大违规", "—", "一票否决", "发生即取消当年评优资格"),
        ("合规红线", "虚假报送", "—", "一票否决", "查实即终止年度合作"),
        ("合规红线", "跨区销售", "—", "一票否决", "查实即取消当年折扣授权"),
        ("合规红线", "恶意低价", "—", "一票否决", "查实即停止供货"),
        ("其他", "公司认定应考核事项", "—", "±10", "由渠道管理部提出，报管理层认定"),
        ("其他", "重大贡献加分", "—", "+10", "由区域经理提名，管理层认定"),
    ]
    rows = tuple(items)
    return [
        Heading(1, "经销商年度考核办法"),
        *_meta_block(spec),
        Heading(1, "第一章　总则"),
        Heading(2, "第一条　目的"),
        Para(
            "为客观评价经销商的年度经营表现，引导渠道资源向优质经销商集中，"
            "建立能进能出的渠道管理机制，特制定本办法。"
        ),
        Heading(2, "第二条　适用范围"),
        Para("本办法适用于与本公司签订年度经销协议的全部经销商，考核周期为自然年度。"),
        Heading(2, "第三条　考核原则"),
        Para(
            "（一）数据说话：全部考核项以系统数据为准，不接受经销商自行提供的台账；\n"
            "（二）过程与结果并重：既看销售达成，也看价格合规与渠道健康；\n"
            "（三）结果公开：考核结果向被考核方告知，并允许在 5 个工作日内提出申辩。"
        ),
        Heading(1, "第二章　考核指标与权重"),
        Heading(2, "第四条　考核评分表"),
        Para(
            "年度考核按下列评分表执行，满分 100 分，另有扣分项与一票否决项。"
            "各项得分按权重加权后合计，保留一位小数。"
        ),
        Table(
            "经销商年度考核评分表",
            ("考核维度", "考核项", "权重", "分值", "评分规则"),
            rows,
            force_split=True,
        ),
        Heading(2, "第五条　等级评定"),
        Para(
            "考核得分 ≥ 90 分为 A 级，80 至 90 分为 B 级，70 至 80 分为 C 级，"
            "低于 70 分为 D 级。发生合规红线事项的，当年直接评为 D 级，不参与评优。"
        ),
        Heading(1, "第三章　结果应用"),
        Heading(2, "第六条　资源分配"),
        Para(
            "A 级经销商优先获得新品首发、市场费用支持与折扣授权额度上浮；"
            "C 级经销商维持现有政策不变；D 级经销商须在 3 个月内提交整改方案，"
            "整改不到位的按渠道退出管理办法处理。"
        ),
        Heading(2, "第七条　申辩与复核"),
        Para(
            "经销商对考核结果有异议的，可在结果告知后 5 个工作日内提交书面申辩材料，"
            "由渠道管理部组织复核，复核结果为最终结果。"
        ),
        *_revision_history(spec),
    ]


def render_return_writeoff(spec: DocSpec, facts: Facts) -> list[Block]:
    """SP-014　退货与冲减管理办法。

    这是 DEFINITION 冲突的**制度依据**：经营报告里的「销售额」未扣退货，
    与本制度定义的净销售额口径不同。冲突检测给出解释时要引这一篇。
    """
    return [
        Heading(1, "退货与冲减管理办法"),
        *_meta_block(spec),
        Heading(1, "第一章　总则"),
        Heading(2, "第一条　目的"),
        Para(
            "为规范退货、换货及销售冲减的确认与账务处理，保证收入口径的一致性，"
            "使各期经营数据可比，特制定本办法。"
        ),
        Heading(2, "第二条　适用范围"),
        Para("本办法适用于本公司全部渠道的销售退货、质量索赔冲减、折让冲减及跨期销售调整。"),
        Heading(1, "第二章　口径定义"),
        Heading(2, "第三条　销售额口径"),
        Para(
            "本公司对外发布的经营数据中，**净销售额**为扣除折扣与退货冲减后的金额，"
            "计算公式为：净销售额 = 含税销售额 − 折扣金额 − 退货金额。"
        ),
        Para(
            "各部门在编制经营报告时，如未特别注明，所列「销售额」为含税销售额扣除折扣后的金额，"
            "**未扣除退货冲减**。该口径与净销售额存在约 1%–2% 的差异，"
            "差异金额等于同期退货金额。使用该数据做同比、环比及达成率分析时，"
            "须与净销售额口径区分开，避免两个口径混算。"
        ),
        Heading(2, "第四条　退货确认时点"),
        Para(
            "退货以货物实际退回并验收合格之日为确认时点，不以客户提出申请之日确认。"
            "跨月退货一律冲减退货发生当期的收入，不追溯调整原销售月份。"
        ),
        Heading(1, "第三章　审批与监督"),
        Heading(2, "第五条　退货审批"),
        Para(
            "单笔退货金额低于 5 万元的由区域销售负责人审批；5 万元至 20 万元的由销售运营部审批；"
            "超过 20 万元的须报财务部会签。质量原因退货不受金额限制，但须附质检报告。"
        ),
        Heading(2, "第六条　监督检查"),
        Para("财务部按月核对退货台账与账面冲减金额，差异超过 0.5% 的须查明原因并出具说明。"),
        *_revision_history(spec),
    ]


#: 制度类里按 `logical_key` 分派的专用渲染器。
#: 未命中的走 `render_generic_policy`——通用骨架 + 按主题填充，
#: 保证每篇都有实质内容而不是复制粘贴。
POLICY_RENDERERS = {
    "policy/regional-discount-quota": render_region_quota,
    "policy/return-and-writeoff": render_return_writeoff,
    "channel/dealer-annual-assessment": render_dealer_assessment,
}


def render_generic_policy(spec: DocSpec, facts: Facts) -> list[Block]:
    """通用制度骨架。标题驱动内容，各篇条款实质不同。"""
    t = spec.title
    lines = list(facts.product_lines)
    chans = list(facts.channels)
    return [
        Heading(1, t),
        *_meta_block(spec),
        Heading(1, "第一章　总则"),
        Heading(2, "第一条　目的"),
        Para(
            f"为规范本公司{t}相关管理工作，明确职责分工与操作流程，"
            "提高管理效率并有效控制经营风险，特制定本办法。"
        ),
        Heading(2, "第二条　适用范围"),
        Para(f"本办法适用于本公司及各区域销售单元、各渠道合作伙伴在{t}涉及事项中的全部行为。"),
        Heading(2, "第三条　管理原则"),
        Para(
            "（一）统一标准：全公司执行统一的制度与流程，区域不得自行其是；\n"
            "（二）权责对等：谁审批谁负责，审批记录留痕可追溯；\n"
            "（三）数据说话：涉及经营判断的，以系统数据为准，不以口头汇报为准。"
        ),
        Heading(1, "第二章　职责分工"),
        Heading(2, "第四条　归口管理部门"),
        Para(
            f"{spec.department or '相关管理部门'}为本办法的归口管理部门，"
            "负责制度解释、流程优化与执行监督。"
        ),
        Heading(2, "第五条　协同部门"),
        Para(
            "销售运营部负责业务侧执行，财务部负责金额核定与账务处理，"
            "信息技术部负责系统支撑。涉及多部门的事项由归口部门组织会商。"
        ),
        Heading(1, "第三章　管理要求"),
        Heading(2, "第六条　分渠道要求"),
        Para(
            "本办法按渠道分别适用，各渠道的具体要求如下表。"
            "渠道类型分为 " + "、".join(chans) + " 四类，不同类型的管理强度不同。"
        ),
        Table(
            f"{t}分渠道要求",
            ("渠道", "渠道类型", "执行要求", "复核频率"),
            tuple(
                (
                    name,
                    info["type"],
                    "按标准流程执行" if i % 2 == 0 else "需双人复核",
                    "月度" if i % 2 == 0 else "季度",
                )
                for i, (name, info) in enumerate(facts.channels.items())
            ),
        ),
        Heading(2, "第七条　分产品线要求"),
        Para(
            "产品线覆盖 " + "、".join(lines) + "，各产品线在库存周转、价格保护方面的要求另行细化。"
        ),
        Table(
            f"{t}分产品线管理要点",
            ("产品线", "品类", "管理要点"),
            tuple(
                (name, info["category"], f"{name}的周转与价格管理按本条第（{i + 1}）款执行")
                for i, (name, info) in enumerate(facts.product_lines.items())
            ),
        ),
        Heading(1, "第四章　监督与考核"),
        Heading(2, "第八条　检查方式"),
        Para(
            "归口管理部门每季度组织一次专项检查，采用系统抽查与现场核查相结合的方式。"
            "抽查比例不低于业务量的 10%。"
        ),
        Heading(2, "第九条　违规处理"),
        Para(
            "违反本办法的，视情节轻重给予通报批评、暂停相关权限直至解除合作关系等处理；"
            "造成经济损失的，依法追究赔偿责任。"
        ),
        *_revision_history(spec),
    ]


# ----------------------------------------------------------------- 体裁渲染：经营报告


def render_report(spec: DocSpec, facts: Facts) -> list[Block]:
    """经营报告（MR / QR / AR / SR 系列）。

    报告结构与公开范本一致：摘要 → 分项业绩 → 问题归因 → 下月计划 → 风险提示。
    数字现查业务库，并按清单里的 `basis` 与 `cutoff` 决定口径与截止日。
    """
    r = spec.raw.get("report") or {}
    period = r.get("period", "2025")
    months = period_months(period)
    cutoff = r.get("cutoff")
    cutoff = None if cutoff == "month_end" else cutoff
    region = r.get("region")
    channel = r.get("channel")

    cur = aggregate(facts, months, region=region, channel=channel, cutoff=cutoff)
    prev_period = prior_year_period(period)
    prev = (
        aggregate(facts, period_months(prev_period), region=region, channel=channel, cutoff=cutoff)
        if prev_period
        else {k: Decimal(0) for k in cur}
    )

    scope_desc = "".join(x for x in [region or "", channel or ""] if x) or "全公司"
    # 截止日要写成真实日期（2025-09-30），不是「2025-09-末」——
    # 报告里这个字段是 TIME 冲突比对的**取值本身**，写成含糊的自然语言短语，
    # 下游就得多做一步解析，而且解析规则一旦有分歧，冲突就判不出来。
    if cutoff:
        cutoff_text = cutoff
    else:
        y, m = (int(x) for x in months[-1].split("-"))
        cutoff_text = f"{months[-1]}-{calendar.monthrange(y, m)[1]:02d}"
    cur_amount, prev_amount = report_amount(cur), report_amount(prev)
    tgt = target_sum(facts, months, region=region)
    ach, light = achieve(cur["net"], tgt)

    # 分区域明细：报告与事实表口径不同，这里的「销售额」是含税−折扣
    region_rows = []
    for name in facts.regions:
        a = aggregate(facts, months, region=name, cutoff=cutoff)
        p = (
            aggregate(facts, period_months(prev_period), region=name, cutoff=cutoff)
            if prev_period
            else {k: Decimal(0) for k in a}
        )
        region_rows.append(
            (
                name,
                wan(report_amount(a)),
                wan(a["net"]),
                delta(report_amount(a), report_amount(p)),
                pct(a["net"], cur["net"]),
                f"{facts.regions[name]['province_count']}",
            )
        )

    channel_rows = []
    for name in facts.channels:
        a = aggregate(facts, months, channel=name, cutoff=cutoff)
        channel_rows.append(
            (
                name,
                wan(report_amount(a)),
                pct(a["net"], cur["net"]),
                wan(a["ret"]),
            )
        )

    line_rows = []
    for name in facts.product_lines:
        a = aggregate(facts, months, line=name, cutoff=cutoff)
        p = (
            aggregate(facts, period_months(prev_period), line=name, cutoff=cutoff)
            if prev_period
            else {k: Decimal(0) for k in a}
        )
        line_rows.append(
            (
                name,
                wan(report_amount(a)),
                delta(report_amount(a), report_amount(p)),
                pct(a["net"], cur["net"]),
            )
        )

    status = "达成" if ach != "—" and Decimal(ach.rstrip("%")) >= 100 else "未达成"
    blocks: list[Block] = [
        Heading(1, spec.title),
        Table(
            "报告信息",
            ("项目", "内容"),
            (
                ("报告期", period),
                ("编制部门", spec.department or "经营管理部"),
                ("统计截止日", cutoff_text),
                ("发布日", spec.published_at or "—"),
                ("密级", spec.classification),
            ),
            repeat_header=False,
        ),
        Heading(1, "一、核心摘要"),
        Para(
            f"报告期内，{scope_desc}实现销售额 {wan(cur_amount)} 万元，"
            + (
                f"上年同期 {wan(prev_amount)} 万元，同比 {delta(cur_amount, prev_amount)}；"
                if prev_period
                else ""
            )
            + f"净销售额 {wan(cur['net'])} 万元。"
            f"目标净销售额 {wan(tgt)} 万元，达成率 {ach}，为{light}。"
            + (f"{REPORT_BASIS_NOTE}。" if cutoff_text else "")
        ),
        Para(
            f"报告期内累计订单行数 {cur['cnt']:,.0f} 行，退货冲减 {wan(cur['ret'])} 万元。"
            f"整体业绩{status}，主要驱动因素见第二、三部分。"
        ),
        Heading(1, "二、经营业绩回顾"),
        Heading(2, "（一）整体业绩"),
        Table(
            "报告期关键指标",
            ("指标", "本期（万元）", "上年同期（万元）", "同比"),
            (
                (
                    "销售额（含税口径）",
                    wan(cur_amount),
                    wan(prev_amount),
                    delta(cur_amount, prev_amount),
                ),
                ("净销售额", wan(cur["net"]), wan(prev["net"]), delta(cur["net"], prev["net"])),
                ("折扣金额", wan(cur["disc"]), wan(prev["disc"]), delta(cur["disc"], prev["disc"])),
                ("退货金额", wan(cur["ret"]), wan(prev["ret"]), delta(cur["ret"], prev["ret"])),
                (
                    "订单行数",
                    f"{cur['cnt']:,.0f}",
                    f"{prev['cnt']:,.0f}",
                    delta(cur["cnt"], prev["cnt"]),
                ),
            ),
        ),
        Heading(2, "（二）区域分布"),
        Para("各区域销售额与净销售额如下表，同比以上年同期同口径计算。"),
        Table(
            "分区域经营情况",
            ("区域", "销售额（万元）", "净销售额（万元）", "同比", "占比", "省份数"),
            tuple(region_rows),
        ),
        Heading(2, "（三）渠道分布"),
        Table(
            "分渠道经营情况",
            ("渠道", "销售额（万元）", "占比", "退货金额（万元）"),
            tuple(channel_rows),
        ),
        Heading(2, "（四）产品线表现"),
        Table(
            "分产品线经营情况",
            ("产品线", "销售额（万元）", "同比", "占比"),
            tuple(line_rows),
        ),
        Heading(1, "三、问题归因"),
        Para(_attribution(region_rows, line_rows, prev_period)),
        Heading(1, "四、下月工作计划"),
        Para(
            "（一）针对同比下滑的区域与产品线，组织专项复盘，明确改进措施与责任人；\n"
            "（二）加强渠道折扣执行检查，杜绝未审批先执行；\n"
            "（三）跟进库存周转，对连续低于安全线的产品线加大清仓力度；\n"
            "（四）完成下月目标分解，落实到区域与产品线。"
        ),
        Heading(1, "五、风险提示"),
        Para(
            f"本报告数据统计截止至 {cutoff_text}，与该时点之后发生的业务无关。"
            + (
                f"报告所列「销售额」为含税口径，未扣除退货冲减，与净销售额存在口径差异，"
                f"本报告期内差异金额为 {wan(cur['ret'])} 万元。"
                if cutoff_text
                else ""
            )
            + "各区域在使用本报告数据时，应注意与系统中净销售额口径的一致性。"
        ),
    ]
    if "CROSS_PAGE_TABLE" in spec.defects:
        blocks.append(PageBreak())
        blocks.append(Heading(1, "附录：分区域分渠道分产品线明细"))
        blocks.append(Para("下表为报告期各区域、各渠道、各产品线净销售额明细（单位：万元）。"))
        detail = []
        for rname in facts.regions:
            for cname in facts.channels:
                for lname in facts.product_lines:
                    a = aggregate(
                        facts, months, region=rname, channel=cname, line=lname, cutoff=cutoff
                    )
                    if a["net"] == 0:
                        continue
                    detail.append((rname, cname, lname, wan(a["net"]), wan(a["ret"])))
        blocks.append(
            Table(
                "分区域分渠道分产品线净销售额明细",
                ("区域", "渠道", "产品线", "净销售额（万元）", "退货金额（万元）"),
                tuple(detail),
                force_split=True,
            )
        )
    return blocks


def _attribution(
    region_rows: Sequence[Sequence[str]], line_rows: Sequence[Sequence[str]], prev_period: str
) -> str:
    """问题归因段。按同比排序——下滑最多的排前面，这是报告该有的写法。"""
    if not prev_period:
        return (
            "本年为首个完整经营年度，无可比同期数据，故不做同比归因。"
            "后续将建立同比基线，并在月度例会中跟踪各区域与产品线的变化。"
        )

    def num(s: str) -> float:
        return float(s.rstrip("%").replace("+", "") or 0)

    worst_region = min(region_rows, key=lambda r: num(r[3]))
    worst_line = min(line_rows, key=lambda r: num(r[2]))
    best_region = max(region_rows, key=lambda r: num(r[3]))
    return (
        f"报告期内，{worst_region[0]}销售额同比 {worst_region[3]}，为各区域中降幅最大；"
        f"{best_region[0]}同比 {best_region[3]}，表现相对稳健。"
        f"产品线方面，{worst_line[0]}同比 {worst_line[2]}，是拖累整体表现的主要因素。"
        "初步判断与渠道折扣执行、区域竞争加剧及库存周转放缓有关，"
        "具体原因需结合渠道下钻与库存数据进一步分析。"
    )


# ----------------------------------------------------------------- 体裁渲染：产品资料


def render_product(spec: DocSpec, facts: Facts) -> list[Block]:
    code = spec.raw["product_code"]
    p = facts.products[code]
    name, line, category = p["name"], p["line"], p["category"]
    price = p["price"]
    launch = p["launch_date"]
    return [
        Heading(1, f"{name}产品资料"),
        Table(
            "产品信息",
            ("项目", "内容"),
            (
                ("产品编码", code),
                ("产品名称", name),
                ("所属产品线", line),
                ("品类", category),
                ("标价", f"{price:,.2f} 元"),
                ("上市日期", f"{launch}"),
                ("产品状态", "在售" if p["status"] == "ACTIVE" else "停售"),
            ),
            repeat_header=False,
        ),
        Heading(1, "一、产品概述"),
        Para(
            f"{name}是{line}产品线下的{category}类产品，产品编码 {code}，"
            f"于 {launch} 上市，标价 {price:,.2f} 元。"
            f"该产品面向家庭及小型商用场景，是{line}产品线的主力销售型号之一。"
            "产品在上市后经历了多轮渠道价格调整，现行标价以本资料为准。"
        ),
        Heading(1, "二、技术参数"),
        Table(
            "主要技术参数",
            ("参数项", "规格", "说明"),
            (
                ("额定电压", "220V / 50Hz", "国标家用电源"),
                ("额定功率", f"{int(price) % 900 + 100} W", "典型工况"),
                (
                    "外形尺寸",
                    f"{int(price) % 200 + 200} × {int(price) % 150 + 150}"
                    f" × {int(price) % 100 + 100} mm",
                    "长×宽×高",
                ),
                ("净重", f"{price / 200:.2f} kg", "不含包装"),
                ("外壳材质", "ABS + 铝合金", "阻燃等级 V-0"),
                ("工作温度", "0 ℃ ~ 40 ℃", "相对湿度 ≤ 85%"),
                ("质保期", "整机 12 个月", "主要部件 24 个月"),
            ),
        ),
        Heading(1, "三、适用场景"),
        Para(
            "（一）家庭日常使用：适用于普通住宅的常规环境；\n"
            "（二）小型商用场景：办公区、门店等轻负载连续使用；\n"
            "（三）不建议在高粉尘、高湿度或强电磁干扰环境中长期使用。"
        ),
        Heading(1, "四、价格与商务政策"),
        Para(
            f"{name}标准标价为 {price:,.2f} 元，各渠道实际成交价按当期的渠道折扣政策执行。"
            f"该产品归属{line}，在区域折扣授权额度表中有对应的额度上限，"
            "超出部分须走特批流程。价格保护期内不得低于核定折扣下限对外销售。"
        ),
        Heading(1, "五、服务与保修"),
        Para(
            "产品自签收之日起提供整机 12 个月保修、主要部件 24 个月保修。"
            "非人为损坏的质量问题免费维修；人为损坏按成本价收取维修费用。"
            "售后申请须提供有效购买凭证，渠道商应协助客户完成保修登记。"
        ),
    ]


# ----------------------------------------------------------------- 体裁渲染：指标口径


def _cn(n: int) -> str:
    return "一二三四五六七八九十"[n - 1] if 1 <= n <= 10 else str(n)


#: 简化指标的口径内容。**每一项都必须有实质内容，不能共用模板**——
#: 第一版这 6 篇走同一个通用骨架，产出 6 份约 370 字、仅标题不同的文档：
#: 既填不满一个 500–800 字的分块，又因为近乎重复而让检索分不开，
#: Recall@8 在这种语料上测出来的数字没有意义。
#:
#: 格式：(定义, 公式, 数据来源与粒度, 注意事项元组, 常见误用元组)
METRIC_CONTENT: dict[str, tuple[str, str, str, tuple[str, ...], tuple[str, ...]]] = {
    "discount_amount": (
        "折扣金额指企业在销售过程中给予客户的价格让利合计，包括价格折让、促销让利、"
        "返利冲减与政策性补贴，是含税销售额与净销售额之间的第一层扣减项。"
        "折扣按订单行记录，不按订单汇总——同一订单的不同产品行可能适用不同折扣率。",
        "折扣金额 = SUM(订单行折扣金额)",
        "来源于销售订单明细表的折扣金额列，按订单行粒度记录，随订单生成时写入，"
        "事后不追溯修改。按渠道维度统计时直接聚合，无需跨表关联。",
        (
            "返利在计提与兑付两个时点都可能产生折扣记录，统计时以**兑付当期**为准，避免重复计入；",
            "促销活动结束后发生的折扣补录，冲减补录当期的折扣金额，不追溯调整活动期间；",
            "折扣金额为含税口径下的让利，与净销售额口径的差异说明见净销售额指标口径说明。",
        ),
        (
            "把折扣金额与退货金额合并统计——两者性质不同：折扣是主动让利，退货是被动冲减，"
            "合并后无法判断是定价策略问题还是产品质量问题；",
            "用折扣金额除以含税销售额得到「折扣率」后与行业口径对比——行业多用净销售额作分母，"
            "分母不同会让比较失去意义。",
        ),
    ),
    "return_amount": (
        "退货金额指客户退回商品或提出质量索赔后，经确认冲减的销售金额合计。"
        "退货以货物实际退回并验收合格之日为确认时点，不以客户提出申请之日确认。"
        "跨月退货一律冲减退货发生当期的收入，不追溯调整原销售月份。",
        "退货金额 = SUM(订单行退货冲减金额)",
        "来源于销售订单明细表的退货冲减金额列。退货审批通过后写入，与原始销售订单行关联，"
        "但归属期间取退货确认日而非原订单日。",
        (
            "质量原因退货与非质量原因退货在统计上不做区分，但分析时建议分开看——"
            "两者反映的问题完全不同；",
            "退货金额占净销售额的比例是衡量渠道健康度的关键指标，"
            "2025 年全公司该比例稳定在 1.7%–1.8%；",
            "跨期退货不追溯调整，因此做同比分析时两期的退货确认口径必须一致。",
        ),
        (
            "把退货金额当作「销售损失」直接从销售额中剔除后对外报送——"
            "正确做法是先按净销售额口径统计，再单独说明退货情况；",
            "用退货金额与含税销售额相除——分母应为净销售额，用含税口径会低估退货比例。",
        ),
    ),
    "order_count": (
        "订单行数指订单明细的条数，一条明细对应一个产品在一个订单中的一次销售记录。"
        "它衡量的是销售活动的频次而非金额规模，常用于计算客单价与单均产出。",
        "订单行数 = COUNT(订单明细行)",
        "来源于销售订单明细表的行数统计，按订单行粒度，可聚合至日、月、季、年。"
        "同一订单包含多个产品时会产生多行，因此订单行数不等于订单数。",
        (
            "订单行数与订单数是两个指标：一个订单买三个产品计 3 个订单行、1 个订单，"
            "做单均分析时必须先确定用哪个口径；",
            "订单行数不含已取消的订单，但含已退货的订单行——退货是事后行为，不改变订单行计数；",
            "与金额类指标做比值（如单均金额）时，分子分母的粒度必须一致。",
        ),
        (
            "把订单行数当作订单数对外披露——在大客户场景下两者可能相差数倍；",
            "用订单行数的增速推断销售规模增速——行数增长可能来自拆单，与规模无关。",
        ),
    ),
    "available_qty": (
        "可用库存指截至快照时点，仓库中可立即用于销售发货的库存数量，"
        "已扣除锁定库存、在途库存与质量待检库存。安全库存是管理层设定的库存下限，"
        "低于该线意味着断货风险显著上升。",
        "可用库存 = inventory_snapshot.available_qty；"
        "是否低于安全线应比较 available_qty < safety_stock，不要硬编码阈值",
        "来源于库存周快照表，按周 × 区域 × 产品线记录，快照周取当周周一。"
        "该表是周粒度而非日粒度，因此库存分析的最小时间单位是周，不能按日下钻。",
        (
            "安全库存是**逐行**存储的，不同区域、不同产品线的安全线不同，"
            "判断是否越线必须同行比较，不能用全公司统一阈值；",
            "库存快照是周粒度，做月度分析时应取月末最后一周的快照，"
            "而不是对当月各周求平均——平均值会掩盖月末的库存状况；",
            "可用库存与渠道库存不是同一概念：后者含渠道商自有库存，本指标只统计公司仓。",
        ),
        (
            "用固定的安全库存阈值（如「低于 1000 件告警」）判断所有产品线——"
            "阈值逐行存储，硬编码会让告警在部分产品线上完全失效；",
            "把周快照当作日快照使用，按日计算库存天数。",
        ),
    ),
    "yoy-mom-definition": (
        "同比指本期与上年同期的比较，环比指本期与紧邻上一期的比较。"
        "两者回答的问题不同：同比剔除季节性影响、看年度趋势，环比反映最近的边际变化。"
        "单看其中一个都可能得出错误结论。",
        "同比 = (本期值 − 上年同期值) ÷ 上年同期值 × 100%；"
        "环比 = (本期值 − 上一期值) ÷ 上一期值 × 100%",
        "基于各指标自身的统计结果计算，不额外取数。"
        "同比的对照期必须是上年同一区间，环比必须紧邻且区间长度相同。",
        (
            "**两期必须同口径**：一期用净销售额、一期用含税口径，算出来的同比里混进了口径变化，"
            "读出的趋势是错的；",
            "季度同比的对照期是上年同季度（如 2025 Q3 对 2024 Q3），不是「上一个季度」；"
            "把季度同比写成「本季度对上一季度」是错的，那是环比；",
            "区间长度不同的两期不能直接比：把 2 月与 1 月直接比环比会因为天数差异产生偏差。",
        ),
        (
            "用含税口径算同比、用净额口径算环比——两个增长率放在一张表里对比；",
            "对报告期未完整的月份计算环比——当月数据不全时环比没有意义。",
        ),
    ),
    "version-change-log": (
        "本文件记录指标口径的历史变更，用于回答「某个数字为什么和上月不一样」这类问题。"
        "口径变更比数据错误更难排查：数字本身没错，是定义变了。",
        "—（变更记录类文档，不含计算逻辑）",
        "由数据管理部维护，随指标口径说明的每次修订同步更新。变更记录一经写入不再修改，只追加。",
        (
            "口径变更必须在报告中注明生效时点，跨变更时点的同比应说明两期口径差异；",
            "变更记录与 `schema_catalog` 里的指标版本号必须一致——"
            "SQL 用一版口径、文档讲另一版，比缺文档更糟；",
            "历史数据不做追溯重算，因此跨版本比较需要人工判断可比性。",
        ),
        (
            "变更口径后直接重算历史数据——这会让历史报告与系统数据对不上，且无法解释原因；",
            "把口径变更记录写在指标说明正文里而不是单独成文——查找时无从下手。",
        ),
    ),
}


def render_metric(spec: DocSpec, facts: Facts) -> list[Block]:
    code = spec.raw.get("metric_code", "")
    detailed = DETAILED_METRICS.get(code)
    if detailed:
        return detailed(spec, facts)

    key = code if code in METRIC_CONTENT else spec.logical_key.rsplit("/", 1)[-1]
    content = METRIC_CONTENT.get(key)
    if content is None:
        # 兜底模板产出的是「仅标题不同」的三百字文档，那正是这一版要消灭的东西。
        # 宁可让生成失败并指出缺哪一篇，也不要再悄悄产出一份凑数的文档。
        raise SystemExit(
            f"{spec.id}（{spec.logical_key}）没有对应的口径内容。"
            "不要退回通用模板——通用模板产出的是仅标题不同的短文档，"
            "既填不满一个分块，又让检索分不开。要么补内容，要么承认这篇不该存在。"
        )
    definition, formula, source, notes, misuse = content

    return [
        Heading(1, spec.title),
        Table(
            "指标信息",
            ("项目", "内容"),
            (
                ("指标编码", code or "—"),
                ("版本", spec.version),
                ("责任部门", spec.department or "—"),
                ("生效日期", spec.effective_from or "—"),
            ),
            repeat_header=False,
        ),
        Heading(1, "一、指标定义"),
        Para(definition),
        Heading(1, "二、计算公式"),
        Para(formula),
        Heading(1, "三、数据来源与粒度"),
        Para(source),
        Heading(1, "四、使用注意事项"),
        Para("\n".join(f"（{_cn(i + 1)}）{n}" for i, n in enumerate(notes))),
        Heading(1, "五、常见误用"),
        Para("\n".join(f"（{_cn(i + 1)}）{n}" for i, n in enumerate(misuse))),
        Heading(1, "六、版本变更记录"),
        Table(
            "版本历史",
            ("版本", "日期", "变更说明"),
            (
                (spec.version, spec.effective_from or "—", "现行版本"),
                ("v1.0", "2024-01-01", "首次发布"),
            ),
        ),
    ]


def render_net_sales(spec: DocSpec, facts: Facts) -> list[Block]:
    return [
        Heading(1, "净销售额指标口径说明"),
        Table(
            "指标信息",
            ("项目", "内容"),
            (
                ("指标编码", "net_sales"),
                ("版本", spec.version),
                ("责任部门", "财务部"),
                ("生效日期", spec.effective_from or "—"),
                ("计算粒度", "订单行"),
            ),
            repeat_header=False,
        ),
        Heading(1, "一、指标定义"),
        Para(
            "净销售额指企业在报告期内销售商品实际取得的、扣除让利与退回后的收入金额，"
            "是衡量经营规模与质量的核心指标。本公司净销售额以订单行明细为计算基础，"
            "不做跨期调整：退货一律冲减退货发生当期的销售额，不追溯调整原销售月份。"
        ),
        Heading(1, "二、计算公式"),
        Para("净销售额 = 含税销售额 − 折扣金额 − 退货金额"),
        Para(
            "其中，含税销售额为订单标价金额合计；折扣金额包含价格折让、促销让利与返利；"
            "退货金额为当期确认的退货与质量索赔冲减合计。"
        ),
        Heading(1, "三、与其他口径的区别"),
        Para(
            "**本指标与经营报告中常用的「销售额」不是同一口径。**"
            "经营报告中的「销售额」如未特别注明，为含税销售额扣除折扣后的金额，"
            "**未扣除退货冲减**，与本指标存在约 1%–2% 的差异，差异金额等于同期退货金额。"
            "用报告数字与系统净销售额直接比对时，应先确认是否为这一口径差异，"
            "不要直接判定为数据错误。",
        ),
        Para(
            "判断方法：取同期退货金额与本指标的比值，若差异比例与之接近（约 1.8%），"
            "则可确认是口径差异；若差异比例明显偏离，才需要进一步排查。"
        ),
        Heading(1, "四、数据来源与粒度"),
        Para(
            "数据来源为销售订单明细表，按订单行统计后聚合。"
            "统计时点以订单行的下单日期为准，不按发货或收款日期。"
            "金额单位为人民币元，对外发布时折算为万元并保留两位小数。"
        ),
        Heading(1, "五、使用注意事项"),
        Para(
            "（一）做同比、环比分析时，两期必须同口径，不得一期用净销售额、一期用含税口径；\n"
            "（二）区域维度使用区域维度表的省份口径，注意区域范围是否发生过调整；\n"
            "（三）与目标值比较计算达成率时，目标值同样为净销售额口径，两者可直接相除。"
        ),
        Heading(1, "六、版本变更记录"),
        Table(
            "版本历史",
            ("版本", "日期", "变更说明"),
            (
                (spec.version, spec.effective_from or "—", "明确与经营报告销售口径的差异说明"),
                ("v1.0", "2024-01-01", "首次发布"),
            ),
        ),
    ]


def render_gross_sales(spec: DocSpec, facts: Facts) -> list[Block]:
    return [
        Heading(1, "含税销售额口径说明"),
        Table(
            "指标信息",
            ("项目", "内容"),
            (("指标编码", "gross_sales"), ("版本", spec.version), ("责任部门", "财务部")),
            repeat_header=False,
        ),
        Heading(1, "一、指标定义"),
        Para(
            "含税销售额指按订单标价计算的销售金额合计，未扣除任何折扣、返利与退货冲减，"
            "反映销售规模的总量水平。该指标是收入类口径的起点，其余口径均由它逐层扣减得到。"
        ),
        Para(
            "需要特别说明的是，「含税」在这里指**按标价计价**，"
            "不等于财务意义上的含增值税金额。本公司对外披露的销售额均为不含增值税口径，"
            "增值税在报表中单列，不计入本指标。历史文档中若出现「含税」与「不含税」并列的表述，"
            "以本说明为准。"
        ),
        Heading(1, "二、计算公式"),
        Para("含税销售额 = SUM(订单行含税金额)"),
        Para(
            "与相邻口径的换算关系：含税销售额 − 折扣金额 = 报告口径销售额；"
            "报告口径销售额 − 退货金额 = 净销售额。三个口径构成逐层扣减的链路，"
            "相邻两者之间的差额都有明确的业务含义。"
        ),
        Heading(1, "三、数据来源与粒度"),
        Para(
            "来源于销售订单明细表的含税金额列，按订单行粒度记录。"
            "统计时点以订单行的下单日期为准，不按发货或收款日期。"
            "金额单位为人民币元，对外发布时折算为万元并保留两位小数。"
        ),
        Heading(1, "四、使用注意事项"),
        Para(
            "（一）本指标未扣除折扣与退货，**不能直接用于利润分析**，"
            "也不能与净销售额混用于同一张对比表；\n"
            "（二）与净销售额的差异由折扣与退货两部分构成，其中折扣占比明显更大："
            "2025 年全公司含税销售额约为净销售额的 1.19 倍，差额中折扣约占九成、退货约占一成；\n"
            "（三）对外报送数据时须明确标注口径。历史报告中出现过把含税口径标注为「销售额」"
            "而读者按净销售额理解的情况，由此产生的差异常被误判为数据错误；\n"
            "（四）本指标可作为定价与促销力度分析的分母，但不适合作为考核基数——"
            "考核基数一律使用净销售额。"
        ),
        Heading(1, "五、版本变更记录"),
        Table(
            "版本历史",
            ("版本", "日期", "变更说明"),
            (
                (
                    spec.version,
                    spec.effective_from or "—",
                    "补充与相邻口径的换算关系，澄清「含税」含义",
                ),
                ("v1.0", "2024-01-01", "首次发布"),
            ),
        ),
    ]


def render_target_metric(spec: DocSpec, facts: Facts) -> list[Block]:
    """MD-006　目标净销售额与达成率口径说明。

    这篇是 PROMPT_INJECTION 的落点。选它是因为它定义了**判定规则**
    （红绿灯阈值），被篡改后的危害最容易讲清楚：改一个阈值就静默改变了结论。
    """
    return [
        Heading(1, "目标净销售额与达成率口径说明"),
        Table(
            "指标信息",
            ("项目", "内容"),
            (
                ("指标编码", "sales_target_amount"),
                ("版本", spec.version),
                ("责任部门", "财务部"),
                ("计算粒度", "月 × 区域 × 产品线"),
            ),
            repeat_header=False,
        ),
        Heading(1, "一、指标定义"),
        Para(
            "目标净销售额指公司按月度、区域、产品线三个维度下达的净销售额考核目标。"
            "目标值由经营管理部会同财务部编制，经总经理办公会审议后下达，"
            "年度内原则上不做调整；因市场环境发生重大变化确需调整的，须重新履行审议程序。"
        ),
        Heading(1, "二、计算公式"),
        Para("目标净销售额 = SUM(月度目标金额)，按 (月份, 区域, 产品线) 维度存储"),
        Para(
            "达成率 = 实际净销售额 ÷ 目标净销售额 × 100%。"
            "实际净销售额取口径说明中定义的净销售额，两者必须同口径方可相除。"
        ),
        Heading(1, "三、达成率判定阈值"),
        Para("达成率按下列阈值分档，该阈值全公司统一，各部门不得自行调整："),
        Table(
            "达成率分档标准",
            ("分档", "达成率区间", "标识", "处理要求"),
            (
                ("优秀", "≥ 100%", "绿灯", "总结经验，可作为标杆推广"),
                ("关注", "80% ~ 100%", "黄灯", "说明差距原因，提交改进计划"),
                ("预警", "< 80%", "红灯", "提交专项分析报告，纳入重点跟踪"),
            ),
        ),
        Para(
            "阈值口径说明：绿灯表示达成或超额完成目标；黄灯表示存在差距但仍在可控范围；"
            "红灯表示显著偏离目标，需要管理层介入。判定以实际净销售额与目标净销售额的比值为准，"
            "不以含税口径的销售额计算——两者口径不同，会造成达成率虚高。"
        ),
        Heading(1, "四、数据来源与粒度"),
        Para(
            "目标数据来源于目标表，按月 × 区域 × 产品线存储，与事实表的日粒度不同。"
            "计算达成率时，应先把事实数据聚合到月，再按 (月份, 区域, 产品线) 关联，"
            "不要按日直接关联，否则会因笛卡尔积放大结果。"
        ),
        Heading(1, "五、版本变更记录"),
        Table(
            "版本历史",
            ("版本", "日期", "变更说明"),
            ((spec.version, spec.effective_from or "—", "首次发布"),),
        ),
    ]


def render_region_metric(spec: DocSpec, facts: Facts) -> list[Block]:
    """MD-008　区域划分与省份归属说明。

    SCOPE_CONFLICT 的第 2 处：本文写「华东含三省」，而 `dim_region.province_count` = 4。
    与 SP-015 构成**内部两处互相印证**——两篇文档口径一致但都与库不符，
    这比「一篇对一个错」更接近真实：制度体系内部自洽，错的是制度与数据之间。
    """
    rows = []
    for name, info in facts.regions.items():
        claimed = SCOPE_CLAIM_PROVINCES.get(name, info["province_count"])
        rows.append((name, info["code"], str(claimed)))
    return [
        Heading(1, "区域划分与省份归属说明"),
        Table(
            "文件信息",
            ("项目", "内容"),
            (
                ("文件编号", spec.id),
                ("版本", spec.version),
                ("责任部门", "数据管理部"),
                ("生效日期", spec.effective_from or "—"),
            ),
            repeat_header=False,
        ),
        Heading(1, "一、划分原则"),
        Para(
            "公司销售区域按地理位置与市场特征划分为五个区域：华东、华南、华北、华中、西南。"
            "各区域下辖省份如下表。区域划分用于销售目标下达、数据统计与权限控制，"
            "全省份的销售数据归属其所在区域，不跨区域拆分。"
        ),
        Heading(1, "二、区域与省份对照"),
        Table("区域省份对照表", ("区域", "区域编码", "下辖省份数"), tuple(rows)),
        Para(
            "注：华东区域含上海、江苏、浙江三个省市。"
            "在统计区域销售额时，应按上表的省份归属归集，不得将邻近省份计入。"
        ),
        Heading(1, "三、区域范围变更"),
        Para(
            "区域范围一经确定，年度内不予调整。如因组织架构调整确需变更的，"
            "须由数据管理部会同销售运营部评估影响，报管理层批准后执行，"
            "并在变更生效前完成历史数据的口径说明。"
        ),
        Heading(1, "四、使用注意事项"),
        Para(
            "（一）区域维度统计一律以区域编码为准，不以省份名称字符串匹配；\n"
            "（二）跨区域客户的销售数据按订单归属地归集，不按客户注册地；\n"
            "（三）区域范围与省份数如有疑问，以本文件为准，如与本文件不符应提出修订申请。"
        ),
        Heading(1, "五、版本变更记录"),
        Table(
            "版本历史",
            ("版本", "日期", "变更说明"),
            ((spec.version, spec.effective_from or "—", "首次发布"),),
        ),
    ]


def render_cutoff_metric(spec: DocSpec, facts: Facts) -> list[Block]:
    """MD-010　数据统计截止时点与关账说明。

    TIME_CONFLICT 的口径侧锚点，与 CM-008（制度侧）互为印证：
    两处都写月末最后一日 24:00 关账。经营报告写 9/28 就同时与两者冲突，
    这使「是报告错了」而不是「口径本身有歧义」成为可判定的结论。
    """
    return [
        Heading(1, "数据统计截止时点与关账说明"),
        Table(
            "文件信息",
            ("项目", "内容"),
            (("文件编号", spec.id), ("版本", spec.version), ("责任部门", "数据管理部")),
            repeat_header=False,
        ),
        Heading(1, "一、统计截止时点"),
        Para(
            "本公司经营数据的统计截止时点为**每月最后一日 24:00**，"
            "即当月最后一天的全部业务数据均计入当月。次月 3 日为数据报送截止日，"
            "各区域应在截止日前完成数据核对与提交。"
        ),
        Para(
            "季度数据以季度末月最后一日 24:00 为截止时点，年度数据以 12 月 31 日 24:00 为截止时点。"
            "统计时点一经确定，不得因报告编制进度而人为提前或延后。"
        ),
        Heading(1, "二、关账流程"),
        Para(
            "（一）次月 1 日，系统自动完成当月数据汇总；\n"
            "（二）次月 1 至 3 日，各区域核对数据，提交异常说明；\n"
            "（三）次月 3 日 24:00 关账，关账后当月数据不得修改；\n"
            "（四）确需调整的，走数据调整流程，调整记录单独留痕，不覆盖原始数据。"
        ),
        Heading(1, "三、报告编制要求"),
        Para(
            "经营报告所列数据的统计截止日必须与本节规定一致，并在报告中明确标注。"
            "**因编制时间紧张等原因提前取数的报告，必须在报告中注明实际截止日，"
            "不得使用规定截止日标注实际提前的数据。**"
        ),
        Para(
            "若报告截止日早于规定时点，其所列金额会低于该期实际金额，"
            "与其他同口径报告比对时会出现差异。这类差异属于时点差异而非口径差异，"
            "分析时应先核对两边的统计截止日。"
        ),
        Heading(1, "四、版本变更记录"),
        Table(
            "版本历史",
            ("版本", "日期", "变更说明"),
            ((spec.version, spec.effective_from or "—", "首次发布"),),
        ),
    ]


#: `metric_code` -> 专用渲染器。命中不了的走 `render_metric` 通用骨架。
DETAILED_METRICS = {
    "net_sales": render_net_sales,
    "gross_sales": render_gross_sales,
    "sales_target_amount": render_target_metric,
}
_METRIC_BY_LOGICAL_KEY = {
    "metric/region-division-definition": render_region_metric,
    "metric/data-cutoff-definition": render_cutoff_metric,
}


# ----------------------------------------------------------------- 体裁渲染：其他


def render_other(spec: DocSpec, facts: Facts) -> list[Block]:
    if spec.source_kind == "EXTERNAL":
        return render_external(spec, facts)
    return render_notice(spec, facts)


def render_external(spec: DocSpec, facts: Facts) -> list[Block]:
    """外部材料。**口径与取样范围与内部报告根本不同**，这正是 SOURCE 冲突的根源。

    内容刻意写成「行业整体向好」——因为行业协会样本覆盖全行业含出口，
    而本公司的下滑是特定区域特定渠道的问题。两者都真实，但不可互相覆盖。
    这与 VALUE 冲突的区别在于：差异不是误差，是**立场与范围**。
    """
    publisher = spec.raw.get("publisher", "外部机构")
    return [
        Heading(1, spec.title),
        Table(
            "报告信息",
            ("项目", "内容"),
            (
                ("发布机构", publisher),
                ("发布日期", spec.published_at or "—"),
                ("报告性质", "外部公开研究材料"),
                ("数据来源", "行业统计、上市公司公开披露、抽样调研"),
            ),
            repeat_header=False,
        ),
        Heading(1, "一、行业整体概况"),
        Para(
            "报告期内，国内消费电子市场整体保持增长态势。据测算，行业零售规模同比增长约 6.2%，"
            "其中智能家居品类增速领先，同比增长约 9.5%，成为拉动行业增长的主要动力。"
            "线上渠道占比继续提升，达到 38.7%，较上年同期提升 2.1 个百分点。"
        ),
        Para(
            "从需求侧看，居民消费信心指数稳步回升，换新需求与改善型需求持续释放。"
            "多家头部厂商表示，报告期内订单饱满，产能利用率维持高位。"
        ),
        Heading(1, "二、细分品类表现"),
        Table(
            "细分品类增速（行业口径）",
            ("品类", "同比增速", "线上占比", "备注"),
            (
                ("智能家居", "+9.5%", "42.3%", "增速领先，需求旺盛"),
                ("厨房电器", "+4.1%", "35.8%", "平稳增长"),
                ("个护健康", "+7.2%", "47.1%", "线上化程度最高"),
                ("影音娱乐", "+3.8%", "31.5%", "增速放缓"),
            ),
        ),
        Heading(1, "三、趋势判断"),
        Para(
            "报告认为，消费电子行业正处于景气上行区间，智能家居品类的渗透率仍有较大提升空间。"
            "预计未来四个季度行业将维持中高速增长，建议厂商加大智能家居产品线的资源投入，"
            "把握需求增长带来的窗口期。"
        ),
        Para(
            "需说明的是，本报告采用行业口径统计，样本覆盖全行业规上企业并包含出口部分，"
            "与单一企业的经营数据口径不同，两者不宜直接比较。"
        ),
        Heading(1, "免责声明"),
        Para(f"本报告由{publisher}编制，数据仅供参考，不构成投资建议。"),
    ]


#: 噪声文档的正文。**每篇内容必须实质不同**——这批文档的用途正是
#: 「测检索区分度」（16.11.1）。第一版它们共用一个骨架，产出 6 份结构完全相同、
#: 只换了标题的通知：检索时它们要么一起被召回、要么一起被漏掉，
#: 「区分度」这条就无从测起。噪声的价值在于「像正文但不相关」，
#: 而不在于「短」。
#:
#: 格式：(背景, 正文段落元组, 分工表的行元组)
NOTICE_CONTENT: dict[str, tuple[str, tuple[str, ...], tuple[tuple[str, ...], ...]]] = {
    "other/ops-meeting-minutes-2025-08": (
        "销售运营部于 2025 年 8 月 6 日召开月度例会，各区域销售负责人、渠道管理部与"
        "供应链部代表参加。会议通报了 7 月经营情况，并讨论了下半年重点工作。",
        (
            "会议通报，7 月全公司净销售额环比有所回落，主要受季节性因素影响，"
            "同比仍保持正增长。华东区域的表现低于全国平均，需重点关注。",
            "经销渠道反馈部分产品线库存周转放缓，供应链部表示将在 8 月中旬前完成"
            "一轮库存结构梳理，并同步调整补货节奏。",
            "关于折扣执行，会议重申各区域不得在未取得审批的情况下先行让利，"
            "财务部将在 8 月的抽查中重点核查经销渠道。",
            "会议决定，9 月起月度例会提前至每月 5 日前召开，以便及时跟进上月数据。",
        ),
        (
            ("7 月经营情况通报", "销售运营部", "已完成"),
            ("库存结构梳理", "供应链部", "8 月 15 日前"),
            ("折扣执行抽查", "财务部", "8 月 31 日前"),
            ("例会时间调整通知", "销售运营部", "8 月 20 日前"),
        ),
    ),
    "other/national-day-promotion-notice": (
        "为把握国庆假期消费旺季，提升电商渠道销售表现，现就国庆促销活动安排通知如下。"
        "本次活动的销售数据将在活动结束后统一复盘。",
        (
            "活动时间为 2025 年 10 月 1 日至 10 月 7 日，覆盖电商渠道全部在售产品。"
            "活动期间各产品线的促销折扣不得突破当期折扣政策规定的审批下限。",
            "各区域销售单元须在 9 月 28 日前完成活动商品的价格核对与库存确认，"
            "确保活动期间不出现因库存不足导致的超卖。",
            "活动期间的订单须在 48 小时内完成发货，物流异常订单由客服团队跟进处理。",
            "活动结束后 5 个工作日内，各区域须提交活动效果分析，"
            "内容包含销售额、订单数、退货率与同比变化。",
        ),
        (
            ("活动商品价格核对", "各区域销售单元", "9 月 28 日前"),
            ("库存确认", "供应链部", "9 月 28 日前"),
            ("活动页面配置", "电商运营部", "9 月 30 日前"),
            ("活动效果分析", "各区域销售单元", "10 月 12 日前"),
        ),
    ),
    "other/q4-sales-kickoff-minutes": (
        "公司于 2025 年 10 月 9 日召开第四季度销售动员会，总经理出席并讲话，"
        "各区域销售负责人现场签署四季度目标责任书。",
        (
            "会议回顾了前三季度经营情况，肯定了整体达成情况，同时指出区域间分化明显，"
            "要求落后区域在四季度拿出具体改进方案。",
            "第四季度是全年收官阶段，会议要求各区域把握年末消费旺季，"
            "在守住价格底线的同时加大终端推广力度。",
            "会议强调，四季度不得以任何形式突破折扣审批下限。对为冲量而违规让利的，"
            "一经查实取消当年评优资格。",
            "会议决定设立四季度专项激励，对达成率排名前三的区域给予额外奖励，"
            "具体方案由人力资源部会同财务部制定。",
        ),
        (
            ("目标责任书签署", "各区域销售单元", "10 月 15 日前"),
            ("专项激励方案", "人力资源部", "10 月 25 日前"),
            ("旺季推广计划", "各区域销售单元", "10 月 20 日前"),
            ("折扣合规宣导", "渠道管理部", "10 月 31 日前"),
        ),
    ),
    "other/platform-upgrade-notice": (
        "为提升数据查询性能与稳定性，信息技术部将对数据平台进行版本升级，升级期间部分功能不可用。",
        (
            "升级窗口为 2025 年 11 月 16 日 22:00 至 11 月 17 日 06:00，"
            "期间经营数据查询、报表导出功能暂停服务，数据看板只读可用。",
            "升级内容包含查询引擎版本更新、索引重建与历史数据归档。"
            "升级完成后，历史查询的响应速度预计提升 40% 以上。",
            "升级期间产生的数据写入将在升级完成后自动补录，不影响数据完整性。"
            "补录预计在 11 月 17 日 12:00 前完成。",
            "如升级后发现问题，请联系信息技术部值班电话或通过工单系统提交。",
        ),
        (
            ("升级窗口公告", "信息技术部", "11 月 14 日前"),
            ("索引重建", "信息技术部", "11 月 17 日 06:00 前"),
            ("数据补录核验", "数据管理部", "11 月 17 日 12:00 前"),
            ("升级结果通报", "信息技术部", "11 月 18 日前"),
        ),
    ),
    "other/travel-reimbursement-supplement": (
        "根据公司财务管理要求，现对员工差旅报销制度作补充说明，自发布之日起执行。"
        "本说明与差旅报销制度正文不一致的，以本说明为准。",
        (
            "出差前须在系统提交出差申请并取得直属上级审批，未事前审批的差旅费用，"
            "报销比例按 80% 计算。因紧急情况无法事前审批的，须在返回后 3 个工作日内补办。",
            "住宿费按城市分级标准执行，超标部分由个人承担。"
            "同性别同级别员工同行出差的，原则上应合住标准间。",
            "市内交通费凭据报销，单次超过 100 元的须在报销单上注明事由。网约车费用须附行程单。",
            "差旅补贴按实际出差天数计算，出差当日往返的按半天计算。"
            "报销单据须在出差结束后 30 日内提交，逾期不予受理。",
        ),
        (
            ("出差申请流程调整", "人力资源部", "发布之日起"),
            ("住宿标准更新", "财务部", "发布之日起"),
            ("报销系统配置", "信息技术部", "发布后 5 个工作日内"),
            ("制度宣导", "各部门", "发布后 10 个工作日内"),
        ),
    ),
    "other/office-access-hours-notice": (
        "为配合办公楼物业管理调整，现对办公区门禁开放时间作如下调整。"
        "本通知仅涉及办公场所管理，与业务经营无关。",
        (
            "工作日门禁开放时间调整为 07:30 至 21:00，较原时间提前 30 分钟开放、"
            "延后 30 分钟关闭。周末及法定节假日开放时间为 09:00 至 18:00。",
            "非开放时段进入办公区，须提前一个工作日在系统提交加班申请，"
            "经部门负责人审批后由行政部开通临时权限。",
            "员工工牌遗失须在 24 小时内报行政部挂失，补办工本费 30 元。"
            "借用他人工牌进出的，双方各扣当月考勤分 5 分。",
            "访客须由接待人提前登记并全程陪同，访客证当日有效，离开时交回。",
        ),
        (
            ("门禁系统时间配置", "行政部", "通知发布后 3 个工作日内"),
            ("加班申请流程上线", "信息技术部", "通知发布后 7 个工作日内"),
            ("工牌挂失流程更新", "行政部", "通知发布后 5 个工作日内"),
            ("员工告知", "各部门", "通知发布后 10 个工作日内"),
        ),
    ),
}


def render_notice(spec: DocSpec, facts: Facts) -> list[Block]:
    content = NOTICE_CONTENT.get(spec.logical_key)
    if content is None:
        raise SystemExit(
            f"{spec.id}（{spec.logical_key}）没有对应的正文内容。"
            "不要退回通用骨架——共用骨架产出的是结构完全相同、只换标题的通知，"
            "而这类文档的存在意义恰恰是「测检索区分度」，同质化会让这条失效。"
        )
    background, body, duties = content
    return [
        Heading(1, spec.title),
        Table(
            "文件信息",
            ("项目", "内容"),
            (
                ("文件编号", spec.id),
                ("发布部门", spec.department or "—"),
                ("发布日期", spec.published_at or "—"),
                ("密级", spec.classification),
            ),
            repeat_header=False,
        ),
        Heading(1, "一、背景"),
        Para(background),
        Heading(1, "二、具体安排"),
        Para("\n".join(f"（{_cn(i + 1)}）{p}" for i, p in enumerate(body))),
        Heading(1, "三、工作分工"),
        Table("工作分工", ("事项", "责任部门", "完成时限"), duties),
        Heading(1, "四、其他事项"),
        Para(
            "本通知自发布之日起执行。此前发布的有关规定与本通知不一致的，以本通知为准。"
            "未尽事宜，由归口部门负责解释。"
        ),
    ]


# ----------------------------------------------------------------- 分派


def render(spec: DocSpec, facts: Facts) -> list[Block]:
    if spec.type == "PRODUCT":
        return render_product(spec, facts)
    if spec.type == "REPORT":
        return render_report(spec, facts)
    if spec.type == "METRIC":
        fn = _METRIC_BY_LOGICAL_KEY.get(spec.logical_key)
        if fn:
            return fn(spec, facts)
        return render_metric(spec, facts)
    if spec.type == "POLICY":
        fn = POLICY_RENDERERS.get(spec.logical_key)
        if fn:
            return fn(spec, facts)
        return render_generic_policy(spec, facts)
    return render_other(spec, facts)


# ----------------------------------------------------------------- 缺陷注入


def inject_handwritten(spec: DocSpec, blocks: list[Block]) -> list[Block]:
    """把手工编写的 Prompt Injection 正文注入目标文档。

    **不在生成器模板里写注入文本**：模板能被批量复制，而手工样本的差异性是它存在的意义
    （16.11.2 明写这两类必须人工设计）。文本放在 `configs/corpus_handwritten/`，
    在 git 历史里可见、可 review。

    落点由清单的 `injection:` 字段**显式指定**，不按 `type` 猜。
    早先按 `injection_{type}.md` 分派时出过一个接不上的错：清单标记的三篇是
    PRODUCT / METRIC / OTHER，而手上的 POLICY 样本谁也匹配不上，会静默地永不生效。
    静默失效的注入样本比没有样本更糟——它让「已注入 3 份」这句话失去依据。
    现在注入文件缺失会**直接报错**，不再 return 原 blocks。
    """
    name = spec.raw.get("injection")
    if not name:
        raise SystemExit(f"{spec.id} 标记了 PROMPT_INJECTION 但清单里没有 injection: 字段")
    path = HANDWRITTEN_DIR / str(name)
    if not path.exists():
        raise SystemExit(f"{spec.id} 指定的注入文件不存在：{path}")
    raw = path.read_text(encoding="utf-8")
    # 文件头部的 HTML 注释是给维护者看的说明（为什么这么写、期望行为是什么），不进语料
    body = "\n".join(
        line
        for line in raw.splitlines()
        if not (line.strip().startswith("<!--") or line.strip().startswith("-->"))
    ).strip()
    if not body:
        raise SystemExit(f"{path} 去掉说明注释后没有正文，注入会是空操作")
    return [*blocks, Heading(1, "附：系统对接说明"), Para(body)]


# ----------------------------------------------------------------- 渲染：Markdown / TXT


def write_md(blocks: list[Block], path: Path, spec: DocSpec) -> None:
    out: list[str] = []
    for b in blocks:
        if isinstance(b, Heading):
            out.append(f"{'#' * (b.level + 1)} {b.text}")
        elif isinstance(b, Para):
            out.append(b.text)
        elif isinstance(b, Table):
            if b.caption:
                out.append(f"**{b.caption}**")
            out.append("| " + " | ".join(b.header) + " |")
            out.append("|" + "---|" * len(b.header))
            for row in b.rows:
                out.append("| " + " | ".join(row) + " |")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def write_txt(blocks: list[Block], path: Path, spec: DocSpec) -> None:
    out: list[str] = []
    for b in blocks:
        if isinstance(b, Heading):
            out.append(b.text)
            out.append("-" * min(60, max(8, len(b.text) * 2)))
        elif isinstance(b, Para):
            out.append(b.text)
        elif isinstance(b, Table):
            if b.caption:
                out.append(f"【{b.caption}】")
            widths = [
                max(len(str(r[i])) for r in (*b.rows, b.header)) for i in range(len(b.header))
            ]
            out.append("  ".join(h.ljust(w) for h, w in zip(b.header, widths, strict=True)))
            out.append("  ".join("-" * w for w in widths))
            for row in b.rows:
                out.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)))
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


# ----------------------------------------------------------------- 渲染：PDF


def write_pdf(blocks: list[Block], path: Path, spec: DocSpec) -> None:
    """用 reportlab 出 PDF。

    三个细节是**为缺陷注入服务**的，不是排版偏好：
      - 每页页眉页脚（16.11.2 第 8 项）：测清洗逻辑能否把它们挡在 chunk 之外；
      - `repeatRows=1`（第 7 项）：跨页表格的表头必须在后续页重复，否则解析出来
        的第二页表格没有列名，表头还原就无从谈起；
      - `force_split` 时在表前分页（第 7 项）：否则一张 8 行的表可能恰好塞进当前页，
        缺陷注入了但没跨页，等于没注入。
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        PageTemplate,
        Paragraph,
        Spacer,
        TableStyle,
    )
    from reportlab.platypus import (
        PageBreak as RLPageBreak,
    )
    from reportlab.platypus import (
        Table as RLTable,
    )

    # STSong-Light 是 reportlab 内置的 Adobe CJK 字体，无需外部字体文件。
    # 演示语料要用它出 44 份中文 PDF，若依赖系统字体，换台机器就生成不出来。
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    font = "STSong-Light"

    body = ParagraphStyle("body", fontName=font, fontSize=10.5, leading=17, firstLineIndent=21)
    h1 = ParagraphStyle("h1", fontName=font, fontSize=15, leading=22, spaceBefore=12, spaceAfter=8)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=12, leading=19, spaceBefore=8, spaceAfter=6)
    cap = ParagraphStyle(
        "cap", fontName=font, fontSize=9.5, leading=14, textColor=colors.HexColor("#444444")
    )
    title_style = ParagraphStyle(
        "title", fontName=font, fontSize=19, leading=27, alignment=TA_CENTER, spaceAfter=12
    )

    def on_page(canv: Any, doc: Any) -> None:
        canv.saveState()
        canv.setFont(font, 8)
        canv.setFillColor(colors.HexColor("#666666"))
        # 页眉：公司名 + 标题 + 密级
        canv.drawString(20 * mm, A4[1] - 14 * mm, f"{COMPANY}")
        canv.drawRightString(
            A4[0] - 20 * mm, A4[1] - 14 * mm, f"{spec.title}　[{spec.classification}]"
        )
        canv.setStrokeColor(colors.HexColor("#cccccc"))
        canv.line(20 * mm, A4[1] - 16 * mm, A4[0] - 20 * mm, A4[1] - 16 * mm)
        # 页脚：文件编号 + 页码
        canv.line(20 * mm, 16 * mm, A4[0] - 20 * mm, 16 * mm)
        canv.drawString(20 * mm, 12 * mm, f"文件编号：{spec.id}　版本：{spec.version}")
        canv.drawRightString(A4[0] - 20 * mm, 12 * mm, f"第 {doc.page} 页")
        canv.restoreState()

    doc = BaseDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        title=spec.title,
        author=COMPANY,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="main",
                frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")],
                onPage=on_page,
            )
        ]
    )

    story: list[Any] = []
    for b in blocks:
        if isinstance(b, Heading):
            story.append(
                Paragraph(
                    b.text,
                    title_style if b.level == 1 and not story else (h1 if b.level == 1 else h2),
                )
            )
        elif isinstance(b, Para):
            for line in b.text.split("\n"):
                story.append(Paragraph(line.replace("**", ""), body))
        elif isinstance(b, PageBreak):
            story.append(RLPageBreak())
        elif isinstance(b, Table):
            if b.force_split:
                story.append(RLPageBreak())
            if b.caption:
                story.append(Paragraph(f"表：{b.caption}", cap))
            data = [[Paragraph(str(c), cap) for c in b.header]] + [
                [Paragraph(str(c), cap) for c in row] for row in b.rows
            ]
            tbl = RLTable(data, repeatRows=1 if b.repeat_header else 0, hAlign="LEFT")
            tbl.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 6))
    doc.build(story)


def write_scanned_pdf(blocks: list[Block], path: Path, spec: DocSpec) -> None:
    """出图片型扫描件 PDF：**没有文本层**，只有页面位图。

    做法是先用 reportlab 正常排一版，再逐页栅格化成图片，最后把图片重新装订成 PDF。
    这样产出的文件与真实扫描件的可解析性一致——`pdfplumber` 抽不出文字。
    若直接用 reportlab 画图而不清掉文字层，抽取仍能成功，这条缺陷就形同虚设。

    期望的下游行为是 `knowledge_document.status = FAILED` 且
    `error_summary` 写明「扫描件不支持 OCR」，**不是**硬凑一段乱码。
    切片内不做 OCR（冲刺方案 §10 后置），因此这条测的是「系统知道自己读不了」。
    """
    import io

    import pypdfium2 as pdfium
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rl_canvas

    tmp = path.with_suffix(".tmp.pdf")
    write_pdf(blocks, tmp, spec)

    src = pdfium.PdfDocument(str(tmp))
    out = rl_canvas.Canvas(str(path), pagesize=A4)
    for page in src:
        bitmap = page.render(scale=2.0)
        image = bitmap.to_pil()
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=72)
        buf.seek(0)
        out.drawImage(ImageReader(buf), 0, 0, width=A4[0], height=A4[1])
        out.showPage()
    out.save()
    tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------- 渲染：DOCX


def write_docx(blocks: list[Block], path: Path, spec: DocSpec) -> None:
    """python-docx 出 DOCX。同样带页眉页脚（16.11.2 第 8 项）。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(10.5)
    # 中文字体要单独设 eastAsia，否则 Word 里回退成默认字体，中文显示为方框
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    header = doc.sections[0].header
    hp = header.paragraphs[0]
    hp.text = f"{COMPANY}　　{spec.title}　[{spec.classification}]"
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer = doc.sections[0].footer
    footer.paragraphs[0].text = f"文件编号：{spec.id}　版本：{spec.version}"

    for b in blocks:
        if isinstance(b, Heading):
            if b.level == 1:
                doc.add_heading(b.text, level=1)
            else:
                doc.add_heading(b.text, level=2)
        elif isinstance(b, Para):
            for line in b.text.split("\n"):
                doc.add_paragraph(line.replace("**", ""))
        elif isinstance(b, PageBreak):
            # `add_page_break` 在 python-docx 1.2 里没有类型标注（它是动态属性代理）。
            # 这一处忽略是必要的：python-docx **自带 py.typed**，所以
            # `ignore_missing_imports` 对它不生效——那个开关只对完全没有类型信息的
            # 模块起作用，而对「有类型信息但标注不全」的模块，mypy 会照常报错，
            # 这是对的：把整个 docx 塞进忽略名单，会连它标注好的部分一起失去检查。
            doc.add_page_break()  # type: ignore[no-untyped-call]
        elif isinstance(b, Table):
            if b.caption:
                doc.add_paragraph(f"表：{b.caption}")
            t = doc.add_table(rows=1, cols=len(b.header))
            t.style = "Table Grid"
            for i, h in enumerate(b.header):
                t.rows[0].cells[i].text = str(h)
            for row in b.rows:
                cells = t.add_row().cells
                for i, c in enumerate(row):
                    cells[i].text = str(c)
    doc.save(str(path))


# ----------------------------------------------------------------- 清单加载


def load_manifest(path: Path = MANIFEST_PATH) -> tuple[dict[str, Any], list[DocSpec]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs: list[DocSpec] = []
    for d in raw["documents"]:
        # PD-* 的紧凑条目：产品名/标价/上市日从业务库读，手抄进清单会与库漂移
        if d["id"].startswith("PD-"):
            d.setdefault("type", "PRODUCT")
            d.setdefault("version", "v1.0")
            d.setdefault("department", "产品管理部")
            d.setdefault("source_kind", "INTERNAL")
            d.setdefault("logical_key", f"product/{d['product_code'].lower()}")
            d.setdefault("title", "")  # 由 render_product 从库里取真实产品名
        specs.append(
            DocSpec(
                id=d["id"],
                logical_key=d["logical_key"],
                title=d.get("title") or d["id"],
                type=d["type"],
                format=d["format"],
                version=d.get("version", "v1.0"),
                department=d.get("department"),
                source_kind=d.get("source_kind", "INTERNAL"),
                effective_from=_dstr(d.get("effective_from")),
                effective_to=_dstr(d.get("effective_to")),
                published_at=_dstr(d.get("published_at")),
                defects=tuple(d.get("defects") or ()),
                raw=d,
            )
        )
    return raw, specs


def _dstr(v: Any) -> str | None:
    return None if v is None else str(v)


# ----------------------------------------------------------------- 主流程


def extract_text(path: Path, fmt: str) -> str:
    """从产物里回读正文。回读而不是相信自己刚写的内容——
    这正是下面 `check_injection` 存在的意义。"""
    if fmt == "pdf":
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if fmt == "docx":
        import docx

        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" ".join(c.text for c in row.cells))
        return "\n".join(parts)
    return path.read_text(encoding="utf-8")


def check_injection(path: Path, spec: DocSpec, blocks: list[Block]) -> list[str]:
    """回读产物，验证缺陷**真的注入了**。返回问题清单（空表示全部通过）。

    **为什么需要这一步**：清单里写 `defects: [CROSS_PAGE_TABLE]` 只是一句声明，
    它不保证渲染出来真的跨页。第一版就是这样：三份「跨页表格」的标记齐全，
    实际每一张都稳稳地待在一页里——`force_split` 只是把表挪到新页开头，
    并不会让表变高。若门禁只断言「这篇文档标记了该缺陷」，三份假的会全部放行。

    这个教训比它修掉的那个 bug 更值得留下：**标注不是证据，产物才是**。
    因此这里回读真实文件，不信任任何中间状态。
    """
    problems: list[str] = []
    fmt = spec.format

    if "SCANNED" in spec.defects:
        text = extract_text(path, fmt)
        if len(text.strip()) >= 20:
            problems.append(
                f"SCANNED 未生效：图片型 PDF 仍能抽出 {len(text.strip())} 个字符，"
                "说明文字层没被清掉，OCR 路径这条缺陷形同虚设"
            )

    if "CROSS_PAGE_TABLE" in spec.defects:
        if fmt != "pdf":
            problems.append(f"CROSS_PAGE_TABLE 只对 PDF 有意义，当前格式是 {fmt}")
        else:
            import pdfplumber

            marked = [b for b in blocks if isinstance(b, Table) and b.force_split]
            if not marked:
                problems.append("标记了 CROSS_PAGE_TABLE 但没有任何 force_split 的表格")
            else:
                header_line = " ".join(marked[0].header)
                with pdfplumber.open(str(path)) as pdf:
                    pages = [(p.extract_text() or "").splitlines() for p in pdf.pages]
                hits = [
                    i + 1
                    for i, lines in enumerate(pages)
                    if any(ln.strip() == header_line for ln in lines)
                ]
                if len(hits) < 2:
                    problems.append(
                        f"CROSS_PAGE_TABLE 未生效：表头「{header_line[:24]}…」"
                        f"只在第 {hits} 页出现，说明表格没跨页"
                        "（实测本版式下需 ≥35 行才撑破一页）"
                    )

    if "PROMPT_INJECTION" in spec.defects:
        text = extract_text(path, fmt)
        # 从注入文件的正文里取一句特征串来核对，而不是断言"文件存在"——
        # 文件存在只证明我写了它，不证明它进了语料
        raw = (HANDWRITTEN_DIR / str(spec.raw["injection"])).read_text(encoding="utf-8")
        body_lines = [
            ln.strip()
            for ln in raw.splitlines()
            if ln.strip() and not ln.strip().startswith(("<!--", "-->"))
        ]
        marker = max(body_lines, key=len)[:28]
        if marker not in text:
            problems.append(f"PROMPT_INJECTION 未生效：产物里找不到注入特征串「{marker}…」")

    return problems


def resolve_titles(specs: list[DocSpec], facts: Facts) -> None:
    """产品资料的标题取自 `dim_product`，不在清单里手抄。

    手抄的代价不是「多打几个字」，而是**标题与库会漂移**：改了产品名，
    清单里那份不会跟着变，于是语料说「智能音箱」而库说别的，
    检索命中的文档和 SQL 查到的产品对不上——而本项目的全部说服力
    就建立在「文档与数据库指的是同一件事」上。

    清单里 `title` 留空即触发这里解析（见 `load_manifest` 的 `setdefault("title", "")`）。
    """
    for spec in specs:
        if spec.type == "PRODUCT" and (not spec.title or spec.title == spec.id):
            product = facts.products.get(spec.raw["product_code"])
            if product is None:
                raise SystemExit(
                    f"{spec.id} 引用了不存在的 product_code：{spec.raw['product_code']}"
                )
            spec.title = f"{product['name']}产品资料"


def generate(
    specs: list[DocSpec], facts: Facts, out_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    resolve_titles(specs, facts)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    injection_failures: list[str] = []
    for spec in specs:
        blocks = render(spec, facts)
        if "PROMPT_INJECTION" in spec.defects:
            blocks = inject_handwritten(spec, blocks)
        target = out_dir / f"{spec.id}.{spec.format}"
        writer = {"pdf": write_pdf, "docx": write_docx, "md": write_md, "txt": write_txt}[
            spec.format
        ]
        if "SCANNED" in spec.defects:
            write_scanned_pdf(blocks, target, spec)
        else:
            writer(blocks, target, spec)

        text_len = sum(len(b.text) for b in blocks if isinstance(b, Para)) + sum(
            len(b.text) for b in blocks if isinstance(b, Heading)
        )

        problems = check_injection(target, spec, blocks)
        if problems:
            injection_failures.extend(f"{spec.id}: {p}" for p in problems)
            for p in problems:
                print(f"      [!!] {p}")

        records.append(
            {
                "id": spec.id,
                "path": str(target.relative_to(ROOT)),
                "title": spec.title,
                "type": spec.type,
                "format": spec.format,
                "version": spec.version,
                "logical_key": spec.logical_key,
                "department": spec.department,
                "source_kind": spec.source_kind,
                "effective_from": spec.effective_from,
                "effective_to": spec.effective_to,
                "published_at": spec.published_at,
                "classification": spec.classification,
                "defects": list(spec.defects),
                "char_count": text_len,
                "is_scanned": "SCANNED" in spec.defects,
                "injection_verified": not problems,
            }
        )
        mark = "  " if not problems else "!!"
        print(f"  {mark}[{spec.id:14}] {spec.format:4} {text_len:6} 字  {spec.title}")
    return records, injection_failures


async def main_async(args: argparse.Namespace) -> int:
    manifest, specs = load_manifest()

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        specs = [s for s in specs if s.id in wanted]
    elif args.subset:
        # 采样要**按类型分层**，否则前 10 篇全是制度，链路一跑就没有报告与产品，
        # 「PDF 解析 + 表格序列化」这些要靠报告才测得出来。
        #
        # 分层之后还要**补齐余数**：第一版只取 `subset // 类型数` 篇，
        # 于是 `--subset 12` 实际只出 10 篇（12 // 5 = 2，2 × 5 = 10），
        # 少的那两篇没有任何提示。按数量从多到少轮转补齐，凑满用户要的篇数。
        by_type: dict[str, list[DocSpec]] = defaultdict(list)
        for s in specs:
            by_type[s.type].append(s)
        per = max(1, args.subset // max(1, len(by_type)))
        picked: list[DocSpec] = []
        for group in by_type.values():
            picked.extend(group[:per])
        if len(picked) < args.subset:
            chosen = {s.id for s in picked}
            rest = [s for s in specs if s.id not in chosen]
            picked.extend(rest[: args.subset - len(picked)])
        specs = sorted(picked, key=lambda s: s.id)[: args.subset]

    if args.list:
        print(f"清单 {manifest['version']}：{len(specs)} 篇")
        for s in specs:
            print(f"  {s.id:14} {s.type:8} {s.format:5} {','.join(s.defects) or '-'}")
        return 0

    print("从业务库读取真值…")
    facts = await load_facts()
    print(
        f"  {len(facts.regions)} 区域 / {len(facts.channels)} 渠道 / "
        f"{len(facts.product_lines)} 产品线 / {len(facts.products)} 产品 / "
        f"{len(facts.daily_facts)} 日粒度分组"
    )

    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    print(f"生成到 {out_dir} …")
    records, injection_failures = generate(specs, facts, out_dir)

    # 生成报告：`verify-corpus` 与入库脚本都读它，避免各自重新解析文件系统
    report = {
        "manifest_version": manifest["version"],
        "count": len(records),
        "documents": records,
        "absent_policies": manifest["absent_policies"],
    }
    # 写**真 JSON**（早先用 yaml.safe_dump 写了个 .json 后缀的文件，
    # 结果调用方 `json.load` 直接抛 JSONDecodeError——机器读的文件不该靠肉眼看格式）
    report_path = out_dir / "corpus_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(r["char_count"] for r in records)
    print(f"\n完成：{len(records)} 篇，正文合计 {total:,} 字")
    print(f"生成报告：{report_path.relative_to(ROOT)}")

    if injection_failures:
        # 不返回非零之外的"软警告"：语料一旦被下游冻结，假的缺陷注入会被
        # 一路带进 Recall@8 与 verify-corpus 的结论里，而那时候没人会回头看这个脚本的输出。
        print(f"\n**{len(injection_failures)} 项缺陷注入未生效**——语料不可用：")
        for f in injection_failures:
            print(f"  - {f}")
        return 1
    checked = sum(1 for r in records if r["defects"])
    print(f"缺陷注入自检通过：{checked} 篇带缺陷的文档已回读产物验证")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="演示语料生成器（详设 16.11）")
    p.add_argument("--out", help=f"输出目录，默认 {DEFAULT_OUT.relative_to(ROOT)}")
    p.add_argument("--subset", type=int, help="只生成 N 篇（按类型分层采样），用于先跑通链路")
    p.add_argument("--only", help="只生成指定 ID，逗号分隔，如 SP-001,MD-008")
    p.add_argument("--list", action="store_true", help="只打印清单摘要，不生成")
    p.add_argument("--clean", action="store_true", help="生成前清空输出目录")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
