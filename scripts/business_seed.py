"""业务演示库建表与种子数据（开发流程 6.4 施工项 5/6，详细设计 16.10）。

**这个脚本是 Phase 2 门禁的一部分**：它必须让 16.10 的四条约束同时成立，
并用断言直接查库证明。不通过就不得进入 Phase 4——因为 SQL Tool 的
全部评测问题都建立在这一套数据上，数据不成立，后面测的是空气。

## 反向构造，不是随机生成

「先确定结论，再生成明细」是硬要求。随机生成的数据不会恰好出现
「某区域某产品线在某渠道异常下滑」这种结构，而整个下钻案例
（区域对比 → 渠道下钻 → 产品线下钻 → 库存佐证）全靠它。

四条约束（16.10）：

```
约束 1  某区域 Q3 净销售额同比约 -12%，且该区域贡献总降幅的多数
约束 2  该区域内某产品线在某渠道的降幅显著高于同渠道其他产品线
约束 3  该产品线在同期连续 6 周可用库存低于安全线
约束 4  同期其他区域 / 其他产品线的数据保持正常波动，不出现同类异常
```

**约束 4 是必需的**：如果全库都异常，第一步的区域对比就无法定位到具体区域，
下钻案例不成立。它由断言 4 显式钉住，不是「顺便这样」。

## 为什么脚本在 scripts/ 而不是 app/

业务库是**模拟的企业既有系统**，不是本应用的表：应用侧连的是只读账号，
没有权限改结构。建表与灌数属于「DBA 侧的供给」，与 `init-business-db.sql`
同属一类。放进 `app/` 会让读者以为运行时也会写这个库。

## 两条连接、各司其职

- **写**：`DATABASE_URL_BUSINESS_RW`，只在本脚本里用（建表 + 灌数）
- **读**：`DATABASE_URL_BUSINESS_RO`，**断言全部走它**

断言特意用只读账号跑，是一箭双雕：既验证了数据成立，
又顺带证明了「只读账号能读到全部演示数据」——这正是 SQL Tool 的取数路径。

用法：
    uv run python scripts/business_seed.py            # 全量 50 万行
    uv run python scripts/business_seed.py --rows 50000   # 快速试跑
    uv run python scripts/business_seed.py --verify-only  # 只跑断言
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from app.core.config import Settings, get_settings
from app.core.ids import IdPrefix, deterministic_id

# --------------------------------------------------------------- 演示语料的骨架
#: 五个区域。华东是异常区域，其余四个用于「区域对比」的对照组。
REGIONS: tuple[tuple[str, str, int, bool], ...] = (
    ("华东", "R01", 4, True),
    ("华南", "R02", 3, False),
    ("华北", "R03", 5, False),
    ("华中", "R04", 4, False),
    ("西南", "R05", 3, False),
)

#: 四个渠道。经销是异常渠道（约束 2 的落点）。
CHANNELS: tuple[tuple[str, str, str], ...] = (
    ("直营", "C01", "DIRECT"),
    ("经销", "C02", "PARTNER"),
    ("电商", "C03", "ONLINE"),
    ("KA", "C04", "DIRECT"),
)

#: 四条产品线 × 每条 5 个产品 = 20 个产品。智能家居是异常产品线。
PRODUCT_LINES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("智能家居", "L01", ("智能门锁", "扫地机器人", "智能音箱", "智能灯泡", "智能摄像头")),
    ("厨房电器", "L02", ("电饭煲", "微波炉", "破壁机", "洗碗机", "空气炸锅")),
    ("个护健康", "L03", ("电动牙刷", "剃须刀", "吹风机", "按摩仪", "血压计")),
    ("影音娱乐", "L04", ("智能电视", "回音壁", "投影仪", "蓝牙耳机", "游戏手柄")),
)

#: 异常定位。改这三个常量就等于换一套下钻案例——断言会跟着变，不需要改断言代码。
ANOMALY_REGION = "华东"
ANOMALY_CHANNEL = "经销"
ANOMALY_LINE = "智能家居"

#: 约束 1：华东 2025 Q3 同比 -12%
REGION_YOY = Decimal("0.88")
#: 约束 2：该产品线在该渠道同比 -45%（显著高于同渠道其他产品线）
ANOMALY_YOY = Decimal("0.55")
#: 各区域的正常增速。有正有负才叫「波动」而不是「齐涨」（约束 4）。
#:
#: **华东也在表里**：异常是**特定季度**的现象，不是「华东全年塌了」。
#: 少了这一条，华东在非 Q3 月份的因子就无处可取——而华东 Q3 的下滑
#: 恰恰要放在「其余时间正常」的背景里才显得是异常，而不是趋势。
#: 全部落在 ±5% 内：「正常波动」的判定标准（断言 4a）就是 |同比| < 5%，
#: 增速本身必须先满足它，否则断言从一开始就不可能过——
#: 这不是「测试太严」，是构造参数与验收标准脱节。
REGION_GROWTH: dict[str, Decimal] = {
    "华东": Decimal("1.03"),
    "华南": Decimal("1.04"),
    "华北": Decimal("1.02"),
    "华中": Decimal("0.98"),
    "西南": Decimal("1.03"),
}

#: 数据跨度：2024-01 至 2025-12，共 24 个月。同比需要上一年同期，因此不能只有一年。
START = date(2024, 1, 1)
END = date(2025, 12, 31)
TARGET_YEAR = 2025
TARGET_QUARTER_MONTHS = (7, 8, 9)
BASE_YEAR = 2024

#: 每行金额的随机波动。**对称**，因此在组合月聚合后会相互抵消——
#: 这正是「明细有波动、聚合结论精确」得以同时成立的原因。
ROW_NOISE = 0.04

#: 库存低于安全线的目标周数（约束 3）。
ANOMALY_LOW_WEEKS = 8
SAFETY_STOCK = 1200

#: 分块写入的行数。单条 INSERT 塞太多行会撞 max_allowed_packet，
#: 太小又会让 50 万行变成 50 万次往返。
BATCH_ROWS = 1000


# ------------------------------------------------------------------ 基础工具
def _rng(*parts: str) -> random.Random:
    """由种子确定性派生的随机源。

    用 `hashlib` 而不是 `hash()`：后者对 str 的结果在同一进程内才稳定，
    换个进程（或开了 PYTHONHASHSEED）就变，数据无法复现——而「换台机器
    重灌一次，结论还成立吗」是这个脚本最需要保证的事。
    """
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _money(value: Decimal | float) -> Decimal:
    return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _month_iter(start: date, end: date) -> list[date]:
    months: list[date] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        months.append(cursor)
        cursor = date(cursor.year + (cursor.month // 12), cursor.month % 12 + 1, 1)
    return months


def _quarter(month: date) -> int:
    return (month.month - 1) // 3 + 1


# ---------------------------------------------------------------- 维度构建
@dataclass(frozen=True, slots=True)
class Dimensions:
    regions: tuple[dict[str, Any], ...]
    channels: tuple[dict[str, Any], ...]
    product_lines: tuple[dict[str, Any], ...]
    products: tuple[dict[str, Any], ...]
    customers: tuple[dict[str, Any], ...]

    def region_id(self, name: str) -> str:
        return _by_name(self.regions, name)

    def channel_id(self, name: str) -> str:
        return _by_name(self.channels, name)

    def product_line_id(self, name: str) -> str:
        return _by_name(self.product_lines, name)


def _by_name(rows: tuple[dict[str, Any], ...], name: str) -> str:
    """按名称取 ID。三张维度表结构相近，用「哪个键在就用哪个」统一处理。"""
    for row in rows:
        for name_key, id_key in (
            ("region_name", "region_id"),
            ("channel_name", "channel_id"),
            ("product_line_name", "product_line_id"),
        ):
            if row.get(name_key) == name:
                return str(row[id_key])
    raise KeyError(name)


def build_dimensions(now: datetime) -> Dimensions:
    regions = tuple(
        {
            "region_id": deterministic_id(IdPrefix.REGION, name),
            "region_code": code,
            "region_name": name,
            "province_count": provinces,
            "is_anomaly": int(is_anomaly),
            "created_at": now,
        }
        for name, code, provinces, is_anomaly in REGIONS
    )
    channels = tuple(
        {
            "channel_id": deterministic_id(IdPrefix.CHANNEL, name),
            "channel_code": code,
            "channel_name": name,
            "channel_type": ctype,
            "created_at": now,
        }
        for name, code, ctype in CHANNELS
    )
    product_lines = tuple(
        {
            "product_line_id": deterministic_id(IdPrefix.PRODUCT_LINE, name),
            "product_line_code": code,
            "product_line_name": name,
            "category": "消费电子",
            "created_at": now,
        }
        for name, code, _ in PRODUCT_LINES
    )
    products: list[dict[str, Any]] = []
    for line_name, _, names in PRODUCT_LINES:
        line_id = deterministic_id(IdPrefix.PRODUCT_LINE, line_name)
        for name in names:
            rng = _rng("price", name)
            # 标价 199–2999：覆盖到「一件商品就能撑起一行金额」的量级，
            # 使 quantity × unit_price 的构成与真实订单接近。
            price = _money(Decimal(rng.randrange(199, 2999)))
            products.append(
                {
                    "product_id": deterministic_id(IdPrefix.PRODUCT, name),
                    "product_code": f"P{products.__len__() + 1:04d}",
                    "product_name": name,
                    "product_line_id": line_id,
                    "list_price": price,
                    "launch_date": date(2022, 1, 1) + timedelta(days=rng.randrange(0, 700)),
                    "status": "ACTIVE",
                    "created_at": now,
                }
            )

    # 每个 (区域, 渠道) 组合下放几个客户。客户不是分析主线，但 JOIN 链路上要用。
    customers: list[dict[str, Any]] = []
    for region in regions:
        for channel in channels:
            for index in range(6):
                seed = f"{region['region_name']}-{channel['channel_name']}-{index}"
                display = f"{region['region_name']}{channel['channel_name']}客户{index + 1}"
                customers.append(
                    {
                        "customer_id": deterministic_id(IdPrefix.CUSTOMER, seed),
                        "customer_code": f"C{len(customers) + 1:05d}",
                        "customer_name": display,
                        "region_id": region["region_id"],
                        "channel_id": channel["channel_id"],
                        "customer_level": ("A", "B", "C")[index % 3],
                        "created_at": now,
                    }
                )

    return Dimensions(
        regions=regions,
        channels=channels,
        product_lines=product_lines,
        products=tuple(products),
        customers=tuple(customers),
    )


# ---------------------------------------------------------------- 金额因子
def _discount_rate(channel: str) -> Decimal:
    """渠道折扣率。各渠道不同，是「渠道对比」能看出差别的基础。"""
    return {
        "直营": Decimal("0.05"),
        "经销": Decimal("0.18"),
        "电商": Decimal("0.12"),
        "KA": Decimal("0.22"),
    }[channel]


def _seasonal(month: date) -> float:
    """季节性：Q4 走强、Q1 走弱。让月度曲线不是一条直线。"""
    return {
        1: 0.82,
        2: 0.78,
        3: 0.95,
        4: 0.96,
        5: 1.0,
        6: 1.08,
        7: 0.98,
        8: 1.02,
        9: 1.12,
        10: 1.18,
        11: 1.24,
        12: 1.06,
    }[month.month]


def _base_row(
    rng: random.Random,
    *,
    when: date,
    region: str,
    channel: str,
    product: dict[str, Any],
    customer_id: str,
    now: datetime,
) -> dict[str, Any]:
    """生成一行订单明细（未做年度因子缩放）。"""
    quantity = rng.randint(1, 12)
    unit_price = Decimal(product["list_price"]) * Decimal("0.92")  # 成交价略低于标价
    gross = _money(unit_price * quantity)
    discount = _money(gross * _discount_rate(channel))
    # 退货率低且随机：它是「报告用含税、DB 用净额」这类口径差异的来源之一。
    return_amount = _money(gross * Decimal(str(rng.random() * 0.03)))
    return {
        "order_id": deterministic_id(
            IdPrefix.ORDER,
            f"{when.isoformat()}-{region}-{channel}-{product['product_id']}-{rng.random()}",
        ),
        "order_date": when,
        "region_id": None,  # 由调用方回填（需要 Dimensions）
        "channel_id": None,
        "product_id": product["product_id"],
        "customer_id": customer_id,
        "quantity": quantity,
        "gross_amount": gross,
        "discount_amount": discount,
        "return_amount": return_amount,
        "net_amount": _money(gross - discount - return_amount),
        "currency": "CNY",
        "created_at": now,
    }


# ---------------------------------------------------------------- 反向构造
@dataclass(frozen=True, slots=True)
class Construction:
    """反向构造算出来的因子，以及它依据的基线。"""

    #: 华东 Q3 中「非异常组合」的同比因子。由约束 1 与约束 2 联立解出。
    other_factor: Decimal
    #: 异常产品线在该渠道占华东 Q3 的份额
    anomaly_share: Decimal
    #: 2024 Q3 华东净销售额（基线）
    base_huadong_q3: Decimal


def solve_factors(rows_2024: list[dict[str, Any]], dims: Dimensions) -> Construction:
    """先算 2024 年的实际落点，再解出 2025 年的因子。

    **不能预设因子**：约束 1 要求华东整体 -12%，而约束 2 要求其中
    异常组合 -45%。两者是「总体」与「其中一部分」的关系，只有先知道
    异常组合占多大比重（w），才能解出其余部分该是多少：

        w × 0.55 + (1 - w) × f_other = 0.88
        =>  f_other = (0.88 - 0.55 w) / (1 - w)

    若直接把 f_other 也设成 0.88，华东整体会掉到 -14.6% 左右——
    偏差不大，但足以让「同比约 -12%」的断言变成一句含糊话。
    """
    huadong_id = dims.region_id(ANOMALY_REGION)
    anomaly_line_id = dims.product_line_id(ANOMALY_LINE)
    anomaly_channel_id = dims.channel_id(ANOMALY_CHANNEL)
    product_line_of = {p["product_id"]: p["product_line_id"] for p in dims.products}

    q3_total = Decimal(0)
    q3_anomaly = Decimal(0)
    for row in rows_2024:
        when: date = row["order_date"]
        if row["region_id"] != huadong_id or _quarter(when) != 3:
            continue
        q3_total += Decimal(row["net_amount"])
        if (
            row["channel_id"] == anomaly_channel_id
            and product_line_of[row["product_id"]] == anomaly_line_id
        ):
            q3_anomaly += Decimal(row["net_amount"])

    share = q3_anomaly / q3_total
    other = (REGION_YOY - ANOMALY_YOY * share) / (Decimal(1) - share)
    return Construction(other_factor=other, anomaly_share=share, base_huadong_q3=q3_total)


def year_factor(
    construction: Construction,
    *,
    when: date,
    region: str,
    channel: str,
    line: str,
) -> Decimal:
    """2025 年相对 2024 年同期的缩放因子。"""
    if when.year == BASE_YEAR:
        return Decimal(1)

    in_target_window = (
        region == ANOMALY_REGION and _quarter(when) in (3,) and when.year == TARGET_YEAR
    )
    if in_target_window:
        if channel == ANOMALY_CHANNEL and line == ANOMALY_LINE:
            return ANOMALY_YOY
        return construction.other_factor
    return REGION_GROWTH[region]


def generate_sales(
    dims: Dimensions,
    construction: Construction,
    months: list[date],
    rows_per_combo: int,
    now: datetime,
) -> list[dict[str, Any]]:
    """按 (区域, 渠道, 产品, 月) 展开订单明细。"""
    line_name_of = {ln["product_line_id"]: ln["product_line_name"] for ln in dims.product_lines}
    # 每个 (区域, 渠道) 下的客户，供订单随机归属
    customers_of: dict[tuple[str, str], list[str]] = defaultdict(list)
    for customer in dims.customers:
        customers_of[(customer["region_id"], customer["channel_id"])].append(
            customer["customer_id"]
        )

    rows: list[dict[str, Any]] = []
    for region in dims.regions:
        for channel in dims.channels:
            for product in dims.products:
                line_name = line_name_of[product["product_line_id"]]
                for month in months:
                    rng = _rng(
                        "sales",
                        region["region_name"],
                        channel["channel_name"],
                        product["product_name"],
                        month.isoformat(),
                    )
                    factor = year_factor(
                        construction,
                        when=month,
                        region=region["region_name"],
                        channel=channel["channel_name"],
                        line=line_name,
                    )
                    seasonal = Decimal(str(_seasonal(month)))
                    pool = customers_of[(region["region_id"], channel["channel_id"])]
                    for index in range(rows_per_combo):
                        row = _base_row(
                            rng,
                            when=month + timedelta(days=index % 27),
                            region=region["region_name"],
                            channel=channel["channel_name"],
                            product=product,
                            customer_id=pool[index % len(pool)],
                            now=now,
                        )
                        noise = Decimal(str(1 + rng.uniform(-ROW_NOISE, ROW_NOISE)))
                        scale = factor * seasonal * noise
                        row["region_id"] = region["region_id"]
                        row["channel_id"] = channel["channel_id"]
                        for field in (
                            "gross_amount",
                            "discount_amount",
                            "return_amount",
                            "net_amount",
                        ):
                            row[field] = _money(Decimal(row[field]) * scale)
                        row["quantity"] = max(1, int(row["quantity"] * float(scale)))
                        rows.append(row)
    return rows


def calibrate(
    rows_2024: list[dict[str, Any]], rows_2025: list[dict[str, Any]], dims: Dimensions
) -> tuple[Decimal, Decimal]:
    """把 2025 Q3 华东的落点**精确**校准到目标比例。

    为什么需要这一步：生成用的是「因子 + 逐行噪声」。噪声在组合数多时会相互
    抵消（全量 50 万行时每个组合月有 52 行），但在小样本下不会——2 万行试跑时
    每组合月只有 2 行，华东整体会偏出 3 个百分点，断言直接不通过。

    这暴露的是设计问题而不是测试问题：**聚合结论不该依赖样本量**。
    反向构造的最后一步本就该是校准——「先定结论」意味着结论是精确的，
    噪声只允许留在明细层。校准后无论跑 2 万行还是 50 万行，-12% 都是 -12%。

    返回 (异常组合实际修正系数, 其余组合实际修正系数)，用于打印核对。
    """
    huadong = dims.region_id(ANOMALY_REGION)
    anomaly_channel = dims.channel_id(ANOMALY_CHANNEL)
    anomaly_line = dims.product_line_id(ANOMALY_LINE)
    line_of = {p["product_id"]: p["product_line_id"] for p in dims.products}

    def is_anomaly(row: dict[str, Any]) -> bool:
        # bool()：两侧都是 Any（行是 dict[str, Any]），`and` 的结果被推成 Any，
        # 而函数声明返回 bool。显式收口比放宽返回类型更容易在调用处发现错误。
        return bool(
            row["channel_id"] == anomaly_channel and line_of[row["product_id"]] == anomaly_line
        )

    def q3_huadong(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if r["region_id"] == huadong and _quarter(r["order_date"]) == 3]

    base_rows = q3_huadong(rows_2024)
    base_total = sum((Decimal(r["net_amount"]) for r in base_rows), Decimal(0))
    base_anomaly = sum((Decimal(r["net_amount"]) for r in base_rows if is_anomaly(r)), Decimal(0))
    target_rows = q3_huadong(rows_2025)
    actual_anomaly = sum(
        (Decimal(r["net_amount"]) for r in target_rows if is_anomaly(r)), Decimal(0)
    )
    actual_other = sum((Decimal(r["net_amount"]) for r in target_rows), Decimal(0)) - actual_anomaly

    anomaly_fix = (ANOMALY_YOY * base_anomaly) / actual_anomaly
    other_fix = (REGION_YOY * base_total - ANOMALY_YOY * base_anomaly) / actual_other

    for row in target_rows:
        fix = anomaly_fix if is_anomaly(row) else other_fix
        for field in ("gross_amount", "discount_amount", "return_amount", "net_amount"):
            row[field] = _money(Decimal(row[field]) * fix)
        # quantity 不缩放：修正的语义是「价格 / 产品结构变化」，
        # 而不是「凭空多卖了货」。缩数量会让「件单价」这个派生口径跟着漂。

    return anomaly_fix, other_fix


# ------------------------------------------------------------------ 目标与库存
def generate_targets(
    dims: Dimensions, sales: list[dict[str, Any]], months: list[date], now: datetime
) -> list[dict[str, Any]]:
    """月度目标 = 2024 同期实际 × 1.05。

    目标是**回望**的：华东 Q3 的达成率因此会明显偏低，
    这正是「目标达成率」这个指标在下钻案例里的用处。
    """
    product_line_of = {p["product_id"]: p["product_line_id"] for p in dims.products}
    actual: dict[tuple[str, str, date], Decimal] = defaultdict(Decimal)
    for row in sales:
        actual[(row["region_id"], product_line_of[row["product_id"]], row["order_date"])] += (
            Decimal(row["net_amount"])
        )

    targets: list[dict[str, Any]] = []
    for region in dims.regions:
        for line in dims.product_lines:
            for month in months:
                if month.year != TARGET_YEAR:
                    continue
                base = sum(
                    amount
                    for (rid, lid, when), amount in actual.items()
                    if rid == region["region_id"]
                    and lid == line["product_line_id"]
                    and when == month
                )
                targets.append(
                    {
                        "target_id": deterministic_id(
                            IdPrefix.ORDER,
                            f"target-{region['region_name']}-{line['product_line_name']}-{month.isoformat()}",
                        ),
                        "period_month": month,
                        "region_id": region["region_id"],
                        "product_line_id": line["product_line_id"],
                        "target_amount": _money(base * Decimal("1.05")),
                        "created_at": now,
                    }
                )
    return targets


def generate_inventory(dims: Dimensions, months: list[date], now: datetime) -> list[dict[str, Any]]:
    """周库存快照。约束 3 的落点。

    异常产品线在华东、2025 Q3 起连续 `ANOMALY_LOW_WEEKS` 周低于安全线，
    其余 (区域, 产品线) 全程高于安全线——**包括异常产品线在其他区域**，
    否则「只有华东这一处」这个结论就不成立。
    """
    weeks: list[date] = []
    cursor = START - timedelta(days=START.weekday())
    while cursor <= END:
        weeks.append(cursor)
        cursor += timedelta(days=7)

    anomaly_start = date(TARGET_YEAR, TARGET_QUARTER_MONTHS[0], 1)
    anomaly_start -= timedelta(days=anomaly_start.weekday())

    rows: list[dict[str, Any]] = []
    for region in dims.regions:
        for line in dims.product_lines:
            rng = _rng("inventory", region["region_name"], line["product_line_name"])
            for week in weeks:
                is_anomaly_window = (
                    region["region_name"] == ANOMALY_REGION
                    and line["product_line_name"] == ANOMALY_LINE
                    and anomaly_start <= week < anomaly_start + timedelta(weeks=ANOMALY_LOW_WEEKS)
                )
                # 正常周：库存是安全线的 1.3–2.0 倍
                available = int(SAFETY_STOCK * rng.uniform(1.3, 2.0))
                if is_anomaly_window:
                    # 异常周：安全线的 0.45–0.80 倍，明确低于线
                    available = int(SAFETY_STOCK * rng.uniform(0.45, 0.80))
                rows.append(
                    {
                        "snapshot_id": deterministic_id(
                            IdPrefix.ORDER,
                            f"snap-{region['region_name']}-{line['product_line_name']}-{week.isoformat()}",
                        ),
                        "snapshot_week": week,
                        "region_id": region["region_id"],
                        "product_line_id": line["product_line_id"],
                        "available_qty": available,
                        "safety_stock": SAFETY_STOCK,
                        "warehouse_count": rng.randint(1, 4),
                        "created_at": now,
                    }
                )
    return rows


# ------------------------------------------------------------------ 写库
async def apply_schema(conn: AsyncConnection) -> None:
    ddl = (Path(__file__).parent / "business_schema.sql").read_text(encoding="utf-8")
    # 按 `;` 切分并逐条执行：asyncmy 不支持一次提交多条语句（除非开
    # CLIENT.MULTI_STATEMENTS，而那会顺带放开 SQL 注入面）。DDL 是静态文件，
    # 这里切分不引入注入风险。
    for statement in filter(None, (s.strip() for s in ddl.split(";"))):
        if statement.startswith("--"):
            statement = "\n".join(
                line for line in statement.splitlines() if not line.strip().startswith("--")
            ).strip()
        if statement:
            await conn.execute(text(statement))


async def insert_rows(conn: AsyncConnection, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    placeholders = ", ".join(f":{name}" for name in columns)
    statement = text(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})")  # noqa: S608 - 表名与列名来自本文件的常量，非用户输入
    for start in range(0, len(rows), BATCH_ROWS):
        await conn.execute(statement, rows[start : start + BATCH_ROWS])


async def truncate_all(conn: AsyncConnection) -> None:
    """清空事实与维度表。顺序无关（没有物理外键），但维度先清更贴合直觉。"""
    for table in (
        "inventory_snapshot",
        "sales_target",
        "fact_sales_order_item",
        "dim_customer",
        "dim_product",
        "dim_product_line",
        "dim_channel",
        "dim_region",
    ):
        await conn.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - 表名是上面的常量


# ------------------------------------------------------------------ 断言
@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    detail: str


async def verify(conn: AsyncConnection) -> list[Check]:
    """四条约束的可执行形式。全部走只读连接（与 SQL Tool 同一条路径）。"""
    checks: list[Check] = []

    async def scalar(sql: str, **params: Any) -> Decimal:
        value = (await conn.execute(text(sql), params)).scalar()
        return Decimal(value if value is not None else 0)

    # ---- 约束 1：华东 2025 Q3 同比约 -12%
    huadong = (
        await conn.execute(
            text("SELECT region_id FROM dim_region WHERE region_name = :n"), {"n": ANOMALY_REGION}
        )
    ).scalar_one()
    yoy_sql = """
        SELECT
          SUM(CASE WHEN YEAR(order_date) = 2025 THEN net_amount ELSE 0 END)
          / NULLIF(SUM(CASE WHEN YEAR(order_date) = 2024 THEN net_amount ELSE 0 END), 0)
        FROM fact_sales_order_item
        WHERE region_id = :region AND QUARTER(order_date) = 3
    """
    region_yoy = await scalar(yoy_sql, region=huadong)
    region_decline_ok = Decimal("0.865") <= region_yoy <= Decimal("0.895")
    checks.append(
        Check(
            "约束1a 华东 Q3 同比约 -12%",
            region_decline_ok,
            f"实际 {region_yoy:.4f}（目标 0.88，容忍 ±1.5pp）",
        )
    )

    # ---- 约束 1b：华东贡献了总降幅的多数
    per_region = (
        await conn.execute(
            text("""
                SELECT r.region_name,
                       SUM(CASE WHEN YEAR(f.order_date) = 2025 THEN f.net_amount ELSE 0 END)
                - SUM(CASE WHEN YEAR(f.order_date) = 2024 THEN f.net_amount ELSE 0 END) AS delta
                FROM fact_sales_order_item f
                JOIN dim_region r ON r.region_id = f.region_id
                WHERE QUARTER(f.order_date) = 3
                GROUP BY r.region_name
            """)
        )
    ).all()
    deltas = {name: Decimal(delta) for name, delta in per_region}
    total_delta = sum(deltas.values())
    huadong_delta = deltas.get(ANOMALY_REGION, Decimal(0))
    dominance = Decimal(0) if total_delta >= 0 else (-huadong_delta / -total_delta)
    checks.append(
        Check(
            "约束1b 华东贡献总降幅的多数",
            total_delta < 0 and dominance >= Decimal("0.5"),
            # 占比会 > 100%：其他区域在增长，是华东一个区域把大盘拉下来的。
            # 这不是数据错误，恰恰是「第一步区域对比能定位到具体区域」的证明。
            f"全公司净降幅 {total_delta:,.0f}，华东 {-huadong_delta:,.0f}"
            f"（占净降幅 {dominance:.0%}，其余区域合计为正增长）",
        )
    )

    # ---- 约束 4：其他区域不出现同类异常（|同比| < 5%）
    others = {
        name: await scalar(yoy_sql, region=rid)
        for name, rid in (
            await conn.execute(
                text("SELECT region_name, region_id FROM dim_region WHERE region_name <> :n"),
                {"n": ANOMALY_REGION},
            )
        ).all()
    }
    others_ok = all(abs(v - 1) < Decimal("0.05") for v in others.values())
    checks.append(
        Check(
            "约束4a 其他区域保持正常波动（|同比| < 5%）",
            others_ok,
            "、".join(f"{k} {v:.3f}" for k, v in sorted(others.items())),
        )
    )

    # ---- 约束 2：异常产品线在异常渠道的降幅显著高于同渠道其他产品线
    line_sql = """
        SELECT ln.product_line_name,
               SUM(CASE WHEN YEAR(f.order_date) = 2025 THEN f.net_amount ELSE 0 END)
               / NULLIF(SUM(CASE WHEN YEAR(f.order_date) = 2024
                                 THEN f.net_amount ELSE 0 END), 0) AS yoy
        FROM fact_sales_order_item f
        JOIN dim_product p ON p.product_id = f.product_id
        JOIN dim_product_line ln ON ln.product_line_id = p.product_line_id
        JOIN dim_channel c ON c.channel_id = f.channel_id
        WHERE f.region_id = :region AND c.channel_name = :channel AND QUARTER(f.order_date) = 3
        GROUP BY ln.product_line_name
    """
    line_yoy = {
        name: Decimal(value)
        for name, value in (
            await conn.execute(text(line_sql), {"region": huadong, "channel": ANOMALY_CHANNEL})
        ).all()
    }
    anomaly_yoy = line_yoy[ANOMALY_LINE]
    others_max = max(v for k, v in line_yoy.items() if k != ANOMALY_LINE)
    # **判定用百分点差，不用倍数**。「显著高于」的语义是两个降幅之间的差距；
    # 用「≤ 其他线的一半」会隐含要求其他线正增长 10%（0.55 ≤ 0.5 × 1.10），
    # 而在一个整体下滑 12% 的区域里，其他线不可能正增长——那条断言
    # 从一开始就没有解。20 个百分点是这条数据里「显著」的可判定形式。
    checks.append(
        Check(
            "约束2a 异常产品线降幅显著高于同渠道其他产品线（差 ≥ 20pp）",
            anomaly_yoy <= others_max - Decimal("0.20"),
            f"{ANOMALY_LINE} {anomaly_yoy:.3f} vs 其他最高 {others_max:.3f}"
            f"（差 {(others_max - anomaly_yoy) * 100:.1f}pp）",
        )
    )
    checks.append(
        Check(
            "约束2b 异常产品线降幅 ≤ -30%",
            anomaly_yoy <= Decimal("0.70"),
            f"实际 {anomaly_yoy:.3f}",
        )
    )

    # ---- 约束 4b：其他产品线在华东同渠道不出现同类异常
    others_lines_ok = all(v > Decimal("0.80") for k, v in line_yoy.items() if k != ANOMALY_LINE)
    checks.append(
        Check(
            "约束4b 同渠道其他产品线无同类异常（> -20%）",
            others_lines_ok,
            "、".join(f"{k} {v:.3f}" for k, v in sorted(line_yoy.items())),
        )
    )

    # ---- 约束 3：连续 6 周以上库存低于安全线
    low_weeks = await scalar(
        """
        SELECT COUNT(*) FROM inventory_snapshot s
        JOIN dim_region r ON r.region_id = s.region_id
        JOIN dim_product_line ln ON ln.product_line_id = s.product_line_id
        WHERE r.region_name = :region AND ln.product_line_name = :line
          AND s.available_qty < s.safety_stock
          AND s.snapshot_week >= '2025-07-01' AND s.snapshot_week < '2025-10-01'
        """,
        region=ANOMALY_REGION,
        line=ANOMALY_LINE,
    )
    checks.append(
        Check(
            f"约束3 异常产品线连续 ≥6 周低于安全线（实际 {int(low_weeks)} 周）",
            low_weeks >= 6,
            f"华东×{ANOMALY_LINE} 2025Q3 低于安全线 {int(low_weeks)} 周",
        )
    )

    # ---- 约束 4c：其他区域 / 产品线的库存不出现同类异常
    other_low = await scalar(
        """
        SELECT COUNT(*) FROM inventory_snapshot s
        JOIN dim_region r ON r.region_id = s.region_id
        JOIN dim_product_line ln ON ln.product_line_id = s.product_line_id
        WHERE s.available_qty < s.safety_stock
          AND NOT (r.region_name = :region AND ln.product_line_name = :line)
        """,
        region=ANOMALY_REGION,
        line=ANOMALY_LINE,
    )
    checks.append(
        Check(
            "约束4c 其他区域/产品线库存均不低于安全线",
            other_low == 0,
            f"越线快照 {int(other_low)} 条",
        )
    )

    return checks


async def explain_checks(conn: AsyncConnection) -> list[Check]:
    """用 EXPLAIN 验证组合索引**真的被用上**（16.10 明确要求）。

    凭直觉建索引的问题不是「多占了空间」，而是「以为有索引、其实全表扫」——
    在 50 万行上它会安静地跑得动，等数据涨上去才暴露。
    """
    huadong = (
        await conn.execute(
            text("SELECT region_id FROM dim_region WHERE region_name = :n"), {"n": ANOMALY_REGION}
        )
    ).scalar_one()
    probes: tuple[tuple[str, str, dict[str, Any]], ...] = (
        (
            "区域 × 时间区间（区域对比 / 同比）",
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE region_id = :region AND order_date BETWEEN :a AND :b",
            {"region": huadong, "a": date(2025, 7, 1), "b": date(2025, 9, 30)},
        ),
        (
            "时间区间 × 区域 × 渠道（下钻到渠道）",
            "SELECT channel_id, SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date BETWEEN :a AND :b AND region_id = :region GROUP BY channel_id",
            {"region": huadong, "a": date(2025, 7, 1), "b": date(2025, 9, 30)},
        ),
        (
            "时间区间 × 产品（下钻到产品线）",
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date BETWEEN :a AND :b AND product_id = :product",
            {"a": date(2025, 7, 1), "b": date(2025, 9, 30), "product": ""},
        ),
    )

    checks: list[Check] = []
    for name, sql, params in probes:
        if params.get("product") == "":
            params = {
                **params,
                "product": (
                    await conn.execute(text("SELECT product_id FROM dim_product LIMIT 1"))
                ).scalar_one(),
            }
        plan_rows = (await conn.execute(text(f"EXPLAIN {sql}"), params)).all()
        # **按列名取，不按位置取**。MySQL 的 EXPLAIN 结果是一种「宽松」的行格式：
        # 各版本列数会变（8.0 起多了 partitions、filtered、rows）。
        # 之前这里写的是 `row[5] != "ALL"`，而第 5 列是 possible_keys ——
        # 它永远不是 'ALL'，于是**全表扫也被判成走索引**，
        # 断言恒真、门禁形同不存在。这正是「检查器自己失效」的典型形态。
        used = all(str(row._mapping["type"]) != "ALL" for row in plan_rows) and all(
            row._mapping.get("key") is not None for row in plan_rows
        )
        plan = " | ".join(
            f"type={row._mapping['type']} key={row._mapping.get('key')} "
            f"rows={row._mapping.get('rows')} extra={row._mapping.get('Extra')}"
            for row in plan_rows
        )
        checks.append(Check(f"EXPLAIN 走索引：{name}", used, plan[:220]))
    return checks


# ------------------------------------------------------------------ 入口
def _rw_url(settings: Settings) -> str:
    url = settings.database_url_business_rw
    if not url:
        print(
            "缺少 DATABASE_URL_BUSINESS_RW。业务库对应用侧只开放只读账号，\n"
            "建表与灌数需要单独的写连接（见 .env.example）。",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return url


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    ro_engine = create_async_engine(settings.database_url_business_ro)

    try:
        if not args.verify_only:
            now = datetime.now(UTC).replace(tzinfo=None)
            dims = build_dimensions(now)
            months = _month_iter(START, END)
            combos = len(dims.regions) * len(dims.channels) * len(dims.products) * len(months)
            rows_per_combo = max(1, args.rows // combos)

            print(f"生成维度…（{len(dims.products)} 个产品，{len(dims.customers)} 个客户）")
            rw_engine = create_async_engine(_rw_url(settings))
            try:
                async with rw_engine.begin() as conn:
                    await apply_schema(conn)
                async with rw_engine.begin() as conn:
                    await truncate_all(conn)
                    await insert_rows(conn, "dim_region", list(dims.regions))
                    await insert_rows(conn, "dim_channel", list(dims.channels))
                    await insert_rows(conn, "dim_product_line", list(dims.product_lines))
                    await insert_rows(conn, "dim_product", list(dims.products))
                    await insert_rows(conn, "dim_customer", list(dims.customers))

                # 先生成 2024，用它解出因子，再生成 2025 —— 这就是「先定结论再生成明细」
                months_2024 = [m for m in months if m.year == BASE_YEAR]
                months_2025 = [m for m in months if m.year == TARGET_YEAR]
                sales_2024 = generate_sales(
                    dims,
                    Construction(Decimal(1), Decimal(0), Decimal(0)),
                    months_2024,
                    rows_per_combo,
                    now,
                )
                construction = solve_factors(sales_2024, dims)
                print(
                    f"反向构造：异常组合占华东 Q3 的 {construction.anomaly_share:.1%}，"
                    f"其余组合同比因子解为 {construction.other_factor:.4f}"
                )
                sales_2025 = generate_sales(dims, construction, months_2025, rows_per_combo, now)
                anomaly_fix, other_fix = calibrate(sales_2024, sales_2025, dims)
                print(
                    f"校准：异常组合 ×{anomaly_fix:.4f}、其余组合 ×{other_fix:.4f}"
                    "（使 -12% 与 -45% 精确成立，与样本量无关）"
                )
                sales = sales_2024 + sales_2025

                targets = generate_targets(dims, sales, months, now)
                inventory = generate_inventory(dims, months, now)
                print(
                    f"写入：订单明细 {len(sales):,} 行、目标 {len(targets):,} 行、"
                    f"库存快照 {len(inventory):,} 行"
                )
                async with rw_engine.begin() as conn:
                    await insert_rows(conn, "fact_sales_order_item", sales)
                    await insert_rows(conn, "sales_target", targets)
                    await insert_rows(conn, "inventory_snapshot", inventory)
                # **批量导入后必须 ANALYZE**：优化器在此之前拿的是「空表」的
                # 统计信息，会据此判定全表扫比走索引便宜。少了这一步，
                # EXPLAIN 断言会把「统计信息过期」误判成「索引建得不对」，
                # 而两者要修的东西完全不同。
                async with rw_engine.begin() as conn:
                    for table in ("fact_sales_order_item", "sales_target", "inventory_snapshot"):
                        await conn.execute(text(f"ANALYZE TABLE {table}"))
            finally:
                await rw_engine.dispose()

        print("\n校验（走只读账号，与 SQL Tool 同一条取数路径）：")
        async with ro_engine.connect() as conn:
            checks = await verify(conn) + await explain_checks(conn)
    finally:
        await ro_engine.dispose()

    failed = 0
    for check in checks:
        mark = "  [OK]" if check.passed else "  [!!]"
        print(f"{mark} {check.name}\n       {check.detail}")
        failed += 0 if check.passed else 1

    if failed:
        print(f"\n{failed} 条断言未通过——**不得进入 Phase 4**（开发流程 6.4 门禁）")
        return 1
    print(f"\n全部 {len(checks)} 条断言通过。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="业务演示库建表与种子数据（反向构造）")
    parser.add_argument(
        "--rows",
        type=int,
        default=500_000,
        help="订单明细目标行数（详细设计 16.10 的规模下限为 50 万）",
    )
    parser.add_argument("--verify-only", action="store_true", help="只跑断言，不重新生成数据")
    return asyncio_run(run(parser.parse_args()))


def asyncio_run(coro: Any) -> int:
    import asyncio

    return int(asyncio.run(coro))


if __name__ == "__main__":
    sys.exit(main())
