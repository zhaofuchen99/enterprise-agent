-- 演示业务库 DDL（开发流程 6.4 施工项 5，详细设计 16.10）
--
-- 这是一套**模拟「企业已有系统」的只读数据**，不是本应用自己的表。
-- 因此它不走 Alembic：应用侧连的是只读账号，根本没有权限改结构；
-- 建表与灌数由 `scripts/business_seed.py` 用单独的写连接完成。
--
-- 全部 `IF NOT EXISTS`：seed 会被反复执行（换机器、清了库、改了演示数据），
-- 每次都在建表上炸掉会让它变成「只能用一次」的命令。
--
-- 金额口径（贯穿全项目，也是 RAG 缺陷注入的构造材料）：
--   gross_amount   含税销售额
--   discount_amount 折扣
--   return_amount  退货冲减
--   net_amount     净销售额 = gross - discount - return
-- 「报告数字与 DB 差 1–2%」这类冲突，正是靠"报告用含税、DB 用净额"制造的。

-- ------------------------------------------------------------------ 维度
CREATE TABLE IF NOT EXISTS dim_region (
  region_id      CHAR(26)    NOT NULL COMMENT '区域 ID',
  region_code    VARCHAR(16) NOT NULL COMMENT '区域编码',
  region_name    VARCHAR(32) NOT NULL COMMENT '区域名称，如「华东」',
  -- 「华东三省 vs 四省」这类 SCOPE 冲突的构造材料：
  -- 权威口径在库里，与语料中的说法可以不一致。
  province_count INT         NOT NULL DEFAULT 0 COMMENT '下辖省份数（口径）',
  is_anomaly     TINYINT(1)  NOT NULL DEFAULT 0 COMMENT '是否为下钻案例的目标区域（仅演示数据用，便于断言）',
  created_at     DATETIME(3) NOT NULL,
  PRIMARY KEY (region_id),
  UNIQUE KEY uk_region_name (region_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='区域维度';

CREATE TABLE IF NOT EXISTS dim_channel (
  channel_id   CHAR(26)    NOT NULL,
  channel_code VARCHAR(16) NOT NULL,
  channel_name VARCHAR(32) NOT NULL COMMENT '直营 / 经销 / 电商 / KA',
  channel_type VARCHAR(16) NOT NULL COMMENT 'DIRECT / PARTNER / ONLINE',
  created_at   DATETIME(3) NOT NULL,
  PRIMARY KEY (channel_id),
  UNIQUE KEY uk_channel_name (channel_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='渠道维度';

CREATE TABLE IF NOT EXISTS dim_product_line (
  product_line_id   CHAR(26)    NOT NULL,
  product_line_code VARCHAR(16) NOT NULL,
  product_line_name VARCHAR(64) NOT NULL,
  category          VARCHAR(32) NOT NULL,
  created_at        DATETIME(3) NOT NULL,
  PRIMARY KEY (product_line_id),
  UNIQUE KEY uk_product_line_name (product_line_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='产品线维度';

CREATE TABLE IF NOT EXISTS dim_product (
  product_id      CHAR(26)     NOT NULL,
  product_code    VARCHAR(32)  NOT NULL,
  product_name    VARCHAR(128) NOT NULL,
  product_line_id CHAR(26)     NOT NULL,
  list_price      DECIMAL(12,2) NOT NULL COMMENT '标价',
  launch_date     DATE         NULL,
  status          VARCHAR(16)  NOT NULL DEFAULT 'ACTIVE',
  created_at      DATETIME(3)  NOT NULL,
  PRIMARY KEY (product_id),
  UNIQUE KEY uk_product_code (product_code),
  KEY idx_product_line (product_line_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='产品维度';

CREATE TABLE IF NOT EXISTS dim_customer (
  customer_id   CHAR(26)     NOT NULL,
  customer_code VARCHAR(32)  NOT NULL,
  customer_name VARCHAR(128) NOT NULL,
  region_id     CHAR(26)     NOT NULL,
  channel_id    CHAR(26)     NOT NULL,
  customer_level VARCHAR(16) NOT NULL COMMENT 'A / B / C',
  created_at    DATETIME(3)  NOT NULL,
  PRIMARY KEY (customer_id),
  UNIQUE KEY uk_customer_code (customer_code),
  KEY idx_customer_region_channel (region_id, channel_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='客户维度';

-- ------------------------------------------------------------------ 事实
CREATE TABLE IF NOT EXISTS fact_sales_order_item (
  id              BIGINT        NOT NULL AUTO_INCREMENT,
  order_id        CHAR(26)      NOT NULL,
  order_date      DATE          NOT NULL COMMENT '下单日期。同比/环比一律按它切',
  region_id       CHAR(26)      NOT NULL,
  channel_id      CHAR(26)      NOT NULL,
  product_id      CHAR(26)      NOT NULL,
  customer_id     CHAR(26)      NOT NULL,
  quantity        INT           NOT NULL,
  gross_amount    DECIMAL(14,2) NOT NULL COMMENT '含税销售额',
  discount_amount DECIMAL(14,2) NOT NULL DEFAULT 0,
  return_amount   DECIMAL(14,2) NOT NULL DEFAULT 0 COMMENT '退货冲减',
  net_amount      DECIMAL(14,2) NOT NULL COMMENT '净销售额 = 含税 - 折扣 - 退货',
  currency        CHAR(3)       NOT NULL DEFAULT 'CNY',
  created_at      DATETIME(3)   NOT NULL,
  PRIMARY KEY (id),
  KEY idx_sales_order (order_id),
  -- 组合索引按「固定评测问题的取数路径」建，不是凭直觉堆：
  --   Q1 区域 × 时间区间（区域对比、同比）      -> (region_id, order_date)
  --   Q2 时间区间 × 区域 × 渠道（下钻到渠道）   -> (order_date, region_id, channel_id)
  --   Q3 时间区间 × 产品线（下钻到产品线）      -> (order_date, product_id)
  -- 三条覆盖下钻链路的三步，彼此不重复。是否真的被用上由 seed 的
  -- EXPLAIN 断言验证（详细设计 16.10：不能凭直觉建索引）。
  KEY idx_sales_region_date (region_id, order_date),
  KEY idx_sales_date_region_channel (order_date, region_id, channel_id),
  KEY idx_sales_date_product (order_date, product_id),
  KEY idx_sales_date_channel_product (order_date, channel_id, product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='销售订单明细（事实表）';

-- ------------------------------------------------------------------ 目标
CREATE TABLE IF NOT EXISTS sales_target (
  target_id      CHAR(26)      NOT NULL,
  period_month   DATE          NOT NULL COMMENT '目标月份（取当月 1 日）',
  region_id      CHAR(26)      NOT NULL,
  product_line_id CHAR(26)     NOT NULL,
  target_amount  DECIMAL(16,2) NOT NULL COMMENT '目标净销售额',
  created_at     DATETIME(3)   NOT NULL,
  PRIMARY KEY (target_id),
  UNIQUE KEY uk_target_month_region_line (period_month, region_id, product_line_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='销售目标';

-- ------------------------------------------------------------------ 库存
CREATE TABLE IF NOT EXISTS inventory_snapshot (
  snapshot_id     CHAR(26)    NOT NULL,
  snapshot_week   DATE        NOT NULL COMMENT '快照周（取当周周一）',
  region_id       CHAR(26)    NOT NULL,
  product_line_id CHAR(26)    NOT NULL,
  available_qty   INT         NOT NULL COMMENT '可用库存',
  safety_stock    INT         NOT NULL COMMENT '安全库存线',
  warehouse_count INT         NOT NULL DEFAULT 1,
  created_at      DATETIME(3) NOT NULL,
  PRIMARY KEY (snapshot_id),
  UNIQUE KEY uk_snapshot_week_region_line (snapshot_week, region_id, product_line_id),
  KEY idx_snapshot_region_line_week (region_id, product_line_id, snapshot_week)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='库存周快照';
