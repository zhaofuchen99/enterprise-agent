-- 演示业务库初始化（Phase 0：只做权限收紧）
--
-- SQL Tool 通过只读账号访问业务库，这是「写操作必须被数据库拒绝」的底座
-- （详细设计 19.2 第 5 条：SQL 安全不依赖模型）。
-- 业务表结构与演示数据在 Phase 2 用迁移脚本灌入。

-- MYSQL_USER 默认授予了 ALL，这里收敛为只读
REVOKE ALL PRIVILEGES ON `business`.* FROM 'readonly'@'%';
GRANT SELECT, SHOW VIEW ON `business`.* TO 'readonly'@'%';
FLUSH PRIVILEGES;
