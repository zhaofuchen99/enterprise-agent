"""Redis 连接、键命名与脚本加载（开发流程 6.3 施工项 1）。

**这个文件存在的理由**：Redis 键名一旦散落在各个模块里，`task:{id}:cancel`
写错成 `task:{id}:cancelled` 不会报错，只会让取消功能静默失效——
排查时你看到的是「Worker 没响应取消」，而不是「键名拼错了」。
因此键名只有这一个定义处，详细设计 4.4 的表格与 `RedisKey` 一一对应。

**两条纪律（详细设计 4.4）**：

1. 缓存键必须带版本号，版本发布即天然失效，不需要任何 DEL 逻辑；
2. Redis 里的一切必须可由 MySQL 与原始文件重建。任务状态以 `agent_task` 为准，
   事件以 `agent_trace_event` 为准。**Phase 1.5 的任务记录是唯一的例外**——
   MySQL 要到 Phase 2 才有，届时 `task:{id}:record` 会被整份删除。
   这个例外登记在 CLAUDE.md 的「【后续扩展】」表里，不在这里长期存在。
"""

from __future__ import annotations

import json
from typing import Any, Final

import redis.asyncio as aioredis
from redis.commands.core import AsyncScript

from app.core.config import Settings

#: 任务队列（arq 管理，消费即删）。不参与业务逻辑，只用于排查时定位。
QUEUE_KEY: Final[str] = "q:agent"

#: 限流固定窗口脚本。返回 `{窗口内计数, 键剩余毫秒}`，一次往返拿齐响应头所需字段。
#:
#: INCR 与 EXPIRE 必须在同一个脚本里：分成两条命令时，若进程在两者之间退出，
#: 键会永久驻留，该用户从此被限流到天荒地老。脚本的原子性正是这里要买的东西。
#:
#: 相比详细设计 19.3 的示例脚本多返回了 PTTL，是为了让 `X-RateLimit-Reset`
#: 报出真实剩余时间而不是让客户端猜窗口边界。
RATE_LIMIT_SCRIPT: Final[str] = """
local key = KEYS[1]
local window_ms = tonumber(ARGV[1])
local count = redis.call('INCR', key)
if count == 1 then
  redis.call('PEXPIRE', key, window_ms)
end
local ttl_ms = redis.call('PTTL', key)
return {count, ttl_ms}
"""

#: 重投计数自增并返回新值。补偿扫描会并发运行（多实例各跑一份），
#: 「读-加-写」用脚本一次做完，重投次数才不会被算少。
INCREMENT_WITH_TTL_SCRIPT: Final[str] = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[1]))
end
return count
"""

#: 任务建档脚本：把「task_id 唯一 + 幂等键唯一 + 两个索引」做成一次原子操作。
#:
#: 为什么非要用脚本：这几步分四条命令发出去，中间任何一步失败都会留下
#: **半截状态**——记录写进去了但没进索引，于是它既不会被补偿扫描捞到、
#: 也不占并发配额，成了一个永远停在 QUEUED 且谁也看不见的幽灵任务。
#: 脚本在 Redis 里是原子执行的，`EXISTS` 与随后的写入之间没有窗口。
#:
#: 字段用 `while` 循环逐个 HSET 而不是 `unpack(ARGV)`：`unpack` 在 Lua 5.2
#: 改名为 `table.unpack`，而真实 Redis 是 Lua 5.1、fakeredis 走的 lupa
#: 可能是 5.4，写 `unpack` 会在其中一边直接报错。
#:
#: `3/4/5` 三个标志由 Python 侧根据 `TaskStatus` 算好再传进来，**不在 Lua 里
#: 拼状态字符串**：那种写法和枚举定义会各改各的，而分裂之后不会报错，
#: 只会表现为「某一类任务永远扫不到」。
#:
#: KEYS: 1=记录 2=幂等索引（'' 表示无幂等键） 3=active 4=queued 5=running
#: ARGV: 1=task_id 2=score 3=进 queued? 4=进 running? 5=进 active? 6...=字段/值交替
#: 返回：0=task_id 重复 1=幂等键重复 2=成功
ADD_TASK_SCRIPT: Final[str] = """
if redis.call('EXISTS', KEYS[1]) == 1 then
  return 0
end
if KEYS[2] ~= '' and redis.call('EXISTS', KEYS[2]) == 1 then
  return 1
end
local i = 6
while i < #ARGV do
  redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1])
  i = i + 2
end
if KEYS[2] ~= '' then
  redis.call('SET', KEYS[2], ARGV[1])
end
if ARGV[5] == '1' then redis.call('ZADD', KEYS[3], ARGV[2], ARGV[1]) end
if ARGV[3] == '1' then redis.call('ZADD', KEYS[4], ARGV[2], ARGV[1]) end
if ARGV[4] == '1' then redis.call('ZADD', KEYS[5], ARGV[2], ARGV[1]) end
return 2
"""

#: 状态流转的 CAS：只有当前状态等于预期值才写入。
#:
#: 「先读出来判断、再写回去」在并发下等于没判断——两个 Worker 可以同时读到
#: QUEUED，然后双双写 RUNNING。领取互斥（QUEUED -> RUNNING）靠的就是这个脚本：
#: 判断与写入在 Redis 里是一次原子执行。
#:
#: KEYS: 1=任务记录
#: ARGV: 1=预期状态（'' 表示不校验） 2...=字段名/值交替
#: 返回：0=任务不存在 1=状态不符 2=成功
CAS_UPDATE_SCRIPT: Final[str] = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end
if ARGV[1] ~= '' and redis.call('HGET', KEYS[1], 'status') ~= ARGV[1] then
  return 1
end
local i = 2
while i < #ARGV do
  redis.call('HSET', KEYS[1], ARGV[i], ARGV[i + 1])
  i = i + 2
end
return 2
"""

#: 释放互斥锁：只删自己持有的那一把。
#:
#: 不能直接 DEL。锁带 TTL，持有者的操作万一超时，锁会过期并被别人拿走，
#: 此时持有者再来 DEL 就会把**别人的锁**删掉，于是两个实例同时进入临界区。
#: 比对 token 再删，是分布式锁唯一正确的释放方式。
#:
#: KEYS: 1=锁键  ARGV: 1=持有者 token
RELEASE_LOCK_SCRIPT: Final[str] = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

SCRIPTS: Final[dict[str, str]] = {
    "rate_limit": RATE_LIMIT_SCRIPT,
    "increment_with_ttl": INCREMENT_WITH_TTL_SCRIPT,
    "add_task": ADD_TASK_SCRIPT,
    "cas_update": CAS_UPDATE_SCRIPT,
    "release_lock": RELEASE_LOCK_SCRIPT,
}


class RedisKey:
    """键命名。详细设计 4.4 的表格 + 本阶段的三个临时索引。

    全部写成 `@staticmethod` 而不是模块级函数，是为了让调用处一眼看出
    「这是个键」而不是某个业务函数：`RedisKey.task_cancel(task_id)`。
    """

    # ------------------------------------------------ 详细设计 4.4 已定义的键
    @staticmethod
    def queue() -> str:
        return QUEUE_KEY

    @staticmethod
    def task_heartbeat(task_id: str) -> str:
        return f"task:{task_id}:heartbeat"

    @staticmethod
    def task_cancel(task_id: str) -> str:
        return f"task:{task_id}:cancel"

    @staticmethod
    def task_events(task_id: str) -> str:
        return f"task:{task_id}:events"

    @staticmethod
    def rate_limit(scope: str, subject: str) -> str:
        return f"rl:{scope}:{subject}"

    @staticmethod
    def cache(name: str, version: str) -> str:
        return f"cache:{name}:{version}"

    @staticmethod
    def lock(purpose: str) -> str:
        return f"lock:{purpose}"

    # ------------------------------------ 【Phase 1.5 临时】Phase 2 接入 MySQL 后删除
    @staticmethod
    def task_record(task_id: str) -> str:
        """任务记录。**权威存储在 Phase 2 移到 `agent_task` 表**，此处只是过渡。"""
        return f"task:{task_id}:record"

    @staticmethod
    def task_requeue_count(task_id: str) -> str:
        """队列补偿扫描的重投次数（详细设计 17.1：最多重投 2 次）。"""
        return f"task:{task_id}:requeue"

    @staticmethod
    def idempotency_index(user_id: str, idempotency_key: str) -> str:
        """幂等键索引。对应 Phase 2 的 `uk_task_user_idempotency` 唯一索引。"""
        return f"idx:task:idem:{user_id}:{idempotency_key}"

    @staticmethod
    def active_tasks(user_id: str) -> str:
        """占用并发配额的任务 id 集合（ZSET，score 为 updated_at 时间戳）。

        用 ZSET 而不是 SET：并发配额泄漏时（任务进入终态却没有被移除），
        可以按 score 从旧到新清理，而不必把整集合读出来再判断。
        """
        return f"idx:task:active:{user_id}"

    @staticmethod
    def queued_tasks() -> str:
        """待领取任务的索引（ZSET，score 为 `queued_at` 时间戳）。

        补偿扫描要找出「写库成功但没进队列」的任务（详细设计 17.1），
        这类任务恰恰**不在** arq 队列里，只能靠任务侧的索引枚举。
        """
        return "idx:task:queued"

    @staticmethod
    def running_tasks() -> str:
        """运行中任务的索引（ZSET，score 为 `heartbeat_at` 时间戳）。

        对应 Phase 2 `agent_task` 上 `(status, heartbeat_at)` 的索引：
        孤儿回收要按心跳时间做范围查询，全表扫 RUNNING 任务在多实例下不可行。
        """
        return "idx:task:running"


# ------------------------------------------------------------------ 序列化约定
#: Redis 中一律存 UTF-8 的 JSON 文本，不存 pickle、不存二进制。
#:
#: pickle 是**代码执行入口**：任何能写 Redis 的人都能构造出反序列化即执行的载荷。
#: 而我们同时又是「Redis 里的东西都能被重建」的，读取方可能不是本进程。
def dump_json(payload: Any) -> str:
    """序列化为 JSON 文本。`ensure_ascii=False` 让中文可读，便于直接 redis-cli 排查。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def load_json(raw: str | bytes | None) -> Any:
    """反序列化。空值返回 None，由调用方决定缺失语义。"""
    if raw is None or raw == "":
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


# ---------------------------------------------------------------------- 连接
def create_client(settings: Settings, *, decode_responses: bool = True) -> aioredis.Redis:
    """按配置建连接池。

    `decode_responses=True`：本项目的 Redis 内容全是 JSON 文本与计数器，
    统一用 str 可以免掉调用处到处 `.decode()`——那些散落的 decode 正是
    一处漏写就变成 `b'...'` 混进日志的来源。

    `socket_timeout` 取 `redis_tuning.operation_timeout_seconds`：
    连接层面的兜底超时，防止 TCP 半开时命令永久挂起。
    """
    # redis-py 的 from_url 既没有类型标注、返回值也是 Any，两条都要就地豁免
    return aioredis.from_url(  # type: ignore[no-untyped-call, no-any-return]
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=decode_responses,
        socket_timeout=settings.redis_tuning.operation_timeout_seconds,
        socket_connect_timeout=settings.redis_tuning.operation_timeout_seconds,
    )


def register_scripts(client: aioredis.Redis) -> dict[str, AsyncScript]:
    """把 Lua 脚本注册成可调用对象。

    用 `register_script` 而不是每次 `eval`：redis-py 会先发 EVALSHA，
    遇到 NOSCRIPT（脚本没被缓存过，比如 Redis 刚重启）自动回退 EVAL 并缓存。
    手写 `eval` 每次都要传整段脚本文本，多实例下还会反复触发全量脚本传输。
    """
    return {name: client.register_script(source) for name, source in SCRIPTS.items()}
