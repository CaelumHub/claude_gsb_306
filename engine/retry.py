"""失败自动重试：策略模型、优先级解析与重试判定。

为什么需要重试
--------------
接口类用例常有「偶发失败」（网络抖动、依赖服务瞬时不可用）：隔几分钟重跑
又绿了，但报告里永远记着它失败，团队对稳定性的判断因此失真。自动重试让
「偶发抖动」在当次构建内自我修正：用例失败后按策略再试若干次，**最后一轮
的结果**作为该用例的结论，同时保留每一轮结果供回溯（``attempts``）。

策略字段（用例 ``case["retry"]`` 与套件 ``suite["retry"]`` 同构）
------------------------------------------------------------------
``max_attempts``         最大尝试次数（含首轮），1 表示不重试，平台硬上限 5
``retry_on``             哪些终态触发重试（failed / error / timeout 的子集）
``only_assertion_types`` 非空时，仅当失败断言的类型命中才重试（如 ["status"]）
``only_envs``            非空时，仅在列出的环境 id 下重试
``backoff_ms``           每轮重试前的等待毫秒数（分片睡眠，可取消）
``max_extra_duration``   重试额外耗时预算（秒）：第 2 轮起的累计执行耗时 +
                         退避等待超过预算即停止重试；0 / None 表示不限。
                         与次数上限共同防止重试让构建时长成倍膨胀。

优先级（与配置先后顺序无关）
----------------------------
1. 用例级 ``case["retry"]`` 里**出现的字段**优先；
2. 套件级 ``suite["retry"]`` 的字段作为默认补全；
3. 两者都没配置 → 平台默认（不重试）。

即用例可以只覆盖它关心的字段（例如 ``{"max_attempts": 1}`` 显式关掉套件
下发的重试），未写的字段继承套件——这是确定性的字段级覆盖，而不是
「谁后配置谁生效」。
"""

from __future__ import annotations

from typing import Optional

# 可触发重试的终态
RETRYABLE_STATUSES = ("failed", "error", "timeout")

# 平台硬上限：无论怎么配置，尝试次数不会超过它（防执行时间膨胀的最后防线）
MAX_ATTEMPTS_LIMIT = 5
MAX_BACKOFF_MS = 60_000
MAX_EXTRA_DURATION_LIMIT = 600.0

# 平台默认策略：不重试（max_attempts = 1）
DEFAULT_RETRY_POLICY = {
    "max_attempts": 1,
    "retry_on": list(RETRYABLE_STATUSES),
    "only_assertion_types": [],
    "only_envs": [],
    "backoff_ms": 0,
    "max_extra_duration": 60.0,
}


def normalize_policy(raw: Optional[dict]) -> Optional[dict]:
    """把用户配置清洗成合法策略；``raw`` 不是 dict 时返回 None。

    未知字段被丢弃；数值字段钳制到平台允许的范围内。``retry_on`` 给空列表
    表示「任何失败都不重试」（与 ``max_attempts = 1`` 等效），予以保留。
    """
    if not isinstance(raw, dict):
        return None
    policy = dict(DEFAULT_RETRY_POLICY)
    policy.update({k: v for k, v in raw.items() if k in policy})

    try:
        policy["max_attempts"] = int(policy["max_attempts"])
    except (TypeError, ValueError):
        policy["max_attempts"] = 1
    policy["max_attempts"] = max(1, min(MAX_ATTEMPTS_LIMIT, policy["max_attempts"]))

    retry_on = policy["retry_on"]
    if not isinstance(retry_on, list):
        retry_on = list(RETRYABLE_STATUSES)
    policy["retry_on"] = [s for s in retry_on if s in RETRYABLE_STATUSES]

    for key in ("only_assertion_types", "only_envs"):
        value = policy[key]
        policy[key] = [str(x) for x in value] if isinstance(value, list) else []

    try:
        policy["backoff_ms"] = int(policy["backoff_ms"])
    except (TypeError, ValueError):
        policy["backoff_ms"] = 0
    policy["backoff_ms"] = max(0, min(MAX_BACKOFF_MS, policy["backoff_ms"]))

    budget = policy["max_extra_duration"]
    try:
        budget = float(budget) if budget else 0.0
    except (TypeError, ValueError):
        budget = 0.0
    policy["max_extra_duration"] = max(0.0, min(MAX_EXTRA_DURATION_LIMIT, budget))
    return policy


def resolve_policy(case_retry: Optional[dict],
                   suite_retry: Optional[dict]) -> Optional[dict]:
    """解析某条用例本次执行的生效策略。

    字段级覆盖：套件策略作底，用例策略里出现的字段覆盖之；两者皆无返回
    None（调用方按「不重试」处理）。返回的策略已经过 :func:`normalize_policy`
    清洗，可直接使用。
    """
    case_retry = case_retry if isinstance(case_retry, dict) else None
    suite_retry = suite_retry if isinstance(suite_retry, dict) else None
    if not case_retry and not suite_retry:
        return None
    merged = dict(suite_retry or {})
    merged.update(case_retry or {})
    return normalize_policy(merged)


def should_retry(policy: Optional[dict], result: dict, env_id: Optional[str],
                 attempts_used: int, extra_duration: float) -> bool:
    """判断一轮失败后是否值得再试一次。

    依次检查：策略存在 → 次数未用尽 → 终态在 ``retry_on`` 内 → 环境限定 →
    断言类型限定 → 额外耗时预算。任一不满足即不再重试。
    """
    if not policy:
        return False
    if attempts_used >= policy["max_attempts"]:
        return False
    status = result.get("status")
    if status not in policy["retry_on"]:
        return False

    only_envs = policy.get("only_envs") or []
    if only_envs and env_id not in only_envs:
        return False

    only_types = policy.get("only_assertion_types") or []
    if only_types:
        # 「只在特定断言失败时重试」：error / timeout 并非断言失败，直接放行
        # 会违背语义，因此只有 failed 且失败断言类型命中时才重试。
        if status != "failed":
            return False
        failed_types = {a.get("type") for a in result.get("assertions", [])
                        if not a.get("ok")}
        if not (failed_types & set(only_types)):
            return False

    budget = policy.get("max_extra_duration") or 0.0
    if budget > 0 and extra_duration >= budget:
        return False
    return True
