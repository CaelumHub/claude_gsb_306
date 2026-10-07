"""失败用例自动重试：策略解析、优先级合并、重试条件判定与重试执行。

背景
----
接口用例偶发抖动（网络瞬断、环境波动），首轮失败几分钟后手动重跑又能过。
若「抖一下就判死刑」，团队对构建稳定性的判断会失真。重试机制让用例在满足
条件时自动再跑几轮，并以**重试后的最终结果**作为用例结论，同时完整保留
每一轮结果、清楚标注重试次数。

策略结构（用例 / 套件共用同一份字段）
------------------------------------
``retry_policy``::

    {
      "max_retries": 2,          # 首轮失败后最多再跑几次；0 = 不重试（硬上限 5）
      "retry_on_status": [...],  # 仅当结果状态落在其中才重试，默认 failed/error/timeout
      "retry_on_assertions": [], # 可选：仅这些断言类型失败时重试；空 = 不限制
      "env_ids": [],             # 可选：仅在这些环境下重试；空 = 任意环境
      "backoff_ms": 200,         # 首轮重试前的基础退避（毫秒）
      "backoff_factor": 1.5,     # 退避递增倍数（1.0 = 固定间隔）
      "max_backoff_ms": 2000,    # 单次退避上限
      "retry_budget_ms": 0,      # 重试总时间预算（不含首轮）；0 = 不额外设限
    }

优先级（关键）
--------------
套件可以配置一份**默认**重试策略，用例可以**单独**配置。两者冲突时按

    用例 retry_policy  >  套件 retry_policy  >  系统默认（不重试）

处理，且是**字段级合并、在构建执行时解析**——与两条配置「谁后写入」毫无
关系：用例只覆写它显式给出的字段，其余字段继承套件；用例把 ``max_retries``
显式设为 ``0`` 即可在套件默认开启重试时单独关闭本用例的重试。

防时间膨胀
----------
- ``max_retries`` 有硬上限 :data:`MAX_MAX_RETRIES`，配错也不会无限重跑；
- 单轮仍受用例自身 ``timeout`` 约束，因此单条用例总耗时最坏为
  ``(1 + max_retries) × timeout + 退避之和``，上界确定；
- 可选 ``retry_budget_ms`` 给整个重试流程设总预算，预算用尽立即停止；
- 重试**不额外占用线程池名额**：就在用例自己的 worker 线程内串行重跑，
  不会因重试把套件并发度成倍放大；
- 退避睡眠分片进行，构建取消时立即中止，不会傻等。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

# 首轮失败后允许重试的结果状态
RETRYABLE_STATUSES = ("failed", "error", "timeout")

# max_retries 的硬上限：防止配置失误让执行时间成倍膨胀
MAX_MAX_RETRIES = 5

# 策略字段与其默认值（系统默认 = 不重试）
_POLICY_FIELDS = (
    "max_retries", "retry_on_status", "retry_on_assertions", "env_ids",
    "backoff_ms", "backoff_factor", "max_backoff_ms", "retry_budget_ms",
)

_STATUS_LABELS = {
    "passed": "通过", "failed": "断言失败", "error": "执行错误",
    "timeout": "超时", "skipped": "跳过",
}


class RetryConfigError(ValueError):
    """重试策略配置非法。"""


# ---------------------------------------------------------------------------
# 策略规范化与优先级合并
# ---------------------------------------------------------------------------

def _as_int(value: Any, field: str, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        raise RetryConfigError(f"{field} 必须是数字")
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise RetryConfigError(f"{field} 必须是数字")
    if n < lo or n > hi:
        raise RetryConfigError(f"{field} 必须在 {lo}~{hi} 之间")
    return n


def _as_number(value: Any, field: str, lo: float, hi: float) -> float:
    if isinstance(value, bool):
        raise RetryConfigError(f"{field} 必须是数字")
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise RetryConfigError(f"{field} 必须是数字")
    if n < lo or n > hi:
        raise RetryConfigError(f"{field} 必须在 {lo}~{hi} 之间")
    return n


def _as_str_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RetryConfigError(f"{field} 必须是数组")
    return [str(v) for v in value]


def normalize_policy(raw: Optional[dict]) -> dict:
    """把外部传入的策略配置规范化成完整字段，非法值抛 :class:`RetryConfigError`。"""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise RetryConfigError("retry_policy 必须是对象")

    max_retries = _as_int(raw.get("max_retries", 0), "max_retries", 0, MAX_MAX_RETRIES)

    statuses = _as_str_list(raw.get("retry_on_status"), "retry_on_status")
    if not statuses:
        statuses = list(RETRYABLE_STATUSES)
    bad = [s for s in statuses if s not in RETRYABLE_STATUSES]
    if bad:
        raise RetryConfigError(f"retry_on_status 含不支持的状态: {', '.join(bad)}")

    assertions = _as_str_list(raw.get("retry_on_assertions"), "retry_on_assertions")
    env_ids = _as_str_list(raw.get("env_ids"), "env_ids")

    backoff_ms = _as_number(raw.get("backoff_ms", 200), "backoff_ms", 0, 60_000)
    backoff_factor = _as_number(raw.get("backoff_factor", 1.5), "backoff_factor", 1.0, 10.0)
    max_backoff_ms = _as_number(raw.get("max_backoff_ms", 2000), "max_backoff_ms", 0, 600_000)
    retry_budget_ms = _as_number(raw.get("retry_budget_ms", 0), "retry_budget_ms", 0, 3_600_000)

    return {
        "max_retries": max_retries,
        "retry_on_status": statuses,
        "retry_on_assertions": assertions,
        "env_ids": env_ids,
        "backoff_ms": backoff_ms,
        "backoff_factor": backoff_factor,
        "max_backoff_ms": max_backoff_ms,
        "retry_budget_ms": retry_budget_ms,
    }


def merge_policy(suite_policy: Optional[dict], case_policy: Optional[dict]) -> dict:
    """合并套件默认策略与用例单独策略，返回最终生效的完整策略。

    规则是**字段级覆盖、执行时解析**，与配置写入顺序无关：

    - 基线 = 规范化后的套件策略（无套件策略时基线为「不重试」）；
    - 用例策略中**显式出现**的字段逐个覆盖基线，其余字段继承套件；
    - 因此用例 ``{"max_retries": 0}`` 能在套件默认重试 2 次时单独关闭，
      而 ``{}`` 表示「完全跟随套件」。
    """
    merged = normalize_policy(suite_policy)
    case_policy = case_policy or {}
    if not isinstance(case_policy, dict) or not case_policy:
        return merged
    case_norm = normalize_policy(case_policy)
    for key in _POLICY_FIELDS:
        if key in case_policy:
            merged[key] = case_norm[key]
    return merged


def describe_policy(policy: Optional[dict]) -> str:
    """一句话描述策略，供日志展示。"""
    p = normalize_policy(policy)
    if p["max_retries"] <= 0:
        return "不重试"
    parts = [f"最多重试 {p['max_retries']} 次",
             "条件: " + "/".join(p["retry_on_status"])]
    if p["retry_on_assertions"]:
        parts.append("断言 " + "/".join(p["retry_on_assertions"]))
    if p["env_ids"]:
        parts.append(f"限定 {len(p['env_ids'])} 个环境")
    parts.append(f"退避 {p['backoff_ms']:g}ms")
    return "，".join(parts)


# ---------------------------------------------------------------------------
# 重试条件判定
# ---------------------------------------------------------------------------

def _failed_assertion_types(result: dict) -> list[str]:
    return [a.get("type") for a in result.get("assertions", []) if not a.get("ok")]


def should_retry(policy: dict, result: dict, attempts_done: int,
                 env_id: Optional[str] = None, elapsed: float = 0.0,
                 cancel_event=None) -> tuple[bool, str]:
    """判断一轮失败后是否应当重试。

    :param attempts_done: 截至目前已执行的次数（首轮刚结束时为 1）
    :param elapsed: 自首轮开始累计耗时（秒）
    :returns: ``(是否重试, 原因说明)``；不重试时原因用于日志
    """
    p = normalize_policy(policy)
    if cancel_event is not None and cancel_event.is_set():
        return False, "构建已取消"
    if attempts_done > p["max_retries"]:
        return False, f"已达最大重试次数 {p['max_retries']}"
    status = result.get("status", "error")
    if status not in p["retry_on_status"]:
        return False, f"结果状态 {status} 不在重试条件 {p['retry_on_status']} 内"
    if p["retry_on_assertions"]:
        failed_types = set(_failed_assertion_types(result))
        if not (failed_types & set(p["retry_on_assertions"])):
            return False, ("失败断言类型 "
                           f"{sorted(failed_types) or '无'} 不在 "
                           f"{p['retry_on_assertions']} 内")
    if p["env_ids"] and env_id not in p["env_ids"]:
        return False, f"当前环境 {env_id} 不在重试环境 {p['env_ids']} 内"
    if p["retry_budget_ms"] and elapsed * 1000 >= p["retry_budget_ms"]:
        return False, f"重试时间预算 {p['retry_budget_ms']:g}ms 已用尽"
    types = _failed_assertion_types(result)
    reason = f"{_STATUS_LABELS.get(status, status)}"
    if types:
        reason += f"（断言 {'/'.join(sorted(set(types)))} 失败）"
    return True, reason


def backoff_seconds(policy: dict, attempt_index: int) -> float:
    """第 ``attempt_index`` 次重试（0 基）前的退避秒数，指数递增并封顶。"""
    p = normalize_policy(policy)
    raw = p["backoff_ms"] * (p["backoff_factor"] ** attempt_index)
    capped = min(raw, p["max_backoff_ms"]) if p["max_backoff_ms"] else raw
    return round(capped / 1000.0, 3)


def _interruptible_wait(seconds: float, started: float, policy: dict,
                        cancel_event=None) -> bool:
    """分片睡眠，及时响应取消与时间预算；返回 False 表示提前中止。"""
    if seconds <= 0:
        return True
    deadline = time.monotonic() + seconds
    budget_ms = normalize_policy(policy)["retry_budget_ms"]
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return False
        if budget_ms and (time.monotonic() - started) * 1000 >= budget_ms:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(0.05, remaining))


# ---------------------------------------------------------------------------
# 重试执行
# ---------------------------------------------------------------------------

def _round_view(index: int, result: dict) -> dict:
    """保留一整轮执行结果（步骤 / 断言 / 日志全保留）。"""
    return {
        "attempt": index + 1,
        "status": result.get("status", "error"),
        "duration": result.get("duration", 0.0),
        "message": result.get("message", ""),
        "steps": result.get("steps", []),
        "assertions": result.get("assertions", []),
        "logs": result.get("logs", []),
    }


def run_with_retry(execute: Callable[[int], dict], policy: Optional[dict],
                   env_id: Optional[str] = None,
                   cancel_event=None) -> dict:
    """按策略执行一个用例（含重试），返回聚合后的最终结果。

    :param execute: 回调，入参为 0 基的执行序号（首轮 0），返回单轮结果。
                    每轮都是一次全新执行（变量表重新初始化），等价于手动重跑。

    聚合结果在**最后一轮结果**上补充：

    - ``status``         最终一轮的状态 —— 用例结论、通过率、构建红绿都以此为准
    - ``first_status``   首轮状态
    - ``attempts``       总执行次数（1 = 没有重试过）
    - ``retry_count``    实际重试次数
    - ``flaky``          首轮失败、重试后转绿
    - ``rounds``         每一轮的完整结果（步骤 / 断言 / 日志）
    - ``duration``       含退避在内的总耗时
    """
    p = normalize_policy(policy)
    started = time.monotonic()
    rounds: list[dict] = []
    attempt = 0

    while True:
        result = execute(attempt)
        rounds.append(result)
        status = result.get("status", "error")

        # 通过 / 跳过不需要重试
        if status in ("passed", "skipped"):
            break

        elapsed = time.monotonic() - started
        can, reason = should_retry(p, result, attempts_done=attempt + 1,
                                   env_id=env_id, elapsed=elapsed,
                                   cancel_event=cancel_event)
        if not can:
            # 仅在「本可重试」时留一行说明，避免未开启重试的普通失败产生噪音
            if p["max_retries"] > 0:
                result.setdefault("logs", []).append(f"不再重试：{reason}")
            break

        delay = backoff_seconds(p, attempt)
        result.setdefault("logs", []).append(
            f"第 {attempt + 1} 次执行{_STATUS_LABELS.get(status, status)}，"
            f"{delay * 1000:g}ms 后进行第 {attempt + 2} 次执行"
            f"（重试 {attempt + 1}/{p['max_retries']}）：{reason}")

        waited = _interruptible_wait(delay, started, p, cancel_event)
        if not waited:
            if cancel_event is not None and cancel_event.is_set():
                result.setdefault("logs", []).append("构建被取消，停止重试")
            else:
                result.setdefault("logs", []).append(
                    f"重试时间预算 {p['retry_budget_ms']:g}ms 已用尽，停止重试")
            break
        attempt += 1

    return _aggregate(rounds, started)


def _aggregate(rounds: list[dict], started: float) -> dict:
    final = dict(rounds[-1])
    first = rounds[0]
    retry_count = len(rounds) - 1
    first_status = first.get("status", "error")
    final_status = final.get("status", "error")
    flaky = first_status in RETRYABLE_STATUSES and final_status == "passed"

    final["attempts"] = len(rounds)
    final["retry_count"] = retry_count
    final["retried"] = retry_count > 0
    final["first_status"] = first_status
    final["flaky"] = flaky
    final["rounds"] = [_round_view(i, r) for i, r in enumerate(rounds)]
    final["duration"] = round(time.monotonic() - started, 3)

    # 合并各轮日志（构建实时日志 / 单用例日志都会用到）
    combined: list[str] = []
    for i, r in enumerate(rounds):
        combined.append(f"──── 第 {i + 1}/{len(rounds)} 次执行 "
                        f"[{_STATUS_LABELS.get(r.get('status', 'error'), r.get('status'))}] ────")
        combined.extend(r.get("logs", []))
    final["logs"] = combined

    if flaky:
        final["message"] = (f"首轮{_STATUS_LABELS.get(first_status, first_status)}"
                            f"，第 {len(rounds)} 次执行通过（共重试 {retry_count} 次）")
    elif retry_count and final_status in RETRYABLE_STATUSES:
        final["message"] = (f"重试 {retry_count} 次后仍"
                            f"{_STATUS_LABELS.get(final_status, final_status)}")
    return final
