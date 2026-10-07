"""自动重试单元测试。

覆盖：
- 策略规范化与硬上限（防止执行时间成倍膨胀的第一道闸）；
- 套件默认策略与用例单独策略的字段级合并优先级（用例 > 套件 > 默认，
  且与配置先后无关）；
- 重试条件判定：结果状态 / 断言类型 / 环境 / 时间预算 / 取消 / 次数；
- 退避递增与封顶；
- run_with_retry 的聚合语义：最终状态、first_status、flaky、rounds 全保留；
- 重试时间预算用尽立即停止。
"""

from __future__ import annotations

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.retry import (MAX_MAX_RETRIES, RETRYABLE_STATUSES,
                          RetryConfigError, backoff_seconds, merge_policy,
                          normalize_policy, run_with_retry, should_retry)


def _fail(status="failed", assertion_type="status"):
    return {
        "status": status,
        "duration": 0.0,
        "steps": [{"index": 0, "status": status, "message": "boom"}],
        "assertions": ([{"name": "a", "type": assertion_type, "ok": False,
                         "message": "期望 == 200", "actual": 500, "expected": 200}]
                       if assertion_type else []),
        "logs": [],
    }


def _pass():
    return {"status": "passed", "duration": 0.0, "steps": [],
            "assertions": [], "logs": []}


class TestNormalize(unittest.TestCase):
    def test_defaults(self):
        p = normalize_policy(None)
        self.assertEqual(p["max_retries"], 0)
        self.assertEqual(p["retry_on_status"], list(RETRYABLE_STATUSES))
        self.assertEqual(p["retry_on_assertions"], [])
        self.assertEqual(p["env_ids"], [])

    def test_hard_cap(self):
        # 超过硬上限的配置被直接拒绝（在 API 层即 400），不允许静默放大执行时间
        with self.assertRaises(RetryConfigError):
            normalize_policy({"max_retries": 99})
        self.assertEqual(normalize_policy({"max_retries": MAX_MAX_RETRIES})["max_retries"],
                         MAX_MAX_RETRIES)

    def test_invalid(self):
        with self.assertRaises(RetryConfigError):
            normalize_policy({"max_retries": -1})
        with self.assertRaises(RetryConfigError):
            normalize_policy({"max_retries": "abc"})
        with self.assertRaises(RetryConfigError):
            normalize_policy({"retry_on_status": ["passed"]})
        with self.assertRaises(RetryConfigError):
            normalize_policy({"backoff_factor": 0.1})  # 必须 >= 1
        with self.assertRaises(RetryConfigError):
            normalize_policy("not-a-dict")


class TestMergePriority(unittest.TestCase):
    def test_suite_default_used_when_case_omits(self):
        suite = {"max_retries": 3, "backoff_ms": 500}
        merged = merge_policy(suite, None)
        self.assertEqual(merged["max_retries"], 3)
        self.assertEqual(merged["backoff_ms"], 500)

    def test_case_overrides_field_by_field(self):
        suite = {"max_retries": 3, "backoff_ms": 500,
                 "retry_on_status": ["failed"]}
        # 用例只显式给出 max_retries，其余字段继承套件
        merged = merge_policy(suite, {"max_retries": 1})
        self.assertEqual(merged["max_retries"], 1)
        self.assertEqual(merged["backoff_ms"], 500)
        self.assertEqual(merged["retry_on_status"], ["failed"])

    def test_case_can_disable_retry_explicitly(self):
        suite = {"max_retries": 3}
        merged = merge_policy(suite, {"max_retries": 0})
        self.assertEqual(merged["max_retries"], 0)

    def test_empty_case_policy_follows_suite(self):
        suite = {"max_retries": 2}
        self.assertEqual(merge_policy(suite, {})["max_retries"], 2)

    def test_priority_independent_of_configuration_order(self):
        # 无论先套件后用例还是反过来，合并结果都一样（纯函数、执行时解析）
        suite = {"max_retries": 2}
        case = {"max_retries": 5}
        m1 = merge_policy(suite, case)
        m2 = merge_policy(dict(suite), dict(case))
        self.assertEqual(m1, m2)
        self.assertEqual(m1["max_retries"], 5)  # 用例胜出

    def test_no_suite_no_retry(self):
        self.assertEqual(merge_policy(None, None)["max_retries"], 0)


class TestShouldRetry(unittest.TestCase):
    def test_max_attempts(self):
        p = normalize_policy({"max_retries": 2})
        ok, _ = should_retry(p, _fail(), attempts_done=1)
        self.assertTrue(ok)
        ok, _ = should_retry(p, _fail(), attempts_done=2)
        self.assertTrue(ok)
        ok, reason = should_retry(p, _fail(), attempts_done=3)
        self.assertFalse(ok)
        self.assertIn("最大重试次数", reason)

    def test_status_filter(self):
        p = normalize_policy({"max_retries": 1, "retry_on_status": ["timeout"]})
        ok, _ = should_retry(p, _fail("failed"), attempts_done=1)
        self.assertFalse(ok)
        ok, _ = should_retry(p, _fail("timeout"), attempts_done=1)
        self.assertTrue(ok)

    def test_assertion_type_filter(self):
        p = normalize_policy({"max_retries": 1,
                              "retry_on_assertions": ["status"]})
        ok, _ = should_retry(p, _fail("failed", "status"), attempts_done=1)
        self.assertTrue(ok)
        ok, reason = should_retry(p, _fail("failed", "equals"), attempts_done=1)
        self.assertFalse(ok)
        self.assertIn("断言类型", reason)
        # 错误发生在断言之前（没有失败断言）也不重试
        ok, _ = should_retry(p, _fail("error", None), attempts_done=1)
        self.assertFalse(ok)

    def test_env_filter(self):
        p = normalize_policy({"max_retries": 1, "env_ids": ["env_a"]})
        ok, _ = should_retry(p, _fail(), attempts_done=1, env_id="env_a")
        self.assertTrue(ok)
        ok, _ = should_retry(p, _fail(), attempts_done=1, env_id="env_b")
        self.assertFalse(ok)

    def test_budget(self):
        p = normalize_policy({"max_retries": 3, "retry_budget_ms": 10})
        ok, _ = should_retry(p, _fail(), attempts_done=1, elapsed=0.005)
        self.assertTrue(ok)
        ok, reason = should_retry(p, _fail(), attempts_done=1, elapsed=0.02)
        self.assertFalse(ok)
        self.assertIn("预算", reason)

    def test_cancel(self):
        p = normalize_policy({"max_retries": 3})
        ev = threading.Event()
        ev.set()
        ok, _ = should_retry(p, _fail(), attempts_done=1, cancel_event=ev)
        self.assertFalse(ok)


class TestBackoff(unittest.TestCase):
    def test_exponential_and_cap(self):
        p = normalize_policy({"backoff_ms": 100, "backoff_factor": 2.0,
                              "max_backoff_ms": 500})
        self.assertAlmostEqual(backoff_seconds(p, 0), 0.1)
        self.assertAlmostEqual(backoff_seconds(p, 1), 0.2)
        self.assertAlmostEqual(backoff_seconds(p, 2), 0.4)
        self.assertAlmostEqual(backoff_seconds(p, 3), 0.5)  # 封顶
        self.assertAlmostEqual(backoff_seconds(p, 9), 0.5)


class TestRunWithRetry(unittest.TestCase):
    def test_pass_first_attempt_no_retry(self):
        calls = []
        def execute(i):
            calls.append(i)
            return _pass()
        out = run_with_retry(execute, {"max_retries": 2})
        self.assertEqual(out["status"], "passed")
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(out["retry_count"], 0)
        self.assertFalse(out["flaky"])
        self.assertEqual(len(out["rounds"]), 1)
        self.assertEqual(calls, [0])

    def test_flaky_passes_on_retry(self):
        seq = [_fail(), _fail(), _pass()]
        def execute(i):
            return seq[i]
        out = run_with_retry(execute, {"max_retries": 2, "backoff_ms": 0})
        self.assertEqual(out["status"], "passed")
        self.assertEqual(out["first_status"], "failed")
        self.assertEqual(out["attempts"], 3)
        self.assertEqual(out["retry_count"], 2)
        self.assertTrue(out["flaky"])
        self.assertEqual(len(out["rounds"]), 3)
        # 每一轮的步骤 / 断言都保留
        self.assertEqual(out["rounds"][0]["status"], "failed")
        self.assertFalse(out["rounds"][0]["assertions"][0]["ok"])
        self.assertEqual(out["rounds"][2]["status"], "passed")
        self.assertIn("通过", out["message"])

    def test_still_fails_after_retries(self):
        def execute(i):
            return _fail()
        out = run_with_retry(execute, {"max_retries": 2, "backoff_ms": 0})
        self.assertEqual(out["status"], "failed")
        self.assertFalse(out["flaky"])
        self.assertEqual(out["attempts"], 3)
        self.assertEqual(out["retry_count"], 2)

    def test_assertion_filter_skips_retry(self):
        def execute(i):
            return _fail("failed", "equals")
        p = normalize_policy({"max_retries": 2, "backoff_ms": 0,
                              "retry_on_assertions": ["status"]})
        out = run_with_retry(execute, p)
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(out["retry_count"], 0)

    def test_budget_stops_retrying(self):
        # 预算 1ms，第一次退避（100ms）即被预算打断，停在第 1 轮失败
        calls = []
        def execute(i):
            calls.append(i)
            return _fail()
        p = normalize_policy({"max_retries": 3, "backoff_ms": 100,
                              "retry_budget_ms": 1})
        out = run_with_retry(execute, p)
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(len(calls), 1)

    def test_cancel_stops_retrying(self):
        ev = threading.Event()
        calls = []
        def execute(i):
            calls.append(i)
            if i == 1:
                ev.set()
            return _fail()
        out = run_with_retry(execute, {"max_retries": 3, "backoff_ms": 50},
                             cancel_event=ev)
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["attempts"], 2)
        self.assertEqual(len(calls), 2)

    def test_skipped_not_retried(self):
        def execute(i):
            return {"status": "skipped", "duration": 0.0, "steps": [],
                    "assertions": [], "logs": []}
        out = run_with_retry(execute, {"max_retries": 2})
        self.assertEqual(out["status"], "skipped")
        self.assertEqual(out["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
