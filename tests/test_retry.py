"""自动重试测试。

覆盖：策略解析优先级（用例 > 套件 > 平台默认）、重试条件判定、执行器的
重试编排（逐轮结果保留 / flaky 标记 / 耗时预算 / 退避）、调度器端到端
（构建红绿按最终结论、报告首轮通过率、缺陷自动闭环）。
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (CoverageAnalyzer, DefectManager, EnvironmentManager,
                    NotificationManager, ReportGenerator, Scheduler,
                    TestExecutor)
from engine.executor import _seeded_int
from engine.retry import (MAX_ATTEMPTS_LIMIT, normalize_policy, resolve_policy,
                          should_retry)
from storage import BuildStoreRegistry, StoreRegistry


def _policy(**kw):
    base = {"max_attempts": 3, "retry_on": ["failed", "error", "timeout"],
            "only_assertion_types": [], "only_envs": [],
            "backoff_ms": 0, "max_extra_duration": 0}
    base.update(kw)
    return base


class TestNormalizePolicy(unittest.TestCase):
    def test_clamps_attempts_to_hard_limit(self):
        p = normalize_policy({"max_attempts": 99})
        self.assertEqual(p["max_attempts"], MAX_ATTEMPTS_LIMIT)
        p = normalize_policy({"max_attempts": 0})
        self.assertEqual(p["max_attempts"], 1)

    def test_filters_invalid_retry_on(self):
        p = normalize_policy({"retry_on": ["failed", "bogus", "error"]})
        self.assertEqual(p["retry_on"], ["failed", "error"])

    def test_empty_retry_on_means_never(self):
        p = normalize_policy({"max_attempts": 3, "retry_on": []})
        self.assertEqual(p["retry_on"], [])

    def test_non_dict_returns_none(self):
        self.assertIsNone(normalize_policy("x"))
        self.assertIsNone(normalize_policy(None))


class TestResolvePolicy(unittest.TestCase):
    def test_neither_configured_returns_none(self):
        self.assertIsNone(resolve_policy(None, None))

    def test_suite_only(self):
        p = resolve_policy(None, {"max_attempts": 2})
        self.assertEqual(p["max_attempts"], 2)

    def test_case_overrides_suite_field_level(self):
        """冲突字段用例优先；用例没写的字段继承套件（与配置顺序无关）。"""
        p = resolve_policy({"max_attempts": 5},
                           {"max_attempts": 2, "backoff_ms": 300})
        self.assertEqual(p["max_attempts"], 5)      # 用例覆盖
        self.assertEqual(p["backoff_ms"], 300)      # 继承套件

    def test_case_can_disable_suite_retry(self):
        """用例显式 max_attempts=1 → 屏蔽套件下发的重试。"""
        p = resolve_policy({"max_attempts": 1}, {"max_attempts": 4})
        self.assertEqual(p["max_attempts"], 1)


class TestShouldRetry(unittest.TestCase):
    def test_no_policy(self):
        self.assertFalse(should_retry(None, {"status": "failed"}, "e1", 1, 0))

    def test_attempts_exhausted(self):
        self.assertFalse(should_retry(_policy(max_attempts=2),
                                      {"status": "failed"}, "e1", 2, 0))

    def test_status_not_retryable(self):
        self.assertFalse(should_retry(_policy(), {"status": "passed"}, "e1", 1, 0))
        p = _policy(retry_on=["error"])
        self.assertFalse(should_retry(p, {"status": "failed"}, "e1", 1, 0))

    def test_env_restriction(self):
        p = _policy(only_envs=["staging"])
        r = {"status": "failed", "assertions": []}
        self.assertFalse(should_retry(p, r, "dev", 1, 0))
        self.assertTrue(should_retry(p, r, "staging", 1, 0))

    def test_assertion_type_restriction(self):
        p = _policy(only_assertion_types=["status"])
        hit = {"status": "failed",
               "assertions": [{"ok": False, "type": "status"}]}
        miss = {"status": "failed",
                "assertions": [{"ok": False, "type": "equals"}]}
        self.assertTrue(should_retry(p, hit, None, 1, 0))
        self.assertFalse(should_retry(p, miss, None, 1, 0))
        # error / timeout 不是断言失败，限定断言类型时不重试
        self.assertFalse(should_retry(p, {"status": "error"}, None, 1, 0))

    def test_extra_duration_budget(self):
        p = _policy(max_extra_duration=10)
        self.assertTrue(should_retry(p, {"status": "failed"}, None, 1, 3))
        self.assertFalse(should_retry(p, {"status": "failed"}, None, 1, 10))


class _ScriptedExecutor(TestExecutor):
    """按预设序列返回每轮状态的执行器（隔离重试编排逻辑做确定性测试）。"""

    def __init__(self, outcomes):
        super().__init__()
        self._outcomes = list(outcomes)
        self.calls = 0

    def execute_case(self, case, env_config=None, cancel_event=None,
                     timeout=None, attempt=0):
        self.calls += 1
        status = self._outcomes[min(self.calls - 1, len(self._outcomes) - 1)]
        return {
            "case_id": case.get("id"), "case_name": case.get("name"),
            "status": status, "duration": 0.01, "steps": [],
            "assertions": ([{"ok": False, "type": "status", "name": "a",
                             "message": "期望 200"}]
                           if status == "failed" else []),
            "logs": [f"round {self.calls}: {status}"], "message": "",
        }


class TestExecuteWithRetry(unittest.TestCase):
    CASE = {"id": "c1", "name": "偶发接口", "timeout": 5}

    def test_first_attempt_pass_no_retry(self):
        ex = _ScriptedExecutor(["passed"])
        r = ex.execute_with_retry(self.CASE, {}, policy=_policy())
        self.assertEqual(r["status"], "passed")
        self.assertEqual(r["attempt_count"], 1)
        self.assertFalse(r["retried"])
        self.assertFalse(r["flaky"])
        self.assertEqual(ex.calls, 1)

    def test_flaky_recovers_and_keeps_every_attempt(self):
        ex = _ScriptedExecutor(["failed", "failed", "passed"])
        r = ex.execute_with_retry(self.CASE, {}, policy=_policy())
        self.assertEqual(r["status"], "passed")          # 最终结论 = 最后一轮
        self.assertEqual(r["attempt_count"], 3)
        self.assertTrue(r["retried"])
        self.assertTrue(r["flaky"])                      # 首轮失败、重试后转绿
        self.assertEqual(r["first_status"], "failed")
        self.assertEqual([a["status"] for a in r["attempts"]],
                         ["failed", "failed", "passed"])  # 每一轮都保留
        self.assertIn("3 次尝试", r["message"])

    def test_persistent_failure_uses_all_attempts(self):
        ex = _ScriptedExecutor(["failed"])
        r = ex.execute_with_retry(self.CASE, {}, policy=_policy(max_attempts=4))
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["attempt_count"], 4)
        self.assertFalse(r["flaky"])

    def test_env_condition_blocks_retry(self):
        ex = _ScriptedExecutor(["failed", "passed"])
        p = _policy(only_envs=["staging"])
        r = ex.execute_with_retry(self.CASE, {}, policy=p, env_id="dev")
        self.assertEqual(r["attempt_count"], 1)
        self.assertEqual(r["status"], "failed")

    def test_assertion_type_condition(self):
        ex = _ScriptedExecutor(["failed", "passed"])
        p = _policy(only_assertion_types=["equals"])  # 失败的是 status 断言
        r = ex.execute_with_retry(self.CASE, {}, policy=p)
        self.assertEqual(r["attempt_count"], 1)
        ex2 = _ScriptedExecutor(["failed", "passed"])
        p2 = _policy(only_assertion_types=["status"])
        r2 = ex2.execute_with_retry(self.CASE, {}, policy=p2)
        self.assertEqual(r2["attempt_count"], 2)
        self.assertEqual(r2["status"], "passed")

    def test_budget_stops_retry(self):
        """额外耗时预算耗尽后停止重试，防止构建时长成倍膨胀。"""
        ex = _ScriptedExecutor(["failed"])
        p = _policy(max_attempts=5, max_extra_duration=0.005)
        r = ex.execute_with_retry(self.CASE, {}, policy=p)
        self.assertLess(r["attempt_count"], 5)
        self.assertGreaterEqual(r["attempt_count"], 2)  # 至少补了一轮

    def test_backoff_waits_between_attempts(self):
        ex = _ScriptedExecutor(["failed", "passed"])
        p = _policy(backoff_ms=200)
        started = time.time()
        r = ex.execute_with_retry(self.CASE, {}, policy=p)
        self.assertEqual(r["status"], "passed")
        self.assertGreaterEqual(time.time() - started, 0.2)
        self.assertGreaterEqual(r["duration"], 0.2)  # 总耗时含退避

    def test_no_policy_single_attempt(self):
        ex = _ScriptedExecutor(["failed"])
        r = ex.execute_with_retry(self.CASE, {}, policy=None)
        self.assertEqual(r["attempt_count"], 1)
        self.assertEqual(r["status"], "failed")


def _find_flaky_case_id(fail_rate=0.5):
    """找一个「首轮被注入失败、第 2 轮不再失败」的用例 id（确定性）。"""
    for i in range(2000):
        cid = f"flaky_{i}"
        first = _seeded_int(cid, 0, "fail", 0) % 1000 < fail_rate * 1000
        second = _seeded_int(cid, 0, "fail", 1) % 1000 < fail_rate * 1000
        if first and not second:
            return cid
    raise AssertionError("找不到满足条件的用例 id")


def _find_stable_case_id(fail_rate=0.5, max_attempts=5):
    """找一个「各轮都不会被注入失败」的用例 id（确定性）。"""
    for i in range(2000):
        cid = f"stable_{i}"
        if all(_seeded_int(cid, 0, "fail", a) % 1000 >= fail_rate * 1000
               for a in range(max_attempts)):
            return cid
    raise AssertionError("找不到满足条件的用例 id")


def _make_scheduler(data_root):
    registry = StoreRegistry(os.path.join(data_root, "store"), shard_size=50)
    builds = BuildStoreRegistry(os.path.join(data_root, "builds"))
    env_mgr = EnvironmentManager(registry, data_root)
    sched = Scheduler(registry, builds, TestExecutor(), env_mgr,
                      ReportGenerator(builds), CoverageAnalyzer(builds),
                      DefectManager(registry), NotificationManager(registry),
                      max_build_workers=2, max_case_workers=4, tick_seconds=0.2)
    return registry, builds, env_mgr, sched


def _wait_build(builds, pid, build_id, sched, timeout=20):
    """等构建到终态，并且等 _finalize（报告/缺陷收尾）也完成。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = builds.for_project(pid).get(build_id)
        if b and b["status"] in ("passed", "failed", "cancelled", "error"):
            with sched._running_lock:
                if build_id not in sched._running:
                    return b
        time.sleep(0.05)
    raise AssertionError("构建超时未结束")


class TestSchedulerRetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry, self.builds, self.env_mgr, self.sched = _make_scheduler(self.tmp.name)
        self.pid = self.registry.store("projects").insert(
            {"name": "P", "auto_create_defects": True})
        self.env = self.env_mgr.create(self.pid, {
            "name": "flaky-env",
            "config": {"latency_ms": 0, "fail_rate": 0.5}})
        self.cases = self.registry.store("cases")

    def tearDown(self):
        # 等后台构建线程收尾完毕再清理临时目录，避免与落盘线程竞争
        deadline = time.time() + 10
        while time.time() < deadline:
            with self.sched._running_lock:
                if not self.sched._running:
                    break
            time.sleep(0.05)
        self.sched.shutdown()
        self.tmp.cleanup()

    def _add_case(self, case_id, url="/api/health", retry=None):
        case = {
            "id": case_id, "project_id": self.pid, "name": case_id,
            "priority": "P1", "tags": ["g"], "timeout": 30, "enabled": True,
            "steps": [
                {"action": "request", "method": "GET", "url": url},
                {"action": "assert", "type": "status",
                 "actual": "${resp.status}", "expected": 200},
            ],
        }
        if retry is not None:
            case["retry"] = retry
        self.cases.insert(case)
        return case_id

    def _add_suite(self, suite_id, case_ids, retry=None):
        suite = {"id": suite_id, "project_id": self.pid, "name": suite_id,
                 "env_id": self.env["id"], "case_ids": case_ids}
        if retry is not None:
            suite["retry"] = retry
        self.registry.store("suites").insert(suite)

    def test_flaky_case_turns_build_green_and_marks_attempts(self):
        """偶发失败用例：首轮挂、重试转绿 → 构建绿、flaky 计数、逐轮结果保留。"""
        flaky_id = _find_flaky_case_id()
        stable_id = _find_stable_case_id()
        self._add_case(flaky_id)
        self._add_case(stable_id)
        self._add_suite("suite_flaky", [flaky_id, stable_id],
                        retry={"max_attempts": 3, "backoff_ms": 10})

        bid = self.sched.submit_build(self.pid, "suite_flaky")["id"]
        build = _wait_build(self.builds, self.pid, bid, self.sched)

        self.assertEqual(build["status"], "passed")   # 红绿按最终结论
        self.assertEqual(build["passed"], 2)
        self.assertEqual(build["flaky"], 1)
        self.assertEqual(build["retried"], 1)

        results = {r["case_id"]: r
                   for r in self.builds.for_project(self.pid).results(bid)}
        flaky = results[flaky_id]
        self.assertEqual(flaky["status"], "passed")
        self.assertEqual(flaky["attempt_count"], 2)
        self.assertEqual(flaky["first_status"], "failed")
        self.assertEqual([a["status"] for a in flaky["attempts"]],
                         ["failed", "passed"])
        self.assertEqual(results[stable_id]["attempt_count"], 1)

        # 报告：首轮通过率 < 最终通过率，flaky 清单标注重试次数
        report = ReportGenerator(self.builds).build_report(self.pid, bid)
        self.assertEqual(report["summary"]["pass_rate"], 100.0)
        self.assertEqual(report["summary"]["first_pass_rate"], 50.0)
        self.assertEqual(report["summary"]["flaky"], 1)
        self.assertEqual(len(report["flaky_cases"]), 1)
        self.assertEqual(report["flaky_cases"][0]["attempt_count"], 2)

    def test_case_policy_overrides_suite_policy(self):
        """用例显式 max_attempts=1 屏蔽套件重试（明确优先级）。"""
        flaky_id = _find_flaky_case_id()
        self._add_case(flaky_id, retry={"max_attempts": 1})
        self._add_suite("suite_prio", [flaky_id],
                        retry={"max_attempts": 3, "backoff_ms": 1})

        bid = self.sched.submit_build(self.pid, "suite_prio")["id"]
        build = _wait_build(self.builds, self.pid, bid, self.sched)

        self.assertEqual(build["status"], "failed")  # 不重试 → 首轮失败即结论
        results = self.builds.for_project(self.pid).results(bid)
        self.assertEqual(results[0]["attempt_count"], 1)

    def test_persistent_failure_creates_defect_then_auto_resolves(self):
        """重试后仍失败 → 建缺陷；后续构建该用例通过 → 缺陷自动闭环。"""
        self._add_case("always_bad", url="/api/error")
        self._add_suite("suite_def", ["always_bad"],
                       retry={"max_attempts": 2, "backoff_ms": 1})

        bid1 = self.sched.submit_build(self.pid, "suite_def")["id"]
        build1 = _wait_build(self.builds, self.pid, bid1, self.sched)
        self.assertEqual(build1["status"], "failed")

        defects = DefectManager(self.registry)
        ds = defects.list(self.pid)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["status"], "open")
        self.assertTrue(ds[0]["auto"])
        self.assertEqual(ds[0]["source_case_id"], "always_bad")

        # 接口修复后重跑：用例通过 → 缺陷自动 fixed，与报告结论一致
        self.cases.update("always_bad", {
            "steps": [
                {"action": "request", "method": "GET", "url": "/api/health"},
                {"action": "assert", "type": "status",
                 "actual": "${resp.status}", "expected": 200},
            ],
        })
        bid2 = self.sched.submit_build(self.pid, "suite_def")["id"]
        build2 = _wait_build(self.builds, self.pid, bid2, self.sched)
        self.assertEqual(build2["status"], "passed")

        d = defects.get(ds[0]["id"])
        self.assertEqual(d["status"], "fixed")
        self.assertEqual(d["resolved_build_id"], bid2)

    def test_flaky_case_does_not_create_defect(self):
        """重试后转绿的用例：报告说它过了，缺陷列表也不会挂着它。"""
        flaky_id = _find_flaky_case_id()
        self._add_case(flaky_id)
        self._add_suite("suite_nodefect", [flaky_id],
                        retry={"max_attempts": 3, "backoff_ms": 1})
        bid = self.sched.submit_build(self.pid, "suite_nodefect")["id"]
        _wait_build(self.builds, self.pid, bid, self.sched)
        self.assertEqual(DefectManager(self.registry).list(self.pid), [])

    def test_retry_does_not_explode_build_duration(self):
        """次数硬上限 + 预算共同约束：重试轮次有界。"""
        self._add_case("always_bad2", url="/api/error")
        self._add_suite("suite_cap", ["always_bad2"],
                        retry={"max_attempts": 99, "backoff_ms": 1,
                               "max_extra_duration": 5})
        bid = self.sched.submit_build(self.pid, "suite_cap")["id"]
        _wait_build(self.builds, self.pid, bid, self.sched)
        results = self.builds.for_project(self.pid).results(bid)
        self.assertLessEqual(results[0]["attempt_count"], MAX_ATTEMPTS_LIMIT)


if __name__ == "__main__":
    unittest.main()
