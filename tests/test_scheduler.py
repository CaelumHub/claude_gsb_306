"""调度器集成测试。

覆盖：并发调度（构建池 + 用例池）、结果收集与聚合、报告/覆盖率/通知收尾、
取消、以及定时任务的触发去重。
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (CoverageAnalyzer, DefectManager, EnvironmentManager,
                    NotificationManager, ReportGenerator, Scheduler, TestExecutor)
from storage import BuildStoreRegistry, StoreRegistry


def _make_scheduler(data_root):
    registry = StoreRegistry(os.path.join(data_root, "store"), shard_size=50)
    builds = BuildStoreRegistry(os.path.join(data_root, "builds"))
    executor = TestExecutor()
    env_mgr = EnvironmentManager(registry, data_root)
    coverage = CoverageAnalyzer(builds)
    report = ReportGenerator(builds)
    defects = DefectManager(registry)
    notify = NotificationManager(registry)
    sched = Scheduler(registry, builds, executor, env_mgr, report, coverage,
                      defects, notify, max_build_workers=2, max_case_workers=4,
                      tick_seconds=0.2)
    return registry, builds, env_mgr, sched


class TestSchedulerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry, self.builds, self.env_mgr, self.sched = _make_scheduler(self.tmp.name)

    def tearDown(self):
        self.sched.shutdown()
        self.tmp.cleanup()

    def _setup_project(self, n_cases=12):
        pid = self.registry.store("projects").insert({"name": "P"})
        env = self.env_mgr.create(pid, {"name": "dev", "config": {"latency_ms": 0, "fail_rate": 0.0}})
        cases_store = self.registry.store("cases")
        ids = []
        for i in range(n_cases):
            ids.append(cases_store.insert({
                "id": f"case_{i}", "project_id": pid, "name": f"用例{i}",
                "priority": "P2", "tags": ["g1" if i % 2 else "g2"], "timeout": 30,
                "steps": [
                    {"action": "request", "method": "GET", "url": "/api/health"},
                    {"action": "assert", "type": "status", "actual": "${resp.status}", "expected": 200},
                ],
            }))
        suite = {
            "id": "suite_1", "project_id": pid, "name": "冒烟",
            "env_id": env["id"], "case_ids": ids,
        }
        self.registry.store("suites").insert(suite)
        return pid, suite

    def test_full_run_collects_results(self):
        pid, suite = self._setup_project(12)
        result = self.sched.submit_build(pid, suite["id"], trigger="manual")
        self.assertIn("id", result)
        build_id = result["id"]

        # 等待构建完成
        deadline = time.time() + 20
        build = None
        while time.time() < deadline:
            build = self.builds.for_project(pid).get(build_id)
            if build and build["status"] in ("passed", "failed", "cancelled", "error"):
                break
            time.sleep(0.05)
        self.assertIsNotNone(build)
        self.assertEqual(build["status"], "passed")
        self.assertEqual(build["passed"], 12)
        self.assertEqual(len(self.builds.for_project(pid).results(build_id)), 12)

        # 报告 / 覆盖率已生成
        self.assertIsNotNone(self.builds.for_project(pid).read_report(build_id))
        self.assertIsNotNone(self.builds.for_project(pid).read_coverage(build_id))

    def test_concurrent_builds(self):
        pid, suite = self._setup_project(20)
        results = [self.sched.submit_build(pid, suite["id"]) for _ in range(3)]
        ids = [r["id"] for r in results]
        deadline = time.time() + 30
        while time.time() < deadline:
            builds = [self.builds.for_project(pid).get(b) for b in ids]
            if all(b and b["status"] in ("passed", "failed", "cancelled", "error") for b in builds):
                break
            time.sleep(0.05)
        for b in self.builds.for_project(pid).list_builds():
            if b["id"] in ids:
                self.assertEqual(b["status"], "passed")
                self.assertEqual(b["passed"], 20)
                self.assertEqual(len(self.builds.for_project(pid).results(b["id"])), 20)

    def test_cancel_build(self):
        pid, suite = self._setup_project(30)
        result = self.sched.submit_build(pid, suite["id"])
        build_id = result["id"]
        self.sched.cancel_build(build_id)
        deadline = time.time() + 15
        while time.time() < deadline:
            b = self.builds.for_project(pid).get(build_id)
            if b and b["status"] in ("cancelled", "passed", "failed"):
                break
            time.sleep(0.05)
        b = self.builds.for_project(pid).get(build_id)
        self.assertIn(b["status"], ("cancelled", "passed", "failed"))

    def test_schedule_fires_once_per_minute(self):
        pid, suite = self._setup_project(3)
        cron = "* * * * *"  # 每分钟
        sch = {
            "id": "sch_1", "project_id": pid, "name": "每分", "cron": cron,
            "suite_id": suite["id"], "env_id": suite["env_id"], "enabled": True,
            "last_fired_minute": None,
        }
        self.registry.store("schedules").insert(sch)
        self.sched.start()
        # 直接调用一次扫描，再立即调用一次，同分钟只应触发一次
        self.sched._scan_schedules()
        self.sched._scan_schedules()
        runs = self.registry.store("schedule_runs").query(where=[("schedule_id", "eq", sch["id"])])
        self.assertEqual(len(runs), 1)


class _FlakyExecutor:
    """前 ``fail_times`` 次（按 case_id 计数）失败、之后通过的假执行器。

    真正的 execute_case 每轮都是无状态重跑，这里模拟「首轮抖、重跑又绿」。
    """

    def __init__(self, fail_map: dict, status: str = "failed",
                 assertion_type: str = "status"):
        self.fail_map = fail_map
        self.calls: dict[str, int] = {}
        self.status = status
        self.assertion_type = assertion_type

    def execute_case(self, case, env_config=None, cancel_event=None, timeout=60):
        cid = case["id"]
        n = self.calls.get(cid, 0)
        self.calls[cid] = n + 1
        if n < self.fail_map.get(cid, 0):
            return {
                "case_id": cid, "case_name": case.get("name", cid),
                "group": (case.get("tags") or ["默认"])[0],
                "priority": case.get("priority", "P2"),
                "status": self.status, "duration": 0.0,
                "steps": [{"index": 0, "status": self.status, "message": "boom"}],
                "assertions": [{"name": "a", "type": self.assertion_type,
                                "ok": False, "message": "期望 == 200",
                                "actual": 500, "expected": 200}],
                "logs": [f"第 {n + 1} 次执行失败"],
            }
        return {
            "case_id": cid, "case_name": case.get("name", cid),
            "group": (case.get("tags") or ["默认"])[0],
            "priority": case.get("priority", "P2"),
            "status": "passed", "duration": 0.0,
            "steps": [{"index": 0, "status": "passed", "message": "ok"}],
            "assertions": [{"name": "a", "type": self.assertion_type,
                            "ok": True, "message": "期望 == 200",
                            "actual": 200, "expected": 200}],
            "logs": [f"第 {n + 1} 次执行通过"],
        }


class TestRetryEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry, self.builds, self.env_mgr, self.sched = _make_scheduler(self.tmp.name)

    def tearDown(self):
        self.sched.shutdown()
        self.tmp.cleanup()

    def _wait(self, pid, build_id, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            build = self.builds.for_project(pid).get(build_id)
            if build and build["status"] in ("passed", "failed", "cancelled", "error"):
                return build
            time.sleep(0.02)
        raise AssertionError("构建未在超时内结束")

    def _make_case(self, pid, cid, retry_policy=None):
        case = {
            "id": cid, "project_id": pid, "name": cid, "priority": "P1",
            "tags": ["g1"], "timeout": 10,
            "steps": [
                {"action": "request", "method": "GET", "url": "/api/health"},
                {"action": "assert", "type": "status",
                 "actual": "${resp.status}", "expected": 200},
            ],
        }
        if retry_policy is not None:
            case["retry_policy"] = retry_policy
        return self.registry.store("cases").insert(case)

    def _make_suite(self, pid, env_id, case_ids, retry_policy=None):
        suite = {"id": "suite_retry", "project_id": pid, "name": "重试套件",
                 "env_id": env_id, "case_ids": case_ids}
        if retry_policy is not None:
            suite["retry_policy"] = retry_policy
        self.registry.store("suites").insert(suite)
        return suite

    def test_flaky_case_green_after_retry_no_defect(self):
        """首轮失败、第 2 次通过：最终通过率 100%、构建绿、不产生缺陷。"""
        pid = self.registry.store("projects").insert(
            {"name": "P", "auto_create_defects": True})
        env = self.env_mgr.create(pid, {"name": "dev",
                                        "config": {"latency_ms": 0}})
        self._make_case(pid, "c_flaky")
        self._make_case(pid, "c_ok")
        suite = self._make_suite(pid, env["id"], ["c_flaky", "c_ok"],
                                 retry_policy={"max_retries": 2, "backoff_ms": 0})
        self.sched.executor = _FlakyExecutor({"c_flaky": 1})

        build = self.sched.submit_build(pid, suite["id"], trigger="manual")
        b = self._wait(pid, build["id"])

        # 构建按最终结果判定为绿
        self.assertEqual(b["status"], "passed")
        self.assertEqual(b["passed"], 2)
        self.assertEqual(b["first_passed"], 1)       # 首轮只有 1 条过
        self.assertEqual(b["flaky"], 1)              # 1 条重试后转绿
        self.assertGreaterEqual(b["retry_count"], 1)

        store = self.builds.for_project(pid)
        results = {r["case_id"]: r for r in store.results(build["id"])}
        flaky = results["c_flaky"]
        self.assertEqual(flaky["status"], "passed")  # 最终结论
        self.assertEqual(flaky["first_status"], "failed")
        self.assertTrue(flaky["flaky"])
        self.assertEqual(flaky["retry_count"], 1)
        self.assertEqual(flaky["attempts"], 2)
        self.assertEqual(len(flaky["rounds"]), 2)    # 每轮结果都保留

        # 最终失败为 0 条 → 不建缺陷（报告与缺陷不矛盾）
        defects = self.registry.store("defects").query(
            where=[("project_id", "eq", pid)])
        self.assertEqual(defects, [])

        # 报告同时给出两种口径
        report = self.sched.report_gen.build_report(pid, build["id"], force=True)
        self.assertEqual(report["summary"]["pass_rate"], 100.0)
        self.assertEqual(report["summary"]["first_pass_rate"], 50.0)
        self.assertEqual(len(report["flaky_cases"]), 1)

    def test_persistent_failure_creates_defect(self):
        """重试 2 次仍失败：构建红、缺陷描述带重试信息。"""
        pid = self.registry.store("projects").insert(
            {"name": "P", "auto_create_defects": True})
        env = self.env_mgr.create(pid, {"name": "dev",
                                        "config": {"latency_ms": 0}})
        self._make_case(pid, "c_bad")
        suite = self._make_suite(pid, env["id"], ["c_bad"],
                                 retry_policy={"max_retries": 2, "backoff_ms": 0})
        self.sched.executor = _FlakyExecutor({"c_bad": 99})

        build = self.sched.submit_build(pid, suite["id"], trigger="manual")
        b = self._wait(pid, build["id"])
        self.assertEqual(b["status"], "failed")
        self.assertEqual(b["passed"], 0)
        self.assertEqual(b["flaky"], 0)
        self.assertEqual(b["retry_count"], 2)

        store = self.builds.for_project(pid)
        result = store.results(build["id"])[0]
        self.assertEqual(result["attempts"], 3)
        self.assertFalse(result["flaky"])

        defects = self.registry.store("defects").query(
            where=[("project_id", "eq", pid)])
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["source_case_id"], "c_bad")
        self.assertIn("重试 2 次", defects[0]["description"])

    def test_case_policy_overrides_suite_default(self):
        """套件默认重试 2 次，用例显式 max_retries=0：抖动也不重试。"""
        pid = self.registry.store("projects").insert({"name": "P"})
        env = self.env_mgr.create(pid, {"name": "dev",
                                        "config": {"latency_ms": 0}})
        self._make_case(pid, "c_off", retry_policy={"max_retries": 0})
        suite = self._make_suite(pid, env["id"], ["c_off"],
                                 retry_policy={"max_retries": 2, "backoff_ms": 0})
        self.sched.executor = _FlakyExecutor({"c_off": 1})

        build = self.sched.submit_build(pid, suite["id"], trigger="manual")
        b = self._wait(pid, build["id"])
        self.assertEqual(b["status"], "failed")
        self.assertEqual(b["retry_count"], 0)       # 用例显式关闭生效
        self.assertEqual(self.sched.executor.calls["c_off"], 1)

    def test_assertion_condition_limits_retry(self):
        """套件策略只在 status 断言失败时重试；equals 失败不重试。"""
        pid = self.registry.store("projects").insert({"name": "P"})
        env = self.env_mgr.create(pid, {"name": "dev",
                                        "config": {"latency_ms": 0}})
        self._make_case(pid, "c_eq")
        suite = self._make_suite(
            pid, env["id"], ["c_eq"],
            retry_policy={"max_retries": 3, "backoff_ms": 0,
                          "retry_on_assertions": ["status"]})
        self.sched.executor = _FlakyExecutor({"c_eq": 1},
                                             assertion_type="equals")
        build = self.sched.submit_build(pid, suite["id"], trigger="manual")
        b = self._wait(pid, build["id"])
        self.assertEqual(b["status"], "failed")
        self.assertEqual(b["retry_count"], 0)
        self.assertEqual(self.sched.executor.calls["c_eq"], 1)


if __name__ == "__main__":
    unittest.main()
