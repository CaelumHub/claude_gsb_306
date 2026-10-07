"""测试报告生成：通过率、耗时、分组、失败明细与趋势。

报告生成的性能关注点
--------------------
一次构建可能跑几十上百个用例，若每次打开报告都去扫全部分片结果并重新
聚合，既慢又重复。这里的策略是：

1. **聚合前置**：通过 / 失败计数、分组计数、耗时序列在结果落盘时（
   :meth:`storage.buildstore.BuildStore.record_result`）就增量算好，存进
   轻量的 ``build.json``，报告只读这个汇总，几乎零成本；
2. **缓存**：完整报告在构建结束时生成一次写入 ``report.json``，之后
   请求直接返回缓存，只有结果变化时才重算；
3. **按需明细**：失败明细、最慢用例只在生成报告时扫一次结果，并限定
   top-N，避免全量扫描失控。
"""

from __future__ import annotations

import statistics
import time
from typing import Optional

FAILED_STATUSES = ("failed", "error", "timeout")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 3)


class ReportGenerator:
    """测试报告生成器。"""

    def __init__(self, build_store_registry):
        self.builds = build_store_registry

    # -- 单次构建报告 -----------------------------------------------------
    def build_report(self, project_id: str, build_id: str,
                     force: bool = False) -> dict:
        store = self.builds.for_project(project_id)
        build = store.get(build_id)
        if build is None:
            return {"error": "构建不存在"}

        # 命中缓存（构建已结束且未强制重算）
        cached = store.read_report(build_id)
        if cached and not force:
            return cached

        report = self._compute(store, build_id, build)
        if build.get("status") not in ("running", "pending"):
            store.write_report(build_id, report)
        return report

    def _compute(self, store, build_id: str, build: dict) -> dict:
        total = build.get("total", 0)
        passed = build.get("passed", 0)
        failed = build.get("failed", 0)
        error = build.get("error", 0)
        skipped = build.get("skipped", 0)
        timeout = build.get("timeout", 0)
        finished = total - skipped
        # 通过率按「最终结论」算（重试后转绿计为通过）——它决定构建红绿；
        # 首轮通过率单独给出，作为套件稳定性的真实参照（暴露抖动程度）。
        pass_rate = round(passed / finished * 100, 1) if finished else 0.0
        first_passed = build.get("first_passed")
        if first_passed is None:
            first_passed = passed  # 兼容重试功能上线前的旧构建
        first_pass_rate = round(first_passed / finished * 100, 1) if finished else 0.0

        durations = build.get("durations", [])
        report = {
            "build_id": build_id,
            "project_id": build.get("project_id"),
            "name": build.get("name") or build_id,
            "status": build.get("status"),
            "trigger": build.get("trigger"),
            "suite_id": build.get("suite_id"),
            "env_id": build.get("env_id"),
            "started_at": build.get("started_at"),
            "finished_at": build.get("finished_at"),
            "duration": build.get("duration", 0.0),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "error": error,
                "skipped": skipped,
                "timeout": timeout,
                "pass_rate": pass_rate,
                "first_pass_rate": first_pass_rate,
                "retried": build.get("retried", 0),
                "flaky": build.get("flaky", 0),
            },
            "durations": {
                "avg": round(statistics.mean(durations), 3) if durations else 0.0,
                "median": round(statistics.median(durations), 3) if durations else 0.0,
                "p95": _percentile(durations, 0.95),
                "p99": _percentile(durations, 0.99),
                "max": round(max(durations), 3) if durations else 0.0,
                "min": round(min(durations), 3) if durations else 0.0,
            },
            "by_group": build.get("by_group", {}),
            "by_priority": build.get("by_priority", {}),
            "slowest": self._slowest(store, build_id, 10),
            "failures": self._failures(store, build_id, 50),
            "flaky_cases": self._flaky_cases(store, build_id, 50),
            "generated_at": time.time(),
        }
        return report

    def _slowest(self, store, build_id: str, top: int) -> list[dict]:
        records = store.results(build_id, order_by="duration", order="desc",
                                limit=top)
        return [{"case_id": r.get("case_id"), "case_name": r.get("case_name"),
                 "status": r.get("status"), "duration": r.get("duration"),
                 "group": r.get("group")} for r in records]

    def _failures(self, store, build_id: str, top: int) -> list[dict]:
        records = store.results(build_id, where=[("status", "in", list(FAILED_STATUSES))],
                                limit=top)
        out = []
        for r in records:
            failing_step = next((s for s in r.get("steps", [])
                                 if s.get("status") in ("failed", "error")), None)
            failing_assert = next((a for a in r.get("assertions", [])
                                   if not a.get("ok")), None)
            out.append({
                "case_id": r.get("case_id"),
                "case_name": r.get("case_name"),
                "status": r.get("status"),
                "group": r.get("group"),
                "priority": r.get("priority"),
                "duration": r.get("duration"),
                "reason": (failing_assert or {}).get("message")
                or (failing_step or {}).get("message") or r.get("message") or "",
            })
        return out

    def _flaky_cases(self, store, build_id: str, top: int) -> list[dict]:
        """首轮失败、重试后转绿的用例清单（清楚标注尝试次数与每轮状态）。"""
        records = store.results(build_id, where=[("flaky", "eq", True)], limit=top)
        return [{
            "case_id": r.get("case_id"),
            "case_name": r.get("case_name"),
            "group": r.get("group"),
            "priority": r.get("priority"),
            "first_status": r.get("first_status"),
            "attempt_count": r.get("attempt_count", 1),
            "attempts": r.get("attempts", []),
            "duration": r.get("duration"),
        } for r in records]

    # -- 项目级趋势 / 汇总 ------------------------------------------------
    def project_report(self, project_id: str, limit: int = 20) -> dict:
        store = self.builds.for_project(project_id)
        builds = store.list_builds()[:limit]
        points = []
        for b in reversed(builds):
            finished = b.get("total", 0) - b.get("skipped", 0)
            points.append({
                "build_id": b["id"],
                "name": b.get("name") or b["id"],
                "status": b.get("status"),
                "trigger": b.get("trigger"),
                "total": b.get("total", 0),
                "passed": b.get("passed", 0),
                "failed": b.get("failed", 0) + b.get("error", 0) + b.get("timeout", 0),
                "flaky": b.get("flaky", 0),
                "pass_rate": round(b.get("passed", 0) / finished * 100, 1) if finished else 0,
                "duration": b.get("duration", 0.0),
                "finished_at": b.get("finished_at"),
            })
        return {
            "project_id": project_id,
            "builds": points,
            "aggregate": self._aggregate(builds),
        }

    def _aggregate(self, builds: list[dict]) -> dict:
        if not builds:
            return {"total_builds": 0, "avg_pass_rate": 0.0, "avg_duration": 0.0,
                    "total_passed": 0, "total_cases": 0}
        rates, durations = [], []
        tp = tc = 0
        for b in builds:
            finished = b.get("total", 0) - b.get("skipped", 0)
            if finished:
                rates.append(b.get("passed", 0) / finished * 100)
            if b.get("duration"):
                durations.append(b["duration"])
            tp += b.get("passed", 0)
            tc += b.get("total", 0)
        return {
            "total_builds": len(builds),
            "avg_pass_rate": round(statistics.mean(rates), 1) if rates else 0.0,
            "avg_duration": round(statistics.mean(durations), 3) if durations else 0.0,
            "total_passed": tp,
            "total_cases": tc,
        }
