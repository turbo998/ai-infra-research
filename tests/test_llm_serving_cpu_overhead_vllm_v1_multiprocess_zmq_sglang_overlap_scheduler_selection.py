from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmServingCpuOverheadVllmV1MultiprocessZmqSglangOverlapSchedulerSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "llm_serving_cpu_overhead_vllm_v1_multiprocess_zmq_sglang_overlap_scheduler_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["exposed_cpu_bubble_ms_per_step"], 0.2)
        self.assertGreater(data["tiers"][3]["gpu_active_duty_cycle_pct"], 99.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "llm-serving-cpu-overhead-vllm-v1-multiprocess-zmq-sglang-overlap-scheduler-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "llm-serving-cpu-overhead-vllm-v1-multiprocess-zmq-sglang-overlap-scheduler-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "llm-serving-cpu-overhead-vllm-v1-multiprocess-zmq-sglang-overlap-scheduler-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "llm-serving-cpu-overhead-vllm-v1-multiprocess-zmq-sglang-overlap-scheduler-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "llm-serving-cpu-overhead-vllm-v1-multiprocess-zmq-sglang-overlap-scheduler-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
