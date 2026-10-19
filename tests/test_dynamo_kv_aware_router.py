from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.dynamo_kv_aware_router.run import (
    dynamo_worker_logit,
    generate_summary,
    should_bypass_remote_prefill_isl_bounding,
)


class DynamoKVAwareRouterTest(unittest.TestCase):
    def test_rational_decay_and_isl_bounding_gate(self) -> None:
        summary = generate_summary()
        curve = summary["rational_decay_curve"]
        self.assertAlmostEqual(curve[0]["overlap_credit_decay"], 1.0, places=4)
        self.assertLess(curve[-1]["overlap_credit_decay"], 0.3)

        bypass, eff_isl, ratio = should_bypass_remote_prefill_isl_bounding(
            prompt_tokens=16384, decode_chosen_cached_tokens=15360, decode_busy=False
        )
        self.assertTrue(bypass)
        self.assertEqual(eff_isl, 1024)
        self.assertLess(ratio, 0.7)

        busy_bypass, _, _ = should_bypass_remote_prefill_isl_bounding(
            prompt_tokens=16384, decode_chosen_cached_tokens=15360, decode_busy=True
        )
        self.assertFalse(busy_bypass)

        score = dynamo_worker_logit(
            isl_tokens=8192,
            block_size=64,
            device_overlap_blocks=100.0,
            host_overlap_blocks=10.0,
            shared_beyond_device_blocks=4,
            active_prefill_tokens=2048,
            min_active_prefill_tokens=0,
            decode_cost_blocks=6.0,
            active_requests=3,
        )
        self.assertLess(score["overlap_credit_decay"], 1.0)
        self.assertGreater(score["logit"], 0.0)

    def test_router_policy_comparison_and_cli_check(self) -> None:
        summary = generate_summary()
        hf = summary["headline_findings"]
        self.assertGreater(hf["dynamo_bypass_hit_rate_pct"], hf["round_robin_hit_rate_pct"] * 2.2)
        self.assertLess(hf["dynamo_bypass_p99_ttft_ms"], hf["sticky_no_decay_p99_ttft_ms"] * 0.2)
        self.assertLess(hf["dynamo_bypass_p99_ttft_ms"], hf["round_robin_p99_ttft_ms"] * 0.35)

        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "experiments/dynamo_kv_aware_router/run.py", "--check"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
