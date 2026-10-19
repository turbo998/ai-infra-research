from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashmla_paged_decode_scheduling.run import (
    generate_summary,
    simulate_kv_formats_and_roofline,
    simulate_variable_length_decode_schedulers,
)


class FlashMLAPagedDecodeSchedulingTest(unittest.TestCase):
    def test_kv_formats_and_cta_pair_split(self) -> None:
        rows = {r["format_id"]: r for r in simulate_kv_formats_and_roofline()}
        self.assertEqual(rows["GQA_8_BF16"]["bytes_per_token"], 4096)
        self.assertEqual(rows["MLA_ABSORBED_BF16_D576"]["bytes_per_token"], 1152)
        self.assertEqual(rows["FLASHMLA_V41_FP8_528B"]["bytes_per_token"], 528)
        self.assertEqual(rows["FLASHMLA_V41_FP8_528B"]["cta0_raw_bytes"], 256)
        self.assertEqual(rows["FLASHMLA_V41_FP8_528B"]["cta1_raw_bytes"], 272)
        self.assertEqual(rows["FLASHMLA_V41_FP4_288B"]["bytes_per_token"], 288)
        self.assertEqual(rows["FLASHMLA_V41_FP4_288B"]["cta0_raw_bytes"], 128)
        self.assertEqual(rows["FLASHMLA_V41_FP4_288B"]["cta1_raw_bytes"], 160)
        self.assertGreater(
            rows["FLASHMLA_V41_FP4_288B"]["arithmetic_intensity_flop_per_byte"],
            rows["FLASHMLA_V41_FP8_528B"]["arithmetic_intensity_flop_per_byte"],
        )

    def test_get_mla_metadata_eliminates_wave_imbalance(self) -> None:
        sched = {r["scheduler"]: r for r in simulate_variable_length_decode_schedulers()}
        static = sched["STATIC_REQ_PER_PART_74"]
        sm100 = sched["FLASHMLA_GET_MLA_METADATA_SM100_74"]
        self.assertGreater(static["wave_imbalance_pct"], 85.0)
        self.assertLess(sm100["wave_imbalance_pct"], 2.0)
        self.assertGreaterEqual(sm100["speedup_vs_static_x"], 5.5)

    def test_cli_check_passes(self) -> None:
        summary = generate_summary()
        self.assertEqual(summary["experiment_id"], "flashmla_paged_decode_scheduling")
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "experiments/flashmla_paged_decode_scheduling/run.py", "--check"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
