from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.innovator_pareto_v2.run import (
    DEFAULT_INPUT,
    AdaptiveStepSimulator,
    compute_fp4_quantized_shapes,
    evaluate_fixture,
    nvfp4_v_scale_swizzle_coord,
)


class InnovatorParetoV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))

    def test_committed_outputs_match_check_flag(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "innovator_pareto_v2" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_adaptive_step_controller_disables_spec_at_bs64(self) -> None:
        res = evaluate_fixture(self.fixture)["adaptive_eagle3"]
        by_id = {r["case_id"]: r for r in res}
        bs1 = by_id["bs1_code_gen_high_accept_0_82"]
        bs64 = by_id["bs64_high_load_random_creative_0_25_counterexample"]
        self.assertGreaterEqual(bs1["converged_adaptive_steps"], 3)
        self.assertGreater(bs1["adaptive_vs_nospec_speedup_x"], 1.5)
        self.assertEqual(bs64["converged_adaptive_steps"], 0)
        self.assertGreater(bs64["adaptive_vs_fixed_speedup_x"], 1.3)

    def test_fp4_swizzle_128x4_padding_overhead_vs_8x4(self) -> None:
        s128 = compute_fp4_quantized_shapes(m=8, k=7168, is_sf_swizzled_layout=True, is_sf_8x4_layout=False)
        s8 = compute_fp4_quantized_shapes(m=8, k=7168, is_sf_swizzled_layout=True, is_sf_8x4_layout=True)
        self.assertEqual(s128["swizzled_sf_rows"], 128)
        self.assertEqual(s8["swizzled_sf_rows"], 8)
        self.assertGreater(s128["sf_padding_overhead_pct"], 1000.0)
        self.assertEqual(s8["sf_padding_overhead_pct"], 0.0)

    def test_nvfp4_v_scale_swizzle_bijection_within_4_token_tile(self) -> None:
        scale_dim = 8
        seen = set()
        for tok in range(4):
            for sc in range(scale_dim):
                coord = nvfp4_v_scale_swizzle_coord(tok, sc, scale_dim)
                self.assertTrue(0 <= coord[0] < 4)
                self.assertTrue(0 <= coord[1] < scale_dim)
                seen.add(coord)
        self.assertEqual(len(seen), 4 * scale_dim)

    def test_pareto_frontier_ordering(self) -> None:
        pareto = {r["case_id"]: r for r in evaluate_fixture(self.fixture)["innovator_pareto"]}
        # Wafer-scale / SRAM-LPU achieve highest single-user tok/s, while B200/NVL72 achieve lowest $/1M tokens
        self.assertGreater(
            pareto["cerebras_wse3_wafer_scale_70b"]["single_user_tok_s"],
            pareto["fireworks_together_b200_nvfp4_eagle3_70b"]["single_user_tok_s"],
        )
        self.assertLess(
            pareto["fireworks_together_b200_nvfp4_eagle3_70b"]["cost_per_1m_output_tokens_usd"],
            pareto["groq_lpu_sram_deterministic_70b"]["cost_per_1m_output_tokens_usd"],
        )


if __name__ == "__main__":
    unittest.main()
