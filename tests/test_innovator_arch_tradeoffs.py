import json
from pathlib import Path
import unittest

from experiments.innovator_arch_tradeoffs.run import (
    calculate,
    evaluate_kvouter_case,
    evaluate_speculative_tree_case,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "innovator_arch_tradeoffs"


class InnovatorArchTradeoffsTest(unittest.TestCase):
    def test_checked_in_results_match(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_kvouter_b200_analytical_byte_models(self) -> None:
        fixture = json.loads((EXP / "input.json").read_text(encoding="utf-8"))
        out = calculate(fixture)
        kv_by_id = {entry["result"]["case_id"]: entry["result"] for entry in out["kvouter_cases"]}

        high_reuse = kv_by_id["m3_b200_bf16_q4096_kv32768_sel16_high_reuse"]
        self.assertAlmostEqual(high_reuse["fio1x_gb"], 0.295, places=3)
        self.assertAlmostEqual(high_reuse["fio_act_gb"], 0.546, places=3)
        self.assertAlmostEqual(high_reuse["cio_gb"], 0.294, places=3)
        self.assertEqual(high_reuse["kv_only_compression_ratio"], 4096.0)
        self.assertGreater(high_reuse["total_forward_io_saving_ratio_vs_no_reuse"], 8.0)

    def test_speculative_tree_roofline_and_spatial_sram(self) -> None:
        fixture = json.loads((EXP / "input.json").read_text(encoding="utf-8"))
        out = calculate(fixture)
        spec_by_id = {
            entry["result"]["case_id"]: entry["result"] for entry in out["speculative_tree_cases"]
        }
        sram_by_id = {
            entry["result"]["case_id"]: entry["result"] for entry in out["spatial_sram_cases"]
        }

        tree64 = spec_by_id["sequoia_70b_fp16_b1_tree64_memory_bound_harvest"]
        tree512 = spec_by_id["sequoia_70b_fp16_b1_tree512_compute_overshoot"]
        self.assertEqual(tree64["regime"], "memory_bound")
        self.assertEqual(tree512["regime"], "compute_bound")
        self.assertGreater(tree64["speedup_over_b1_greedy"], tree512["speedup_over_b1_greedy"])

        groq = sram_by_id["groq_tsp_lpu_70b_int8_sram_daisy_chain"]
        wse3 = sram_by_id["cerebras_wse3_900k_pe_44gb_sram_wafer"]
        self.assertFalse(groq["fits_single_unit_sram"])
        self.assertEqual(groq["min_units_for_pure_sram_residency"], 405)
        self.assertFalse(wse3["fits_single_unit_sram"])
        self.assertEqual(wse3["min_units_for_pure_sram_residency"], 3)

    def test_invalid_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_kvouter_case(
                case_id="bad_heads",
                tq=128,
                hq=15,
                hkv=2,
                head_dim=128,
                block_size=128,
                topk=16,
                distinct_blocks=16,
                input_bytes=2,
                partial_bytes=2,
                out_bytes=2,
            )
        with self.assertRaises(ValueError):
            evaluate_speculative_tree_case(
                case_id="bad_accept",
                model_params=70000000000,
                weight_bytes_per_param=2,
                tree_tokens_k=16,
                expected_accepted_tokens=0.5,
                draft_overhead_seconds=0.0,
                hbm_bytes_per_second=3350000000000,
                peak_flops_per_second=989000000000000,
            )


if __name__ == "__main__":
    unittest.main()
