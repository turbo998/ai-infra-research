import unittest
from experiments.llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection.run import (
    evaluate_lora_paging_tiers,
)


class TestMultiLoraPagingSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_lora_paging_tiers()
        self.assertAlmostEqual(res["naive_decode_stall_ms"], 64.00, places=2)
        self.assertAlmostEqual(res["tier4_decode_stall_ms"], 0.08, places=2)
        self.assertAlmostEqual(res["stall_reduction_x"], 800.0, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["decode_stall_on_miss_ms"], 64.00)
        self.assertEqual(t4["decode_stall_on_miss_ms"], 0.08)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
