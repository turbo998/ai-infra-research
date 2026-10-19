import unittest

from experiments.sm100_blackwell_fmha_fa4.run import build_summary, run_check


class TestSm100BlackwellFmhaFa4(unittest.TestCase):
    def test_sm100_tmem_layout_and_warp_specialization(self) -> None:
        summary = build_summary()
        run_check(summary)
        tmem = summary["tmem_layout"]
        self.assertEqual(tmem["offsets"]["S0"], [0, 128])
        self.assertEqual(tmem["offsets"]["S1"], [128, 256])
        self.assertEqual(tmem["offsets"]["P0_aliased_in_S0"], [32, 64])
        self.assertEqual(tmem["offsets"]["P1_aliased_in_S1"], [160, 192])
        self.assertEqual(tmem["offsets"]["O0"], [256, 384])
        self.assertEqual(tmem["offsets"]["O1"], [384, 512])
        self.assertEqual(tmem["offsets"]["kEnd"], 512)

    def test_mufu_emulation_and_lazy_rescale_speedup(self) -> None:
        summary = build_summary()
        hl = summary["headline_metrics"]
        self.assertGreater(hl["bf16_fa4_tflops"], 1450.0)
        self.assertGreater(hl["fp8_fa4_tflops"], 2250.0)
        self.assertGreater(hl["sm103_b300_fp8_tflops"], hl["fp8_fa4_tflops"])


if __name__ == "__main__":
    unittest.main()
