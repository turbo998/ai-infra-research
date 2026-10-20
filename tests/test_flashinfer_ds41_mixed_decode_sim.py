import unittest

from experiments.flashinfer_ds41_mixed_decode_sim.run import (
    compute_non_interleaved_page_offsets,
    plan_ds41_decode_schedule,
    run_simulation,
    simulate_green_ctx_partitioning,
    simulate_sink_exclusive_lse,
)


class TestFlashInferDS41MixedDecodeSim(unittest.TestCase):
    def test_non_interleaved_page_layout(self) -> None:
        swa_last = compute_non_interleaved_page_offsets(63, fp4=False)
        self.assertEqual(swa_last["token_stride_bytes"], 528)
        self.assertEqual(swa_last["page_bytes"], 64 * 528)
        self.assertEqual(swa_last["data_base_offset"], 63 * 512)
        self.assertEqual(swa_last["scale_base_offset"], 64 * 512 + 63 * 16)

        fp4_last = compute_non_interleaved_page_offsets(63, fp4=True)
        self.assertEqual(fp4_last["token_stride_bytes"], 288)
        self.assertEqual(fp4_last["page_bytes"], 64 * 288)
        self.assertEqual(fp4_last["data_base_offset"], 63 * 256)
        self.assertEqual(fp4_last["scale_base_offset"], 64 * 256 + 63 * 32)

    def test_decode_schedule_transitions(self) -> None:
        s_b4 = plan_ds41_decode_schedule(batch=4, compressed_k=512, sm_count=148)
        self.assertEqual(s_b4["split"], 10)
        self.assertFalse(s_b4["use_ws"])
        self.assertEqual(s_b4["persistent_ctas"], 0)

        s_b160 = plan_ds41_decode_schedule(batch=160, compressed_k=512, sm_count=148)
        self.assertEqual(s_b160["split"], 1)
        self.assertTrue(s_b160["use_ws"])
        self.assertEqual(s_b160["dequant_warps"], 16)
        self.assertEqual(s_b160["waves"], 2)
        self.assertEqual(s_b160["persistent_ctas"], 80)

    def test_sink_exclusive_lse_and_green_ctx(self) -> None:
        lse = simulate_sink_exclusive_lse()
        self.assertLess(lse["lse_merge_error"], 1e-12)
        self.assertAlmostEqual(lse["mass_conservation"], 1.0, places=6)

        gctx = simulate_green_ctx_partitioning(148, [20, 36, 64])
        self.assertEqual(
            [p["aligned_sms"] for p in gctx["partitions"]],
            [24, 40, 64],
        )
        self.assertEqual(gctx["unallocated_remaining_sms"], 20)

        full = run_simulation()
        self.assertGreater(
            full["cache_quantization_and_layout"]["halfway_perturbation_analysis"][
                "reciprocal_perturbation_count"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
