from __future__ import annotations

import unittest

from experiments.warp_specialized_kernels.run import (
    evaluate_pingpong_overlap,
    evaluate_register_reallocation,
    evaluate_sm90_vs_sm100_workloads,
    evaluate_wave_quantization,
    run_simulation,
    validate_results,
)


class TestWarpSpecializedKernels(unittest.TestCase):
    def test_register_reallocation_contract(self) -> None:
        c192 = evaluate_register_reallocation(128, 192, 64)
        self.assertEqual(c192.accum_regs_per_mma_thread, 192)
        self.assertFalse(c192.heavy_register_pressure)
        self.assertEqual((c192.load_wg_regs_per_thread, c192.mma_wg_regs_per_thread), (40, 232))
        self.assertEqual(c192.total_allocated_regs_32b, 64512)

        c256 = evaluate_register_reallocation(128, 256, 64)
        self.assertEqual(c256.accum_regs_per_mma_thread, 256)
        self.assertTrue(c256.heavy_register_pressure)
        self.assertEqual((c256.load_wg_regs_per_thread, c256.mma_wg_regs_per_thread), (24, 240))
        self.assertEqual(c256.total_allocated_regs_32b, 64512)
        self.assertGreater(c256.speedup_vs_uniform, 1.5)

    def test_pingpong_and_sm100_tmem(self) -> None:
        pp = evaluate_pingpong_overlap()
        self.assertTrue(all(row.pingpong_speedup > 1.15 for row in pp))
        waves = evaluate_wave_quantization()
        self.assertTrue(all(w.b200_clc_persistent_efficiency_pct > w.b200_static_efficiency_pct for w in waves))
        wl = evaluate_sm90_vs_sm100_workloads()
        self.assertEqual(wl[3].l2_traffic_reduction_pct, 50.0)
        self.assertEqual(wl[3].sm100_mma_threads_per_cta, 32)
        self.assertGreater(wl[3].sm100_sol_pct, wl[3].sm90_sol_pct)

    def test_full_simulation_validation(self) -> None:
        data = run_simulation()
        validate_results(data)


if __name__ == "__main__":
    unittest.main()
