import unittest
from experiments.gpu_power_capping_psu_ats_inrush_transient_spike_throttling_selection.run import evaluate_gpu_power_capping_and_edpp


class TestGpuPowerCappingAndEdpp(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_gpu_power_capping_and_edpp()
        self.assertEqual(res["round"], 289)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_uncapped_700w_sync_square_wave_edpp_spikes"]["transient_edpp_peak_w"]
            - tiers["tier4_uniform_620w_locked_clock_plus_dualpipe_comm_overlap_smoothing"]["transient_edpp_peak_w"],
            250.0,
        )
        self.assertGreater(
            tiers["tier4_uniform_620w_locked_clock_plus_dualpipe_comm_overlap_smoothing"]["perf_per_watt_norm"],
            1.14,
        )


if __name__ == "__main__":
    unittest.main()
