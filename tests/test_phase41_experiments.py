import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41.run import (
    run_experiment as run_rubin_experiment,
)
from experiments.megatron_muon_stream_te_cublasmp_nvfp4_mooncake_cc_nixl.run import (
    run_experiment as run_muon_experiment,
)
from experiments.sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc.run import (
    run_experiment as run_sglang_experiment,
)


ROOT = Path(__file__).resolve().parents[1]


class TestPhase41Experiments(unittest.TestCase):
    def test_flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41(self) -> None:
        summary = run_rubin_experiment()
        self.assertTrue(all(summary["invariants"].values()))
        rubin = summary["metrics"]["rubin_mla_nvfp4"]
        self.assertEqual(rubin["kv_token_bytes"], 352)
        self.assertEqual(rubin["query_row_bytes"], 356)
        self.assertAlmostEqual(rubin["geomean_speedup_all_16"], 1.0436, places=4)
        self.assertAlmostEqual(rubin["geomean_speedup_q4_8"], 1.1039, places=4)
        self.assertAlmostEqual(rubin["peak_speedup"], 1.2483, places=4)
        stepfun = summary["metrics"]["stepfun_moe"]
        self.assertEqual(stepfun["kernels_per_arch"], 111)
        self.assertEqual(stepfun["fc1_pair_raster_l2_weight_read_reduction_ratio"], 2.0)
        dsv41 = summary["metrics"]["vllm_dsv41_replay"]
        self.assertEqual(dsv41["fixed_target_token_single_chunk_rows"], 300)
        self.assertEqual(dsv41["fixed_target_token_two_chunk_rows"], [150, 150])

    def test_sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc(self) -> None:
        summary = run_sglang_experiment()
        self.assertTrue(all(summary["invariants"].values()))
        sgl = summary["metrics"]["sglang_pd_dp_role_switch"]
        self.assertTrue(sgl["legacy_single_phase_partial_fail"]["split_brain"])
        self.assertFalse(sgl["two_phase_preflight_partial_fail"]["split_brain"])
        self.assertEqual(sgl["two_phase_preflight_partial_fail"]["ranks_torn_down"], 0)
        adp = summary["metrics"]["trtllm_adp_kv_event_alignment"]
        self.assertEqual(adp["fixed_overlap_deadlock_free_flags"], [True, True, True])
        self.assertTrue(adp["fixed_rebalance_deadlock_free"])
        mzc = summary["metrics"]["trtllm_mooncake_zerocopy"]
        self.assertFalse(mzc["default_stage_through_host"])
        self.assertEqual(mzc["straddling_page_split_sizes_mib"], [2, 2])
        self.assertEqual(mzc["pinned_host_staging_saved_mib_8gpu"], 1024)

    def test_megatron_muon_stream_te_cublasmp_nvfp4_mooncake_cc_nixl(self) -> None:
        summary = run_muon_experiment()
        self.assertTrue(all(summary["invariants"].values()))
        muon = summary["metrics"]["layersharded_muon_streaming"]
        self.assertEqual(muon["streaming_batched_ns_launches"], 5)
        self.assertEqual(muon["streaming_peak_transient_mib"], 192.0)
        self.assertAlmostEqual(muon["peak_transient_memory_reduction_ratio"], 5.6667, places=4)
        te = summary["metrics"]["te_cublasmp_nvfp4"]
        self.assertEqual(te["device_scalars_buffer_bytes"], 12)
        self.assertEqual(te["algo_for_n4096_tp4"], "SPLIT_P2P_OVERLAP")
        self.assertEqual(te["algo_for_n4352_tp4"], "DEFAULT")
        cc = summary["metrics"]["mooncake_cc_and_nixl_trace"]
        self.assertEqual(cc["grown_window_mib_after_12_low_epochs"], 8)
        self.assertEqual(cc["shrunk_window_mib_after_2_high_epochs"], 4)
        self.assertEqual(cc["nixl_fixed_complete_marks"], 50)

    def test_phase41_cli_check_mode(self) -> None:
        scripts = [
            ROOT / "experiments" / "flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41" / "run.py",
            ROOT / "experiments" / "sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc" / "run.py",
            ROOT / "experiments" / "megatron_muon_stream_te_cublasmp_nvfp4_mooncake_cc_nixl" / "run.py",
        ]
        for script in scripts:
            proc = subprocess.run(
                [sys.executable, str(script), "--check"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{script} failed: {proc.stderr or proc.stdout}")


if __name__ == "__main__":
    unittest.main()
