import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPhase45Experiments(unittest.TestCase):
    def _run_and_load(self, exp_dir: str) -> dict:
        script = REPO_ROOT / "experiments" / exp_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("passed all checks", res.stdout)
        summary_path = REPO_ROOT / "experiments" / exp_dir / "summary.json"
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_cake_mhc_msa_kmajor_multiwave_bmm(self):
        data = self._run_and_load("flashinfer_cake_mhc_msa_kmajor_multiwave_bmm")
        self.assertTrue(data["all_invariants_passed"])
        mhc = data["cake_mhc_four_stream_training"]
        self.assertEqual(mhc["hidden_total"], 16384)
        self.assertEqual(mhc["gb300_benchmarks"]["T65536"]["fwd_speedup"], 15.95)
        self.assertEqual(mhc["gb300_benchmarks"]["T65536"]["ckpt_speedup"], 14.79)
        msa = data["cake_msa_and_comm_optimizations"]
        self.assertEqual(msa["msa_routes"]["gb300_geomean_speedup_106_shapes"], 3.39)
        self.assertEqual(msa["fp4_allreduce_tp4_pass2_l2_bytes_per_row_after"], 16)
        bmm = data["trtllm_gen_multiwave_and_bmm_ordering"]
        self.assertEqual(bmm["new_partial_stats_entries"], 9216)
        self.assertEqual(bmm["selected_default_kernel"], "bmm_nvfp4_c1x1x1_unsplit")

    def test_trtllm_minimax_m3_router_flux_sglang_quark_vllm_moriio(self):
        data = self._run_and_load("trtllm_minimax_m3_router_flux_sglang_quark_vllm_moriio")
        self.assertTrue(data["all_invariants_passed"])
        router = data["minimax_m3_bf16_split_router"]
        self.assertEqual(router["router_bf16_split_terms"], 3)
        self.assertLess(router["abs_error_split_bf16"], 1e-9)
        quark = data["sglang_quark_asm_and_rust_health_ack"]
        self.assertTrue(quark["swizzle_bijection_verified"])
        self.assertEqual(quark["e8m0_one_sentinel"], 0x7F)
        mori = data["vllm_moriio_step_barrier"]
        self.assertEqual(mori["new_sub_ms_timeout_ms"], 1)
        self.assertEqual(mori["reads_pending_at_forward_start"], 0)

    def test_dynamo_kv_cursor_1000slot_nixl_multiqp_torchtitan_mla_cp(self):
        data = self._run_and_load("dynamo_kv_cursor_1000slot_nixl_multiqp_torchtitan_mla_cp")
        self.assertTrue(data["all_invariants_passed"])
        dyn = data["dynamo_kv_state_agent_scaling"]
        self.assertEqual(dyn["arcswap_stores_after"], 2)
        self.assertEqual(dyn["socket_reduction_factor"], 8)
        nixl = data["nixl_multiqp_and_mooncake_ascend_tent"]
        self.assertEqual(nixl["gpunetio_qpn_header"], [0x4E585150, 2, 4, 5])
        self.assertTrue(nixl["tent_cpu_client_dram_to_gpu"])
        mla = data["torchtitan_mla_cp_and_grad_clipper"]
        self.assertEqual(mla["mla_allgather_volume_reduction_x"], 71.1111)
        self.assertEqual(mla["ulysses_a2a_calls_after"], 2)


if __name__ == "__main__":
    unittest.main()
