from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase47Experiments(unittest.TestCase):
    def _run_and_load(self, exp_name: str) -> dict:
        script = REPO_ROOT / "experiments" / exp_name / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[OK] Wrote", res.stdout)
        summary_path = script.with_name("summary.json")
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_topic1_cake_fp4_allreduce_inflight_and_smem_reuse(self) -> None:
        data = self._run_and_load(
            "flashinfer_cake_fp4_allreduce_inflight_kimi_k3_deepgemm_moe_ep_sm107"
        )
        self.assertEqual(data["phase"], 47)
        self.assertEqual(data["priority"], 56)
        ar = data["cake_fp4_allreduce"]
        self.assertTrue(ar["nibble_detector_verified"])
        self.assertEqual(ar["geomean_speedup_nvfp4"]["sm_107a"], 1.023)
        self.assertEqual(ar["tp4_pass2_l2_reduction_pct"], 80.0)

    def test_topic1_kimi_k3_attn_res_round2_and_sm107_moe_ep(self) -> None:
        data = self._run_and_load(
            "flashinfer_cake_fp4_allreduce_inflight_kimi_k3_deepgemm_moe_ep_sm107"
        )
        k3 = data["cake_kimi_k3_attn_res"]
        self.assertEqual(k3["programs_count"], 133)
        self.assertEqual(k3["libraries_total"], 161)
        self.assertEqual(k3["contract_rows_total"], 582)
        dg = data["deepgemm_and_moe_ep"]
        self.assertEqual(
            dg["deepgemm_batched_projection"]["gb300_latencies_us"]["T2_swap_m16"],
            9.727,
        )
        self.assertIn("sm_107a", dg["moe_ep_sm107"]["supported_sm_targets"])

    def test_topic2_trtllm_flux_nvfp4_cutedsl_fp8_mha(self) -> None:
        data = self._run_and_load(
            "trtllm_flux_nvfp4_cutedsl_fp8_custom_mask_gemma4_vllm_sglang"
        )
        flux = data["trtllm_flux"]
        self.assertEqual(flux["step_latency_gb300"]["1024x1024_speedup"], 1.75)
        self.assertEqual(flux["step_latency_gb300"]["2048x2048_speedup"], 1.70)
        self.assertEqual(flux["micro_fusions"]["nvfp4_gemm_gelu_speedup"], 1.43)
        self.assertEqual(flux["micro_fusions"]["dit_qk_norm_rope_quant_speedup"], 1.50)

    def test_topic2_trtllm_fp8_custom_mask_and_gemma4_kv_geometry(self) -> None:
        data = self._run_and_load(
            "trtllm_flux_nvfp4_cutedsl_fp8_custom_mask_gemma4_vllm_sglang"
        )
        fg = data["trtllm_fp8_and_gemma4"]
        self.assertEqual(fg["triton_custom_mask_fp8_saturated"][0], -448.0)
        self.assertEqual(fg["triton_custom_mask_fp8_saturated"][-1], 448.0)
        self.assertTrue(fg["gemma4_auto_kv_cache_manager_v2"])
        self.assertEqual(fg["gemma4_global_gqa_ratio"], 16)

    def test_topic2_vllm_xpu_osa_and_sglang_tp_collective_release(self) -> None:
        data = self._run_and_load(
            "trtllm_flux_nvfp4_cutedsl_fp8_custom_mask_gemma4_vllm_sglang"
        )
        vs = data["vllm_and_sglang"]
        self.assertEqual(
            vs["vllm_xpu_osa_arc_pro_b60"]["sonnet_bs32_k7_peak_tps"], 468.55
        )
        self.assertTrue(vs["sglang_disagg_decode_tp_sync"]["step4_collective_still_inflight"])
        self.assertTrue(vs["sglang_disagg_decode_tp_sync"]["step5_collective_ready_to_free"])

    def test_topic3_mooncake_xpu_dmabuf_and_lmcache_grpc_codec(self) -> None:
        data = self._run_and_load(
            "mooncake_xpu_dmabuf_lmcache_grpc_dynamo_fpm_nixl_megatron_te"
        )
        mc = data["mooncake_xpu_dmabuf"]
        self.assertTrue(mc["valid_slice_accepted"])
        self.assertEqual(mc["computed_fd_offset"], 0x42000)
        self.assertTrue(mc["uint64_overflow_rejected"])
        ldn = data["lmcache_dynamo_nixl"]
        self.assertEqual(ldn["lmcache_grpc_proto_codec"]["encode_8k_speedup"], 3.38)
        self.assertEqual(ldn["lmcache_grpc_proto_codec"]["decode_8k_speedup"], 2.50)

    def test_topic3_dynamo_nixl_megatron_gdn_and_te_jax_mhc(self) -> None:
        data = self._run_and_load(
            "mooncake_xpu_dmabuf_lmcache_grpc_dynamo_fpm_nixl_megatron_te"
        )
        ldn = data["lmcache_dynamo_nixl"]
        self.assertEqual(ldn["nixl_telemetry"]["assigned_execution_ids"], [1, None, 2])
        mtt = data["megatron_te_torchtitan"]
        self.assertEqual(
            mtt["megatron_gdn_cuda_graph_prefill"]["total_active_chunks"], 7
        )
        self.assertEqual(
            mtt["te_jax_triton_mhc_and_kda"]["packed_coeff_width_2n_plus_n2"], 24
        )
        self.assertTrue(
            mtt["te_jax_triton_mhc_and_kda"]["sinkhorn_bwd_debug_barrier_fixes_smem_raw_race"]
        )


if __name__ == "__main__":
    unittest.main()
