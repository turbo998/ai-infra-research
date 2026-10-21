import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase54Experiments(unittest.TestCase):
    def _run_and_load(self, exp_dir_name: str) -> dict:
        exp_dir = ROOT / "experiments" / exp_dir_name
        script = exp_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.returncode, 0)
        summary_path = exp_dir / "summary.json"
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_sm120_megamoe_spill_trtllm_multiwave_kimi_k3_nvfp4(self) -> None:
        data = self._run_and_load("flashinfer-sm120-megamoe-spill-trtllm-multiwave-kimi-k3-nvfp4")
        self.assertTrue(data["all_invariants_passed"])
        sm120 = data["sm120_megamoe_register_budget"]
        self.assertEqual(sm120["spills_at_head_all_toolkits"], 0)
        self.assertEqual(sm120["new_max_rows_cap"], 32640)
        self.assertEqual(sm120["decoded_high_at_32768"], -32768)
        mw = data["trtllm_gen_multi_wave_kv_split"]
        self.assertEqual(mw["b38_exact_scratch_splits"], 7)
        self.assertEqual(mw["b38_one_byte_short_splits"], 1)
        self.assertAlmostEqual(mw["geomean_speedup_64k_lognormal"], 1.86)
        knv = data["kimi_k3_nvfp4_and_visualgen"]
        self.assertEqual(knv["kimi_k3_round2"]["libraries_total"], 161)
        self.assertTrue(knv["trtllm_visualgen_zero_copy"]["zero_copy_shares_memory"])

    def test_sglang_dsv4_window_sparse_prefill_indexk_cache_radix_offload_vllm_moe(self) -> None:
        data = self._run_and_load("sglang-dsv4-window-sparse-prefill-indexk-cache-radix-offload-vllm-moe")
        self.assertTrue(data["all_invariants_passed"])
        dsv4 = data["dsv41_window_and_indexk"]
        self.assertEqual(dsv4["history_src_reduction_factor"], 60)
        self.assertEqual(dsv4["slots_r4_count"], 576)
        self.assertTrue(dsv4["use_sparse_prefill_after"])
        radix = data["radix_cache_offload_ordering"]
        self.assertTrue(radix["old_reads_unwritten_kv"])
        self.assertFalse(radix["new_reads_unwritten_kv"])
        moe = data["moe_routing_simulation_guards"]
        self.assertEqual(moe["vllm_simulated_moe"]["selected_backend"], "decomposed_modular_moe")

    def test_megatron_gdn_cudagraph_titan_gradclip_lmcache_prefetch_vmm_mooncake_nixl(self) -> None:
        data = self._run_and_load("megatron-gdn-cudagraph-titan-gradclip-lmcache-prefetch-vmm-mooncake-nixl")
        self.assertTrue(data["all_invariants_passed"])
        mt = data["megatron_gdn_and_torchtitan"]
        self.assertEqual(mt["gdn_persistent_chunk_descriptors_new"], 64)
        self.assertEqual(mt["lora_b_new_typecheck"], "Partial")
        lm = data["lmcache_watermark_and_vmm_ipc"]
        self.assertEqual(lm["new_admitted_small_8mib"], 32)
        self.assertEqual(lm["vmm_padded_byte_span"], 3584)
        tr = data["mooncake_nixl_dynamo_transport"]
        self.assertTrue(tr["mooncake_cpu_tcp_dram_to_gpu_new"])
        self.assertEqual(tr["nixl_final_execution_count"], 3)
        self.assertIsNone(tr["dynamo_corrupt_mm_tokens"])


if __name__ == "__main__":
    unittest.main()
