import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase81Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        exp_dir = ROOT / "experiments" / rel_dir
        script = exp_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=str(ROOT))
        with open(exp_dir / "summary.json", encoding="utf-8") as f:
            return json.load(f)

    def test_flashinfer_qsa_nvfp4_sm120_wholetile_dcp_a2a_ll128_w4a16_split_moe(self):
        data = self._run_and_load("flashinfer_qsa_nvfp4_sm120_wholetile_dcp_a2a_ll128_w4a16_split_moe")
        self.assertEqual(data["phase"], 81)
        self.assertEqual(data["sm120_qsa_nvfp4"]["co_resident_cap_ctas"], 376)
        self.assertEqual(data["sm120_qsa_nvfp4"]["whole_tile_min_tokens_per_cta"], 704)
        self.assertGreaterEqual(data["dcp_alltoall_ll128"]["max_cold_speedup_x"], 1.33)
        self.assertEqual(data["w4a16_split_moe_and_sm103_attn"]["sm103_tmem_cols_after"], 480)

    def test_sglang_paged_experts_uva_hicache_mamba_vllm_priority_preempt_d2d_seed(self):
        data = self._run_and_load("sglang_paged_experts_uva_hicache_mamba_vllm_priority_preempt_d2d_seed")
        self.assertEqual(data["phase"], 81)
        self.assertEqual(data["sglang_paged_experts"]["vram_savings_pct"], 87.5)
        self.assertEqual(data["sglang_hicache_mamba"]["configs"][0]["orphan_kv_prefixes_recomputed"], 192)
        self.assertEqual(data["vllm_and_trtllm_runtime"]["priority_self_preempt_starved_higher_prio_reqs_after_fix"], 0)
        self.assertGreater(data["vllm_and_trtllm_runtime"]["d2d_preload_speedup_x"], 35.0)

    def test_fa4_sm100_bwd_pack_gqa_ragged_tma_desc_verl_lora_host_stage_mooncake_efa(self):
        data = self._run_and_load("fa4_sm100_bwd_pack_gqa_ragged_tma_desc_verl_lora_host_stage_mooncake_efa")
        self.assertEqual(data["phase"], 81)
        self.assertEqual(data["fa4_sm100_bwd_and_tma"]["ragged_tma_new_bounded_slots"], 888)
        self.assertEqual(data["verl_lora_bucket_staging"]["max_speedup_x"], 5.38)
        self.assertEqual(data["mooncake_efa_and_nixl_spsc"]["bounded_active_ah_entries_after"], 240)


if __name__ == "__main__":
    unittest.main()
