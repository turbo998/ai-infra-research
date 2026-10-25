from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPhase83Experiments(unittest.TestCase):
    def _run_and_check(self, exp_name: str) -> dict:
        exp_dir = ROOT / "experiments" / exp_name
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        res_gen = subprocess.run([sys.executable, str(run_py)], capture_output=True, text=True, check=True)
        self.assertEqual(res_gen.returncode, 0)
        res_chk = subprocess.run([sys.executable, str(run_py), "--check"], capture_output=True, text=True, check=True)
        self.assertEqual(res_chk.returncode, 0)
        self.assertIn("own-offline outputs verified", res_chk.stdout)
        return json.loads(summary_json.read_text(encoding="utf-8"))

    def test_flashinfer_mega_moe_ll32_qsa_nvfp4_sm120_pcie_ce_ring_kimi_k3_mhc_h4096(self) -> None:
        data = self._run_and_check("flashinfer_mega_moe_ll32_qsa_nvfp4_sm120_pcie_ce_ring_kimi_k3_mhc_h4096")
        self.assertEqual(data["evidence_type"], "own-offline")
        m = data["metrics"]
        self.assertEqual(m["mega_moe_ll_max_t"], 32)
        self.assertEqual(m["mega_moe_t1_eager_speedup"], 1.15)
        self.assertEqual(m["qsa_nvfp4_sm120_max_graph_speedup"], 1.90)
        self.assertEqual(m["pcie_ce_ring_tp8_96mib_speedup"], 1.111)
        self.assertEqual(m["kimi_k3_front_evict_first_max_speedup"], 1.201)
        self.assertEqual(m["mega_mhc_h4096_b200_t1024_speedup"], 1.426)

    def test_sglang_msa_v1_nvfp4_kv_cp_dsa_overlap_vllm_bp_eagle_tail_trtllm_awq_moe(self) -> None:
        data = self._run_and_check("sglang_msa_v1_nvfp4_kv_cp_dsa_overlap_vllm_bp_eagle_tail_trtllm_awq_moe")
        self.assertEqual(data["evidence_type"], "own-offline")
        m = data["metrics"]
        self.assertEqual(m["nvfp4_kv_compression_ratio"], 3.56)
        self.assertEqual(m["cp_dsa_overlap_speedup"], 1.71)
        self.assertEqual(m["vllm_bp_failure_threshold"], 3)
        self.assertEqual(m["vllm_eagle_partial_tail_hit_pct"], 96.88)
        self.assertEqual(m["trtllm_awq_error_reduction_factor"], 14.8)

    def test_fa4_sm90_sm100_ragged_tma_m64_bwd_verl_sglang_refit_megatron_ep_recompute_mooncake_nixl_dynamo_te(self) -> None:
        data = self._run_and_check("fa4_sm90_sm100_ragged_tma_m64_bwd_verl_sglang_refit_megatron_ep_recompute_mooncake_nixl_dynamo_te")
        self.assertEqual(data["evidence_type"], "own-offline")
        m = data["metrics"]
        self.assertEqual(m["fa4_m64_hd256_tmem_cols"], 448)
        self.assertEqual(m["fa4_ragged_tma_nan_dq_poison_rate"], 0.0)
        self.assertEqual(m["verl_mxfp4_qat_kl_reduction_x"], 2.86)
        self.assertEqual(m["megatron_ep_recompute_mem_reduction_x"], 4.27)
        self.assertEqual(m["mooncake_shard_lock_reduction_x"], 12.8)


if __name__ == "__main__":
    unittest.main()
