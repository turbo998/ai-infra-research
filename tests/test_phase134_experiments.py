import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase134Experiments(unittest.TestCase):
    def _check_suite(self, dirname: str, expected_qid: str) -> dict:
        exp_dir = ROOT / "experiments" / dirname
        run_py = exp_dir / "run.py"
        summary_path = exp_dir / "results" / "summary.json"
        res = subprocess.run([sys.executable, str(run_py), "--check"], capture_output=True, text=True, check=True)
        self.assertIn("verified:", res.stdout)
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(data["question_id"], expected_qid)
        self.assertEqual(data["phase"], 134)
        self.assertEqual(len(data["scenarios"]), 10)
        self.assertTrue(data["all_invariants_passed"])
        return data

    def test_track_a_phase134(self) -> None:
        data = self._check_suite(
            "flashinfer_cake_mhc_ulysses_topk_nvfp4_mxfp8_triton_cga_cdna5_tilelang_phase134",
            "q-gpu-flashinfer-cake-mhc-ulysses-topk-nvfp4-mxfp8-triton-cga-cdna5-tilelang-phase134",
        )
        by_id = {s["id"]: s for s in data["scenarios"]}
        self.assertAlmostEqual(by_id["flashinfer_mxfp8_early_pdl_launch_dependents"]["optimized"], 8.175)
        self.assertEqual(by_id["tilelang_single_round_cross_warp_allreduce"]["optimized"], 4)

    def test_track_b_phase134(self) -> None:
        data = self._check_suite(
            "sglang_pd_commit_tp_renorm_glm53_vllm_dp_qwen35_dsa_trtllm_eplb_phase134",
            "q-serving-sglang-pd-commit-tp-renorm-glm53-vllm-dp-qwen35-dsa-trtllm-eplb-phase134",
        )
        by_id = {s["id"]: s for s in data["scenarios"]}
        self.assertEqual(by_id["sglang_tp_deterministic_topp_topk_renorm"]["optimized"], 0.0)
        self.assertEqual(by_id["trtllm_openengine_image_routing_metadata"]["optimized"], 89.0)

    def test_track_c_phase134(self) -> None:
        data = self._check_suite(
            "megatron_steplr_rl_logprobs_pytorch_rocm_pickler_mooncake_efa_lmcache_dynamo_nixl_phase134",
            "q-training-megatron-steplr-rl-logprobs-pytorch-rocm-pickler-mooncake-efa-lmcache-dynamo-nixl-phase134",
        )
        by_id = {s["id"]: s for s in data["scenarios"]}
        self.assertAlmostEqual(by_id["mooncake_efa_local_nic_registration_default"]["speedup"], 30.4444)
        self.assertEqual(by_id["pytorch_rocm_mixed_dtype_vectorized_load_arg_t"]["optimized"], 0.0)


if __name__ == "__main__":
    unittest.main()
