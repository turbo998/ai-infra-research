import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase135Experiments(unittest.TestCase):
    def test_flashinfer_triton_tilelang_phase135(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_qwen38_pdl_sdsa_sm90_kimi3_sm120_triton_cdna5_tilelang_phase135"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        data = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["sm120_paged_mqa_logits_routing"]["total_new_tuned_shapes"], 65)
        self.assertGreater(data["tilelang_concurrent_jit_and_ascend_oob"]["compile_speedup_ratio"], 6.0)

    def test_sglang_vllm_trtllm_phase135(self) -> None:
        exp = ROOT / "experiments" / "sglang_mimo_h20_cudagraph_hicache_vllm_tool_eplb_mtp_nccl_trtllm_sm120_phase135"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        data = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["vllm_incremental_custom_tool_decode"]["speedup_ratio"], 20.0)
        self.assertGreater(data["sglang_deduped_cudagraph_preupload"]["speedup_ratio"], 8.0)

    def test_megatron_pytorch_mooncake_lmcache_dynamo_nixl_phase135(self) -> None:
        exp = ROOT / "experiments" / "megatron_rmsnorm_gdp_pytorch_rocm_veclib_mooncake_p2pdma_tent_lmcache_dynamo_nixl_phase135"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        data = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "verified")
        self.assertAlmostEqual(data["mooncake_rocm_mi308x_p2pdma_kallsyms_fix"]["bandwidth_recovery_ratio"], 15.9826, places=4)
        self.assertEqual(data["nixl_per_thread_random_buffer"]["syscall_reduction_ratio"], 24.0)


if __name__ == "__main__":
    unittest.main()
