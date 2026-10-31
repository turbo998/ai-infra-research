import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase136Experiments(unittest.TestCase):
    def test_flashinfer_triton_tilelang_phase136(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_det_gemm_qwen38_gdn_sm107_mxfp8_b200_tinygemm_triton_tilelang_phase136"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        data = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["flashinfer_sm107_mxfp8_and_b200_tinygemm2"]["b200_unbiased_routing_windows_added"], 4)
        self.assertEqual(data["tilelang_cpu_gemm_clear_and_ascend_view"]["zero_fill_traffic_reduction_ratio"], 4.0)

    def test_sglang_vllm_trtllm_phase136(self) -> None:
        exp = ROOT / "experiments" / "sglang_minimax_qwenimage_vllm_parser_int64_dspark_trtllm_musespark_dsa_phase136"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        data = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["sglang_minimax_m3_deterministic_topk"]["tie_break_parity_after"], 1.0)
        self.assertGreater(data["vllm_dspark_split_vocab_and_trtllm_dsa"]["dspark_split_vocab_argmax_speedup"], 2.8)

    def test_megatron_pytorch_mooncake_lmcache_dynamo_nixl_phase136(self) -> None:
        exp = ROOT / "experiments" / "megatron_hybrid_gb300_gdp_pytorch_pdl_transport_mooncake_ep_lmcache_dynamo_nixl_phase136"
        subprocess.run([sys.executable, str(exp / "run.py"), "--check"], check=True)
        data = json.loads((exp / "results" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["mooncake_elastic_ep_and_conductor_cpu_kv"]["elastic_ep_recovery_speedup_ratio"], 25.0)
        self.assertGreater(data["nixl_ucx_progress_thread_min_delay"]["cpu_spin_reduction_ratio"], 28.0)


if __name__ == "__main__":
    unittest.main()
