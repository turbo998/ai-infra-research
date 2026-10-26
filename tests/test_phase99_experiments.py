import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase99Experiments(unittest.TestCase):
    def test_track_a_flashinfer_tilelang_phase99(self) -> None:
        script = ROOT / "experiments" / "flashinfer_qsa_nvfp4_split_cudnn_prefill_kda_wy_kimi_attnres_dsa_store16_minimax_tilelang_phase99" / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track_b_sglang_vllm_trtllm_phase99(self) -> None:
        script = ROOT / "experiments" / "sglang_qsa_indexer_rust_chat_dp_buckets_zimage_failclosed_vllm_megamoe_sp_mxfp4_hisparse_trtllm_phase99" / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track_c_lmcache_nixl_megatron_te_titan_verl_mooncake_dynamo_phase99(self) -> None:
        script = ROOT / "experiments" / "lmcache_mp_bigtable_nixl_ucx_forceclose_megatron_gtp_te_groupedlinear_titan_varlen_verl_mooncake_dynamo_phase99" / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)


if __name__ == "__main__":
    unittest.main()
