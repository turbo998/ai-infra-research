import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase152Experiments(unittest.TestCase):
    def test_track1_flashinfer_triton_tilelang_phase152(self):
        exp_dir = ROOT / "experiments" / "flashinfer_muse_norm_gdn_pairs_dproj_sampling_phase152"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 152)
        self.assertGreater(data["muse_spark_norm_and_valid_rows_guard"]["muse_spark_norm_speedup"], 1.3)
        self.assertGreater(data["sm120_gdn_pair_and_sm100a_8warp_dproj"]["gdn_sm120_pair_speedup"], 1.35)
        self.assertGreater(data["triton_gluon_and_tilelang_ir_cache"]["tilelang_ir_cache_speedup"], 7.0)

    def test_track2_sglang_vllm_trtllm_phase152(self):
        exp_dir = ROOT / "experiments" / "sglang_strided_moe_vllm_nixl_mamba_trtllm_minimax_phase152"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 152)
        self.assertEqual(data["vllm_nixl_mamba_block_outermost_stride"]["corrupted_mamba_blocks_fixed"], 0)
        self.assertEqual(data["trtllm_minimax_m3_vl_and_index_k"]["index_k_cache_memory_saving_pct"], 75.0)

    def test_track3_megatron_mooncake_dynamo_nixl_phase152(self):
        exp_dir = ROOT / "experiments" / "megatron_dsa_routing_mooncake_heal_dynamo_ucx_phase152"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 152)
        self.assertEqual(data["megatron_dsa_min_memory_and_routing_replay"]["dsa_memory_saving_pct"], 58.33)
        self.assertEqual(data["mooncake_self_heal_and_standby_restore"]["dangling_replicas_self_healed"], 12)
        self.assertEqual(data["dynamo_global_router_and_nixl_ucx_sgl"]["ucx_sgl_split_batch_size"], 4096)


if __name__ == "__main__":
    unittest.main()
