from __future__ import annotations

import unittest

from experiments.flashinfer_grouped_mxfp8_prefetch_bf16_fp4_splitk_tilelang_phase114 import (
    run as track_a,
)
from experiments.sglang_megakernels_qwen35_eplb_vllm_dsv41_replay_trtllm_kvcm_phase114 import (
    run as track_b,
)
from experiments.te_fp8_quantizer_megatron_mla_rope_mfsdp_mxfp8_mooncake_phase114 import (
    run as track_c,
)


class TestPhase114Experiments(unittest.TestCase):
    def test_track_a_grouped_mxfp8_prefetch_splitk_tilelang(self) -> None:
        summary = track_a.run_suite()
        track_a.validate(summary)
        self.assertEqual(summary["grouped_mxfp8_prefetch"]["prefetch_groups_ahead"], 3)
        self.assertEqual(summary["deepgemm_minimax_mhc"]["minimax_p8_m6096_program"], "NORM_V5")
        self.assertTrue(summary["cutlass_tilelang_ss_pto"]["tilelang_ss_within_pool"])

    def test_track_b_sglang_megakernel_vllm_dsv41_trtllm_kvcm_cc(self) -> None:
        summary = track_b.run_suite()
        track_b.validate(summary)
        self.assertEqual(summary["sglang_megakernel_bcg_jit_eplb"]["megakernel_api_version"], 2)
        self.assertEqual(summary["vllm_dsv41_stream_serial_gb200"]["graph_pool_memory_savings_mib"], 720)
        self.assertEqual(summary["trtllm_kvcm_v2_cc_child_req_sf"]["cc_copied_plane0_elements"], 4096)

    def test_track_c_te_fp8_megatron_mla_mfsdp_mooncake_nixl(self) -> None:
        summary = track_c.run_suite()
        track_c.validate(summary)
        self.assertEqual(summary["te_megatron_mla_mfsdp"]["mla_rope_transaction_reduction_pct"], 50.0)
        self.assertEqual(summary["torchtitan_moe_rl"]["moe_saved_us_per_step_60layers"], 870.0)
        self.assertEqual(summary["mooncake_lmcache_nixl_hf3fs"]["nixl_hf3fs_data_race_count"], 0)


if __name__ == "__main__":
    unittest.main()
