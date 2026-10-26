from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase94Experiments(unittest.TestCase):
    def _run_and_load(self, name: str) -> dict[str, object]:
        exp_dir = ROOT / "experiments" / name
        run_py = exp_dir / "run.py"
        summary_path = exp_dir / "summary.json"
        res = subprocess.run([sys.executable, str(run_py), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_kimi_fp8_muse_minimax_mxfp8_norm_kda_gdn_tilelang_phase94(self) -> None:
        data = self._run_and_load("flashinfer_kimi_fp8_muse_minimax_mxfp8_norm_kda_gdn_tilelang_phase94")
        self.assertEqual(data["phase"], 94)
        self.assertEqual(data["priority"], 103)
        self.assertTrue(data["all_invariants_passed"])
        kimi = data["kimi_fp8_muse_minimax"]
        self.assertEqual(kimi["launcher_reduction_pct"], 90.74)
        self.assertEqual(kimi["fp8_bf16_speedup_x"], 1.3655)
        mxfp8 = data["mxfp8_quant_norm_combine_kda_gdn"]
        self.assertEqual(mxfp8["cta_reduction_x"], 2)
        self.assertEqual(mxfp8["serial_chain_reduction_x"], 32)
        tl = data["tilelang_ascend_placement_and_int64"]
        self.assertEqual(tl["ub_with_simt_bytes"], 216 * 1024)
        self.assertTrue(tl["int64_preserves_order"])

    def test_sglang_xeon_grok2_xpu_wan_vllm_uva_p2p_loc_trtllm_phase94(self) -> None:
        data = self._run_and_load("sglang_xeon_grok2_xpu_wan_vllm_uva_p2p_loc_trtllm_phase94")
        self.assertEqual(data["phase"], 94)
        self.assertEqual(data["priority"], 103)
        self.assertTrue(data["all_invariants_passed"])
        sgl = data["sglang_grok2_cpu_and_runtime"]
        self.assertEqual(sgl["fixed_presharded_offset"], 0)
        self.assertEqual(sgl["grammar_abort_http_status_after"], 500)
        vllm = data["vllm_uva_p2p_locality_step3"]
        self.assertEqual(vllm["p2p_failfast_speedup_x"], 3.0)
        self.assertEqual(vllm["lookup_failfast_speedup_x"], 3.75)
        self.assertEqual(vllm["remainder_sm_count"], 12)
        self.assertEqual(vllm["step3_new_matches_count"], 2)

    def test_lmcache_direct_copy_nixl_trace_megatron_stash_wgrad_te_titan_verl_mooncake_dynamo_phase94(self) -> None:
        data = self._run_and_load(
            "lmcache_direct_copy_nixl_trace_megatron_stash_wgrad_te_titan_verl_mooncake_dynamo_phase94"
        )
        self.assertEqual(data["phase"], 94)
        self.assertEqual(data["priority"], 103)
        self.assertTrue(data["all_invariants_passed"])
        lmc = data["lmcache_nixl_mooncake_dynamo"]
        self.assertEqual(lmc["sglang_component_format_id"], 18)
        self.assertEqual(lmc["putstart_1t_speedup_x"], 10.0)
        self.assertEqual(lmc["putstart_16t_speedup_x"], 6.54)
        meg = data["megatron_te_torchtitan_verl"]
        self.assertEqual(meg["warmup_peak_saved_gib"], 11.0)
        self.assertEqual(meg["sentinel_device_after"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
