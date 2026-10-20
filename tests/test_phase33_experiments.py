import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase33Experiments(unittest.TestCase):
    def _run_and_load(self, exp_name: str) -> dict:
        script = ROOT / "experiments" / exp_name / "run.py"
        summary_path = ROOT / "experiments" / exp_name / "results" / "summary.json"
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=ROOT)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_gdn_materialize_megamoe_fc2_fmha_l2(self) -> None:
        data = self._run_and_load("flashinfer_gdn_materialize_megamoe_fc2_fmha_l2")
        self.assertEqual(data["phase"], 33)
        self.assertEqual(data["priority"], 42)
        self.assertTrue(all(data["invariants"].values()))
        gdn = data["gdn_prefix_materialize"]
        self.assertEqual(gdn["replayed_count_c"], 8)
        self.assertLess(gdn["max_abs_err_partial_vs_recurrent"], 1e-12)
        self.assertEqual(gdn["active_0_overhead_reduction_factor"], 12.8)
        megamoe = data["cake_w8a8_megamoe_ep16_fc2"]
        self.assertEqual(megamoe["l2_cache_hint_descriptor_hex"], "0x12F0000000000000ULL")
        self.assertAlmostEqual(megamoe["geomean_layer_latency_ratio"], 0.9912, places=3)

    def test_sglang_glm52_dsa_mlaprolog_megatron_meta_embed(self) -> None:
        data = self._run_and_load("sglang_glm52_dsa_mlaprolog_megatron_meta_embed")
        self.assertEqual(data["phase"], 33)
        self.assertEqual(data["priority"], 42)
        self.assertTrue(all(data["invariants"].values()))
        dsa = data["sglang_compact_dsa_indexer"]
        self.assertEqual(dsa["indexer_memory_reduction_pct"], 75.0)
        self.assertGreater(dsa["compact_max_kv_tokens"], dsa["legacy_max_kv_tokens"])
        mla = data["sglang_dynamic_int8_mlaprolog"]["dispatch_cases"]
        self.assertIsNone(mla[0]["weight_quant_mode"])
        self.assertEqual(mla[1]["weight_quant_mode"], 2)
        self.assertFalse(mla[1]["free_w_kc_on_decode"])
        self.assertTrue(mla[1]["recompute_bf16_q_lora_for_indexer"])

    def test_mooncake_hotcache_race_dynamo_kvhash_prefill_gate(self) -> None:
        data = self._run_and_load("mooncake_hotcache_race_dynamo_kvhash_prefill_gate")
        self.assertEqual(data["phase"], 33)
        self.assertEqual(data["priority"], 42)
        self.assertTrue(all(data["invariants"].values()))
        mc = data["mooncake_hotcache_scan"]
        self.assertEqual(mc["legacy_racy_key_reads"], 16000)
        self.assertEqual(mc["fixed_racy_key_reads"], 0)
        kvh = data["dynamo_inplace_kv_hashing"]["scenarios"]
        self.assertEqual(kvh[0]["inplace_token_heap_bytes"], 0)
        self.assertEqual(kvh[0]["heap_bytes_saved"], 1048576)
        self.assertEqual(kvh[2]["inplace_token_heap_bytes"], 416)
        router = data["dynamo_prefill_fail_closed"]["transitions"]
        self.assertEqual(router[1]["activation_snapshot"], "Pending")
        self.assertEqual(router[1]["http_status"], 503)


if __name__ == "__main__":
    unittest.main()
