import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase27Experiments(unittest.TestCase):
    def _run_and_load(self, exp_name: str) -> dict:
        script = ROOT / "experiments" / exp_name / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=str(ROOT))
        summary_path = ROOT / "experiments" / exp_name / "results" / "summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_cake_gemm_lamport_allreduce(self) -> None:
        data = self._run_and_load("flashinfer_cake_gemm_lamport_allreduce")
        self.assertEqual(data["route_count"], 12)
        decode = {r["m"]: r for r in data["decode_band_analysis"]}
        self.assertEqual(decode[16]["route_mode"], "one-shot")
        self.assertEqual(decode[32]["route_mode"], "two-shot")
        self.assertEqual(decode[128]["dynamic_smem_bytes"], 214528)
        self.assertTrue(data["lamport_determinism_and_sentinel_check"]["cake_bitwise_identical"])
        self.assertFalse(data["lamport_determinism_and_sentinel_check"]["sanitized_dirty_sentinel_collision"])
        self.assertEqual(
            data["tcgen05_synchronization_checks"]["tmem_wait_ld_fix"]["persistent_programs_patched"],
            39,
        )

    def test_torchtitan_kimi_k3_mxfp4_adapter_fsdp(self) -> None:
        data = self._run_and_load("torchtitan_kimi_k3_mxfp4_adapter_fsdp")
        self.assertTrue(data["kimi_k3_kda_mxfp4_roundtrip"]["a_log_exact_roundtrip"])
        self.assertTrue(data["kimi_k3_kda_mxfp4_roundtrip"]["dtensor_layout_restored"])
        self.assertEqual(data["deepseek_v3_safetensors_index"]["saved_entries_num_mtp_0"], 45395)
        self.assertEqual(data["deepseek_v3_safetensors_index"]["saved_entries_num_mtp_1"], 46183)
        self.assertEqual(
            data["simple_fsdp_meta_inspection"]["with_disable_active_parametrization"]["passes_completed"],
            2,
        )

    def test_vllm_umbp_kv_connector_slime_rewind(self) -> None:
        data = self._run_and_load("vllm_umbp_kv_connector_slime_rewind")
        umbp = data["vllm_umbp_verification"]
        self.assertTrue(umbp["dcp_partitions_tp_without_cartesian_blowup"])
        self.assertEqual(umbp["max_loadable_tokens_standard"], 192)
        self.assertEqual(umbp["max_loadable_tokens_eagle"], 128)
        slime = data["slime_straw_rewind_verification"]
        self.assertEqual(slime["recovered_epoch_id"], 1)
        self.assertEqual(slime["recovered_sample_offset"], 95)
        self.assertEqual(slime["selected_safe_boundary_step"], 10)


if __name__ == "__main__":
    unittest.main()
