import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase53Experiments(unittest.TestCase):
    def _run_experiment(self, slug: str) -> dict:
        exp_dir = ROOT / "experiments" / slug
        script = exp_dir / "run.py"
        summary_file = exp_dir / "summary.json"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(proc.returncode, 0)
        data = json.loads(summary_file.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment_id"], slug)
        self.assertEqual(data["phase"], 53)
        self.assertEqual(data["priority"], 62)
        self.assertTrue(all(data["invariants"].values()))
        return data

    def test_flashinfer_cake_fp4_ar_mnnvl_fp32_sage_fp8_bsa_trtllm_moe_ws_k3_situ(self) -> None:
        data = self._run_experiment(
            "flashinfer-cake-fp4-ar-mnnvl-fp32-sage-fp8-bsa-trtllm-moe-ws-k3-situ"
        )
        self.assertEqual(
            data["cake_fp4_allreduce_per_arch"]["sm100_nvfp4_to_nvfp4_geomean"], 1.0718
        )
        self.assertEqual(
            data["mnnvl_oneshot_fp32_square"]["rl_val_score_after"], 0.698
        )
        self.assertEqual(
            data["trtllm_moe_workspace_and_k3_situ"]["trtllm_moe_workspace"]["memory_saved_mib"],
            2683.1,
        )

    def test_sglang_dsv41_mxfp8_comfyui_flux_trtllm_exec_vllm_bert_lora(self) -> None:
        data = self._run_experiment(
            "sglang-dsv41-mxfp8-comfyui-flux-trtllm-exec-vllm-bert-lora"
        )
        self.assertEqual(
            data["sglang_dsv41_mxfp8_and_flux_controlnet"]["tuned_bucket_count_after"], 14
        )
        self.assertEqual(
            data["trtllm_executor_cleanup_and_bolt"]["removed_mpi_tags"], 35
        )
        self.assertEqual(
            data["vllm_bert_lora_nixl_and_rocm_eagle3"]["request_sliced_shapes"],
            [[12, 3], [8, 9], [15, 5]],
        )

    def test_te_grouped_mlp_align_megatron_sched_titan_rope32_nixl_proxy_mooncake(self) -> None:
        data = self._run_experiment(
            "te-grouped-mlp-align-megatron-sched-titan-rope32-nixl-proxy-mooncake"
        )
        self.assertEqual(
            data["megatron_scheduler_and_torchtitan_rope32"]["llama32_3b_freqs_off_before"],
            "35/64",
        )
        self.assertEqual(
            data["nixl_ep_proxy_dynamo_stop_and_mooncake"]["dynamo_trtllm_merged_stop_ids"],
            [2, 7, 200002, 200012],
        )


if __name__ == "__main__":
    unittest.main()
