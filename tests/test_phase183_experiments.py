import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase183Experiments(unittest.TestCase):
    def _run_suite(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"{rel_dir} failed:\n{proc.stderr}\n{proc.stdout}")

    def test_gpu_software_stack_phase183(self) -> None:
        self._run_suite(
            "experiments/flashinfer_kda_dbuf_vsa_sm120_fp8_bn256_fa4_mla_2cta_pad_triton_fp6_aiter_dsv4_route_phase183"
        )

    def test_inference_phase183(self) -> None:
        self._run_suite(
            "experiments/trtllm_helix_cp_conv1d_64b_vllm_moe_restorer_eplb_sglang_qwen3_qknorm_dflash_dynamo_router_phase183"
        )

    def test_distributed_training_phase183(self) -> None:
        self._run_suite(
            "experiments/megatron_batch_inv_router_dcp_v2_te_nvfp4_4over6_pytorch_philox_rccl_tdm_mooncake_uring_nixl_phase183"
        )


if __name__ == "__main__":
    unittest.main()
