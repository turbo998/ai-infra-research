import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase182Experiments(unittest.TestCase):
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

    def test_gpu_software_stack_phase182(self) -> None:
        self._run_suite(
            "experiments/flashinfer_cake_moe_v3_gdn_wy5cta_fa4_clc_aiter_cluster_mla_phase182"
        )

    def test_inference_phase182(self) -> None:
        self._run_suite(
            "experiments/trtllm_kimi_k3_pdl_attn_res_vllm_sw_divisor_sglang_strided_q_phase182"
        )

    def test_distributed_training_phase182(self) -> None:
        self._run_suite(
            "experiments/megatron_moe_rl_parity_rccl_gfx1250_dda_mooncake_hip_ipc_phase182"
        )


if __name__ == "__main__":
    unittest.main()
