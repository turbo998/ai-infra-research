import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase180Experiments(unittest.TestCase):
    def _run_suite(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd= str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"{rel_dir} failed:\n{proc.stderr}\n{proc.stdout}")

    def test_gpu_software_stack_phase180(self) -> None:
        self._run_suite(
            "experiments/flashinfer_mla_fp8_gdn_sm107a_mhc_fa_gb300_triton_tilelang_phase180"
        )

    def test_inference_phase180(self) -> None:
        self._run_suite(
            "experiments/vllm_grpc_readiness_sglang_dsv41_tp_trtllm_dep4_dynamo_sticky_phase180"
        )

    def test_distributed_training_phase180(self) -> None:
        self._run_suite(
            "experiments/megatron_offload_fsdp_inductor_lanes_mooncake_engram_lmcache_phase180"
        )


if __name__ == "__main__":
    unittest.main()
