import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase181Experiments(unittest.TestCase):
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

    def test_gpu_software_stack_phase181(self) -> None:
        self._run_suite(
            "experiments/flashinfer_12row_rmsnorm_quadroute_moe_k1_bound_mhc_r200_fa4_autotune_aiter_hd256_phase181"
        )

    def test_inference_phase181(self) -> None:
        self._run_suite(
            "experiments/vllm_dsv4_packed_vision_sglang_warp_moe_trtllm_qkv_tiling_gb300_host_cache_dynamo_phase181"
        )

    def test_distributed_training_phase181(self) -> None:
        self._run_suite(
            "experiments/megatron_thd_cudagraph_minf_snapshot_dsv4_flops_te_mxfp8_rocshmem_rccl_mooncake_nixl_phase181"
        )


if __name__ == "__main__":
    unittest.main()
