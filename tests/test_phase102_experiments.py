import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase102Experiments(unittest.TestCase):
    def _run_suite(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=ROOT)

    def test_flashinfer_mm_fp4_fold_spec_res_ulysses_phase102(self) -> None:
        self._run_suite("experiments/flashinfer_mm_fp4_fold_spec_res_ulysses_phase102")

    def test_sglang_paged_experts_pd_ctx_vllm_grpc_phase102(self) -> None:
        self._run_suite("experiments/sglang_paged_experts_pd_ctx_vllm_grpc_phase102")

    def test_nixl_libfabric_reestablish_te_nvfp4_row_phase102(self) -> None:
        self._run_suite("experiments/nixl_libfabric_reestablish_te_nvfp4_row_phase102")


if __name__ == "__main__":
    unittest.main()
