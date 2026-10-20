import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase39Experiments(unittest.TestCase):
    def _run_check(self, rel_dir: str) -> dict:
        script = REPO_ROOT / rel_dir / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["status"], "ok")
        summary_path = REPO_ROOT / rel_dir / "summary.json"
        results_path = REPO_ROOT / rel_dir / "results" / "summary.json"
        self.assertTrue(summary_path.exists())
        self.assertTrue(results_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe(self) -> None:
        data = self._run_check("experiments/flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe")
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["sm120_minimax_h3_attention"]["fixed_total_proxy_hazards"], 0)
        self.assertEqual(data["sm120_minimax_h3_attention"]["fixed_unjoined_barrier1_rounds"], 0)
        self.assertEqual(data["cake_nvfp4_svdquant_gemm"]["converged_program_count"], 15)
        self.assertAlmostEqual(data["cake_nvfp4_svdquant_gemm"]["gb300_max_speedup_vs_prev_cake"], 3.58)

    def test_sglang_dspark_default_radix_split_storage_load_zmq(self) -> None:
        data = self._run_check("experiments/sglang_dspark_default_radix_split_storage_load_zmq")
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["waiting_storage_load"]["fixed_waiting_uncached_tokens"], 3328)
        self.assertEqual(data["radix_split_action"]["fixed_buffer_only_orphaned_deps"], 0)

    def test_torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp(self) -> None:
        data = self._run_check("experiments/torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp")
        self.assertTrue(all(data["invariants"].values()))
        self.assertAlmostEqual(data["torchtitan_dist_moe_mega_swiglu"]["max_mega_swiglu_speedup_vs_bf16"], 1.72)
        self.assertEqual(data["te_cublasmp_and_linear_attention"]["cublasmp_0_11_linear_bwd_allgathers"], 0)

    def test_phase39_bilingual_docs_and_cards_exist(self) -> None:
        expected_docs = [
            "docs/components/flashinfer-minimax-h3-sm120-epilogue-nvfp4-svdquant-moe-stream.md",
            "docs/components/flashinfer-minimax-h3-sm120-epilogue-nvfp4-svdquant-moe-stream.en.md",
            "docs/components/sglang-dspark-sim-default-radix-split-storage-hit-zmq-allowlist.md",
            "docs/components/sglang-dspark-sim-default-radix-split-storage-hit-zmq-allowlist.en.md",
            "docs/components/torchtitan-dist-moe-swiglu-rl-cp-te-cublasmp-wgrad-megatron-mfsdp.md",
            "docs/components/torchtitan-dist-moe-swiglu-rl-cp-te-cublasmp-wgrad-megatron-mfsdp.en.md",
            "docs/reports/ai-infra-phase39-benchmark-cards.html",
        ]
        for rel in expected_docs:
            p = REPO_ROOT / rel
            self.assertTrue(p.exists(), f"Missing Phase 39 asset: {rel}")


if __name__ == "__main__":
    unittest.main()
