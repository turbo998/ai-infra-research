import json
import pathlib
import subprocess
import sys
import unittest


class TestLongContextAttentionParallelismSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "long_context_attention_parallelism_selection"
        self.results_path = self.exp_dir / "results.json"

    def test_deterministic_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(self.exp_dir / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_provenance_and_metrics_integrity(self) -> None:
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertIn("not_measured", data["provenance"]["physical_gpu_cluster_measurement"])

        by_id = {s["id"]: s for s in data["scenarios"]}
        self.assertAlmostEqual(by_id["single_node_tp8_fa3_baseline"]["ttft_128k_sec"], 4.80)
        self.assertAlmostEqual(by_id["deepspeed_ulysses_sp8_nvlink"]["ttft_128k_sec"], 1.42)
        self.assertAlmostEqual(by_id["hybrid_usp_ulysses8_ring4_cp32"]["ttft_128k_sec"], 0.78)
        self.assertAlmostEqual(by_id["hybrid_usp_ulysses8_ring4_cp32"]["ttft_1m_sec"], 14.60)
        self.assertAlmostEqual(by_id["sparse_linear_hybrid_mla_nsa_cp16"]["ttft_1m_sec"], 3.20)
        self.assertAlmostEqual(by_id["sparse_linear_hybrid_mla_nsa_cp16"]["speedup_1m_x"], 77.50)


if __name__ == "__main__":
    unittest.main()
