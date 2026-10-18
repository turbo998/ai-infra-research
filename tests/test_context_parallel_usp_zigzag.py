import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.context_parallel_usp_zigzag import run as usp_mod


class TestContextParallelUspZigzag(unittest.TestCase):
    def test_zigzag_balance_proof_is_exact(self) -> None:
        proof = usp_mod.verify_zigzag_chunk_balance(ring_degree=8, seq_len=1048576)
        self.assertEqual(proof["zigzag_max_to_avg_ratio"], 1.0)
        self.assertEqual(proof["zigzag_min_to_max_ratio"], 1.0)
        self.assertEqual(proof["zigzag_algorithmic_bubble_pct"], 0.0)
        self.assertGreater(proof["contiguous_idle_bubble_pct"], 45.0)
        self.assertEqual(proof["zigzag_rank_subchunk_pairs"][0], [0, 15])
        self.assertEqual(proof["zigzag_rank_subchunk_pairs"][7], [7, 8])

    def test_usp_zigzag_beats_naive_ring_and_ulysses64(self) -> None:
        results = usp_mod.generate_results()
        findings = results["headline_findings"]
        self.assertGreaterEqual(findings["speedup_1m_vs_naive_ring"], 2.8)
        self.assertGreaterEqual(findings["speedup_4m_vs_naive_ring"], 2.8)
        self.assertGreater(findings["speedup_1m_vs_ulysses64"], 1.05)
        self.assertLess(findings["usp_zigzag_residual_imbalance_pct"], 0.2)
        self.assertGreater(findings["usp_zigzag_1m_efficiency_pct"], 99.0)

    def test_cli_check_passes(self) -> None:
        script = Path(usp_mod.__file__).resolve()
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        data = json.loads(usp_mod.RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment_id"], "context_parallel_usp_zigzag")


if __name__ == "__main__":
    unittest.main()
