import json
import pathlib
import subprocess
import sys
import unittest


class TestSpeculativeDecodingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "speculative_decoding_selection"
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
        self.assertAlmostEqual(by_id["standard_autoregressive_baseline"]["effective_tpot_bs4_ms"], 18.40)
        self.assertAlmostEqual(by_id["prompt_lookup_ngram_speculation"]["effective_tpot_bs4_ms"], 8.20)
        self.assertAlmostEqual(by_id["standalone_7b_draft_model"]["effective_tpot_bs4_ms"], 9.80)
        self.assertAlmostEqual(by_id["eagle3_mtp_feature_head"]["effective_tpot_bs4_ms"], 6.10)
        self.assertAlmostEqual(by_id["eagle3_mtp_feature_head"]["speedup_bs4_x"], 3.02)
        self.assertAlmostEqual(by_id["eagle3_mtp_feature_head"]["mean_accepted_tokens_code_rag"], 3.42)


if __name__ == "__main__":
    unittest.main()
