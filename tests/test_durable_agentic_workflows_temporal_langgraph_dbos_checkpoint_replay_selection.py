import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestDurableAgenticWorkflowsTemporalLangGraphSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "durable_agentic_workflows_temporal_langgraph_dbos_checkpoint_replay_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "durable_agentic_workflows_temporal_langgraph_dbos_checkpoint_replay_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_Temporal_Restate_Durable_Execution_Plus_LangGraph"]["End_To_End_25Step_Success_Rate_Pct"],
            tiers["Tier_1_Stateless_In_Memory_While_Loop_FastAPI"]["End_To_End_25Step_Success_Rate_Pct"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "durable-agentic-workflows-temporal-langgraph-dbos-checkpoint-replay-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "durable-agentic-workflows-temporal-langgraph-dbos-checkpoint-replay-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "durable-agentic-workflows-temporal-langgraph-dbos-checkpoint-replay-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "durable-agentic-workflows-temporal-langgraph-dbos-checkpoint-replay-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "durable-agentic-workflows-temporal-langgraph-dbos-checkpoint-replay-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
