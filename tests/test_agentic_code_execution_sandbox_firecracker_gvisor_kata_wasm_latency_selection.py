import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAgenticCodeExecutionSandboxFirecrackerGVisorSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "agentic_code_execution_sandbox_firecracker_gvisor_kata_wasm_latency_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "agentic_code_execution_sandbox_firecracker_gvisor_kata_wasm_latency_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Firecracker_MicroVM_Snapshot_Restore_CoW"]["Sandbox_Cold_Start_Or_Resume_ms"],
            tiers["Tier_2_Cold_Boot_VM_Or_Uncached_Kata_Container"]["Sandbox_Cold_Start_Or_Resume_ms"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "agentic-code-execution-sandbox-firecracker-gvisor-kata-wasm-latency-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "agentic-code-execution-sandbox-firecracker-gvisor-kata-wasm-latency-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "agentic-code-execution-sandbox-firecracker-gvisor-kata-wasm-latency-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "agentic-code-execution-sandbox-firecracker-gvisor-kata-wasm-latency-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "agentic-code-execution-sandbox-firecracker-gvisor-kata-wasm-latency-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
