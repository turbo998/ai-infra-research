from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestCudaGraphPiecewiseTorchCompileVllmV1MemoryPoolSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "cuda_graph_piecewise_torch_compile_vllm_v1_memory_pool_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["mixed_chunked_prefill_decode_step_ms"], 15.0)
        self.assertLess(data["tiers"][3]["cuda_graph_memory_pool_gb_per_gpu"], 2.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "cuda-graph-piecewise-torch-compile-vllm-v1-memory-pool-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "cuda-graph-piecewise-torch-compile-vllm-v1-memory-pool-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "cuda-graph-piecewise-torch-compile-vllm-v1-memory-pool-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "cuda-graph-piecewise-torch-compile-vllm-v1-memory-pool-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "cuda-graph-piecewise-torch-compile-vllm-v1-memory-pool-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
