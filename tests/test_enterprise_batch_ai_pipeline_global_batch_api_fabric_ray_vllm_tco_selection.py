from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseBatchAiPipelineGlobalBatchApiFabricRayVllmTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_batch_ai_pipeline_global_batch_api_fabric_ray_vllm_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["prefix_kv_hit_rate_pct"], 95.0)
        self.assertGreater(data["tiers"][3]["per_gpu_effective_tok_s"], 7000)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-batch-ai-pipeline-global-batch-api-fabric-ray-vllm-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-batch-ai-pipeline-global-batch-api-fabric-ray-vllm-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-batch-ai-pipeline-global-batch-api-fabric-ray-vllm-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "batch-ai-global-batch-api-fabric-ray-vllm-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-batch-ai-pipeline-global-batch-api-fabric-ray-vllm-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-batch-ai-pipeline-global-batch-api-fabric-ray-vllm-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
