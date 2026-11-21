import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEmbeddingRerankerServingBGEM3TEIInfinitySelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "embedding_reranker_serving_bge_m3_nv_embed_infinity_tei_onnx_trt_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "embedding_reranker_serving_bge_m3_nv_embed_infinity_tei_onnx_trt_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_TEI_Infinity_Unpadded_FlashAttn_Length_Sorted_FP8"]["Rerank_Top50_P99_Latency_ms"],
            tiers["Tier_1_Naive_FastAPI_PyTorch_Padded_Batching_FP16"]["Rerank_Top50_P99_Latency_ms"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "embedding-reranker-serving-bge-m3-nv-embed-infinity-tei-onnx-trt-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "embedding-reranker-serving-bge-m3-nv-embed-infinity-tei-onnx-trt-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "embedding-reranker-serving-bge-m3-nv-embed-infinity-tei-onnx-trt-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "embedding-reranker-serving-bge-m3-nv-embed-infinity-tei-onnx-trt-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "embedding-reranker-serving-bge-m3-nv-embed-infinity-tei-onnx-trt-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
