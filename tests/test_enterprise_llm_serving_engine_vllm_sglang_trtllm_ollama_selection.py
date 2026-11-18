"""Unit tests for Enterprise LLM Inference Engine Architect Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_llm_serving_engine_vllm_sglang_trtllm_ollama_selection.run import (
    evaluate_enterprise_llm_serving_engine_vllm_sglang_trtllm_ollama,
)


class TestEnterpriseLlmServingEngineVllmSglangTrtllmOllamaSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_llm_serving_engine_vllm_sglang_trtllm_ollama()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_llm_serving_engine_vllm_sglang_trtllm_ollama_selection",
        )
        self.assertEqual(data["round"], 367)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_sglang_radixattention_xgrammar_agent_moe_king"]["effective_throughput_tok_s"],
            tiers["tier1_ollama_llamacpp_gguf_local_workstation"]["effective_throughput_tok_s"],
        )
        self.assertLess(
            tiers["tier3_vllm_v1_universal_enterprise_maas_workhorse"]["model_onboarding_days"],
            tiers["tier2_nvidia_tensorrt_llm_static_compiled_engine"]["model_onboarding_days"],
        )
        self.assertGreater(
            data["summary"]["tier4_sglang_vs_tier1_ollama_throughput_x"],
            10.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_llm_serving_engine_vllm_sglang_trtllm_ollama_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_llm_serving_engine_vllm_sglang_trtllm_ollama()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
