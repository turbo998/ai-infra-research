from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiGatewayTokenQuotaChargebackApimKongLitellmEnvoySelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_gateway_token_quota_chargeback_apim_kong_litellm_envoy_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["chargeback_error_rate_pct"], 0.1)
        self.assertGreater(data["tiers"][3]["shared_pool_effective_utilization_pct"], 88.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "ai-gateway-token-quota-chargeback-apim-kong-litellm-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
