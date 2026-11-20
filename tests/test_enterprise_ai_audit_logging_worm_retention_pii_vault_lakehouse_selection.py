import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiAuditLoggingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_audit_logging_worm_retention_pii_vault_lakehouse_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_script_check_and_invariants(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"], "synthetic + anonymized_reported")
        self.assertFalse(data["independently_reproduced"])
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_raw_plaintext_json_in_hot_elasticsearch_siem"]
        t4 = tiers["tier4_per_user_dek_crypto_shred_merkle_worm_iceberg_clickhouse"]
        self.assertEqual(t4["regulatory_audit_reconstruction_pct"], 100.0)
        self.assertEqual(t4["gdpr_and_worm_dual_compliance_pct"], 100.0)
        self.assertEqual(t4["unredacted_pii_exposure_surface_pct"], 0.0)
        self.assertLess(
            t4["monthly_audit_storage_tco_usd"],
            t1["monthly_audit_storage_tco_usd"] * 0.1,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-ai-audit-logging-worm-retention-pii-vault-lakehouse"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
