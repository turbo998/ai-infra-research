import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseEcommerceCatalogEnrichmentSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ecommerce_product_catalog_enrichment_taxonomy_vlm_dedup_selection"
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
        t1 = tiers["tier1_unconstrained_frontier_vlm_every_sku"]
        t4 = tiers["tier4_topdown_taxonomy_xgrammar_tiered_ocr_vlm_siglip_dedup"]
        self.assertEqual(t4["facet_dictionary_compliance_pct"], 100.0)
        self.assertGreater(t4["attribute_extraction_recall_pct"], 97.0)
        self.assertGreater(t4["cross_seller_duplicate_sku_catch_pct"], 99.0)
        self.assertLess(
            t4["monthly_catalog_enrichment_tco_usd"],
            t1["monthly_catalog_enrichment_tco_usd"] * 0.12,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-ecommerce-product-catalog-enrichment-taxonomy-vlm-dedup"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
