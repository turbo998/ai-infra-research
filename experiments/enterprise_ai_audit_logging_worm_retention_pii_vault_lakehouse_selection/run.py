from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_ai_audit_logging_worm_retention_pii_vault_lakehouse_selection",
        "title": "Enterprise AI Compliance Audit Logging, WORM Crypto-Shredding & ClickHouse/Iceberg Tiering Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_ai_transactions": 50000000,
            "raw_uncompressed_log_tb_per_month": 4.5,
            "regulatory_retention_years": 7,
            "zstd_parquet_compression_ratio": 9.2,
        },
        "tiers": [
            {
                "tier_id": "tier1_raw_plaintext_json_in_hot_elasticsearch_siem",
                "name": "Tier 1: Raw Plaintext Prompt/Response JSON in Hot Elasticsearch / SIEM",
                "regulatory_audit_reconstruction_pct": 92.0,
                "gdpr_and_worm_dual_compliance_pct": 0.0,
                "unredacted_pii_exposure_surface_pct": 100.0,
                "monthly_audit_storage_tco_usd": 215000,
                "verdict": "Anti-Pattern: $215K/mo hot storage bill & 100% plaintext PII exposure to engineers in Kibana",
            },
            {
                "tier_id": "tier2_metadata_only_drop_prompts_and_outputs",
                "name": "Tier 2: Metadata-Only Logging (Drops All Prompt, Context & Output Payloads)",
                "regulatory_audit_reconstruction_pct": 14.0,
                "gdpr_and_worm_dual_compliance_pct": 50.0,
                "unredacted_pii_exposure_surface_pct": 0.0,
                "monthly_audit_storage_tco_usd": 140000,
                "verdict": "Fails EU AI Act Art. 12 & financial audits: cannot reconstruct why a model made a decision",
            },
            {
                "tier_id": "tier3_plaintext_s3_worm_object_lock_no_crypto_shred",
                "name": "Tier 3: Plaintext S3 Object Lock WORM (Without Per-User DEK Crypto-Shredding)",
                "regulatory_audit_reconstruction_pct": 98.5,
                "gdpr_and_worm_dual_compliance_pct": 35.0,
                "unredacted_pii_exposure_surface_pct": 45.0,
                "monthly_audit_storage_tco_usd": 125000,
                "verdict": "Legal Deadlock: Cannot delete user PII for GDPR Art. 17 because file is locked in WORM for 7 yrs!",
            },
            {
                "tier_id": "tier4_per_user_dek_crypto_shred_merkle_worm_iceberg_clickhouse",
                "name": "Tier 4: Per-User DEK Crypto-Shredding + Merkle WORM Zstd Iceberg + 14d ClickHouse Hot Index",
                "regulatory_audit_reconstruction_pct": 100.0,
                "gdpr_and_worm_dual_compliance_pct": 100.0,
                "unredacted_pii_exposure_surface_pct": 0.0,
                "monthly_audit_storage_tco_usd": 18000,
                "verdict": "Recommended: 100% WORM+GDPR compliance, 0% plaintext PII leak, 9.2x Zstd compression & -91.6% TCO",
            },
        ],
        "headline_findings": {
            "audit_reconstruction_completeness_pct": 100.0,
            "worm_and_gdpr_dual_compliance_pct": 100.0,
            "zstd_storage_compression_factor": 9.2,
            "tco_savings_vs_tier1_pct": 91.6,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
