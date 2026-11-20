# Enterprise AI Data Governance, Sensitivity Labeling (`Purview + Unity Catalog`) & Corpus Lineage Selection (`R498`)

Deterministic quantitative evaluation model comparing four enterprise AI data governance architectures across **25M unstructured documents + 12,000 Lakehouse tables + 45,000 monthly ACL/GDPR lifecycle events**:
- **Tier 1**: Unclassified "Dump Everything into Vector DB & SFT JSONL" Anti-Pattern
- **Tier 2**: Manual Spreadsheet Approval & Static Folder-Level Whitelisting
- **Tier 3**: Siloed Lakehouse Catalog Only (`Unity Catalog / Iceberg`) Without M365/Unstructured Document Label Propagation
- **Tier 4**: Federated `Microsoft Purview + Databricks Unity Catalog` + Chunk-Inherited Sensitivity Labels (`L1–L4`) + Tombstone CDC Unlearning (`<15m`) + Zero-SFT-on-PII Gate

## Verification

```bash
python experiments/enterprise_ai_data_governance_purview_unity_catalog_lineage_dlp_selection/run.py --check
python -m unittest tests/test_enterprise_ai_data_governance_purview_unity_catalog_lineage_dlp_selection.py
```
