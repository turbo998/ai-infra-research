# Enterprise Legacy Code Modernization Factory (`COBOL / PL-SQL / Java 8 -> Java 21`) Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-faq.md`](../../docs/components/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection/run.py
python experiments/enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection/run.py --check
python -m unittest tests/test_enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection.py
```
