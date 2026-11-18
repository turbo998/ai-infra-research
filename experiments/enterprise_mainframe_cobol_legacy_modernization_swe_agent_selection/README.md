# Enterprise Mainframe COBOL/CICS & Legacy Modernization SWE-Agent: Spec-First Graph, COMP-3 BigDecimal & Shadow Replay Evaluator

Deterministic evaluator comparing four mainframe (`COBOL / CICS / JCL / DB2`) and legacy enterprise system modernization architectures across 8.5M lines of legacy code, 5,900 programs/copybooks, and 25M daily core-ledger transactions:
- **Tier 1**: Mechanical Line-by-Line Transpiler (`JOBOL`) + Manual SI Rewrite (`82.0%` equivalence pass rate, `4.5%` decimal/REDEFINES regression rate, `18.0 KLOC/mo`, `$540,000/mo`)
- **Tier 2**: File-by-File Prompted LLM Directly Translating COBOL to Java/Python (`59.5%` pass rate, `18.4%` `COMP-3` penny-drift / `REDEFINES` regression rate, `110.0 KLOC/mo`, `$690,000/mo`)
- **Tier 3**: COBOL AST Chunk RAG + Self-Generated LLM Unit Tests (`84.6%` pass rate, `5.2%` regression rate, `65.0 KLOC/mo`, `$245,000/mo`)
- **Tier 4**: `JCL / CICS / Copybook` Dependency Graph + Spec-First Domain Extraction + Deterministic `COMP-3 -> BigDecimal` Type-Safe Compiler + Golden-Master Shadow Traffic Replay Agent (`99.9%` pass rate, `0.0%` penny-drift regression rate, `145.0 KLOC/mo`, `$82,000/mo`)

## Usage
```bash
python experiments/enterprise_mainframe_cobol_legacy_modernization_swe_agent_selection/run.py
python experiments/enterprise_mainframe_cobol_legacy_modernization_swe_agent_selection/run.py --check
```
