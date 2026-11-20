# Enterprise Non-Prod AI Sandbox, Referential Synthetic Data & `VCR` Cassette Replay Selection (`R511`)

Deterministic quantitative evaluation model comparing four enterprise non-production (`Dev / Staging / CI`) AI sandbox and test-data architectures across **200 developers and 1,500 daily CI test suites**:
- **Tier 1**: Cloning Raw Production Databases & Chat Logs Directly Into `Dev / Staging` Environments
- **Tier 2**: Random Regex Masking (`XXXX-XXXX` / Faker Without Referential Integrity) + 100% Live Cloud LLM Calls in CI
- **Tier 3**: Static Hand-Written Mock JSON Responses Only (Zero Live Drift Detection or Realistic Distribution)
- **Tier 4**: Deterministic Keyed-HMAC Referential Pseudonymization + Synthetic Edge-Case Generator + Gateway `VCR Cassette Replay` (`85% Cached / 15% Live SLM/Canary`)

## Verification

```bash
python experiments/enterprise_ai_sandbox_synthetic_test_data_anonymization_staging_selection/run.py --check
python -m unittest tests/test_enterprise_ai_sandbox_synthetic_test_data_anonymization_staging_selection.py
```
