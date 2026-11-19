# Enterprise Multi-Tenant SaaS AI Tiered Rate-Limiting (`GCRA`), Asymmetric I/O Compute Credits & Burst-Borrowing (`R456`)

Deterministic evaluation suite comparing four multi-tenant B2B SaaS AI rate-limiting and quota-governance architectures across `5,000` enterprise tenant organizations sharing a `64-GPU (H100)` cluster (`12` API gateway replicas, `85M tokens/min` peak throughput):

1. **Tier 1**: HTTP Request-Per-Minute (`RPM`) Fixed-Window Counter (`4,120 ms` noisy-neighbor P99 TTFT, `24.5%` false 429 on cached RAG, `42.0%` GPU utilization, `$165,000/mo` GPU TCO)
2. **Tier 2**: Flat Unweighted Total-Token-Per-Minute (`TPM`) Hard Cap (`2,680 ms` P99 TTFT, `19.2%` false 429 on cached RAG, `51.0%` GPU utilization, `$138,000/mo` GPU TCO)
3. **Tier 3**: Local Gateway Token-Bucket Without Pre-Flight Escrow (`1,390 ms` P99 TTFT, `8.4%` false 429, `6.2%` concurrent stream overrun, `$104,000/mo` GPU TCO)
4. **Tier 4**: Distributed Redis-Cell `GCRA` + Asymmetric Weighted Compute-Credit (`0.15x/1x/4x`) + Work-Conserving Burst-Borrowing Queue (`310 ms` P99 TTFT, `0.2%` false 429, `84.5%` GPU utilization, `0.0%` overrun leakage, `$62,000/mo` GPU TCO)

## Usage

```bash
python experiments/enterprise_saas_ai_tiered_rate_limiting_gcra_token_credit_burst_selection/run.py
python experiments/enterprise_saas_ai_tiered_rate_limiting_gcra_token_credit_burst_selection/run.py --check
python -m unittest tests/test_enterprise_saas_ai_tiered_rate_limiting_gcra_token_credit_burst_selection.py
```
