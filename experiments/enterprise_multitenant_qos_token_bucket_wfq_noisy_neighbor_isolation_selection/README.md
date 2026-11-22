# Enterprise Multi-Tenant QoS, Dual-Rate CIR/EIR WCU Token Bucket & Preemptive WFQ Selection (`#493`)

Reproducible analytical model comparing 4 multi-tenant rate-limiting and scheduling architectures across a 64-GPU shared enterprise LLM cluster serving 50 departments:
- **Tier 1 (`Simple RPS Limit / FIFO`)**: Long `32K` batch prompts bypass RPS limits and starve Tier-0 VIP traffic (`18,400ms P99 TTFT`).
- **Tier 2 (`Rigid Static Hard TPM Quota`)**: Protects VIP latency (`340ms`) but strands `53.5%` of cluster GPUs (`46.5%` utilization).
- **Tier 3 (`Raw TPM Token Bucket`)**: Blind to `4x` output token cost and `0.1x` cached prefix cost (`2,850ms` VIP tail spikes).
- **Tier 4 (`Dual-Rate CIR/EIR WCU Token Bucket + Preemptive Priority WFQ`)**: Charges Virtual Compute Units (`WCU`), guarantees `CIR` green traffic (`325ms` VIP P99 TTFT), and lets `EIR` yellow bursts borrow idle GPUs (`88.5%` cluster utilization).

## Run
```bash
python experiments/enterprise_multitenant_qos_token_bucket_wfq_noisy_neighbor_isolation_selection/run.py
```
