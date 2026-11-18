# Enterprise Confidential AI (CPU+GPU TEE, Remote Attestation & HSM Key Release) Evaluator 

Deterministic analytical evaluator comparing four enterprise deployment architectures for high-security 70B LLM workloads (`100 RPS` peak):

1. **Tier 1 (`tier1_standard_cloud_gpu_plaintext_ram`)**: Standard Cloud GPU VM (`0.0%` throughput overhead, `1 day` deployment, `0.0%` `Data-in-Use` memory protection, `$24,000/mo`).
2. **Tier 2 (`tier2_airgapped_onprem_datacenter`)**: 100% Air-Gapped On-Premises Self-Built Data Center (`0.0%` overhead, `270 days` [9 months] procurement/retrofit lead time, `95.0%` protection [still exposed to local root insiders], `$62,000/mo` 3-year amortized TCO).
3. **Tier 3 (`tier3_cloud_cvm_gpu_cc_basic`)**: Cloud Confidential VM (`AMD SEV-SNP / Intel TDX`) + NVIDIA H100/H200 Confidential Computing Mode (`5.4%` PCIe AES-GCM-256 overhead, `2 days` deployment, `92.0%` protection, `$26,200/mo`).
4. **Tier 4 (`tier4_confidential_ai_remote_attestation_skr`)**: Zero-Trust Confidential AI Enclave (`CPU+GPU TEE` + Cryptographic Remote Attestation + HSM Secure Key Release [SKR], `5.8%` throughput overhead, `2 days` deployment [`135x` faster than On-Prem], `100.0%` `Data-in-Use` & `Model-IP-in-Use` protection, `$27,800/mo` [`-55.2%` TCO vs. On-Prem]).

## Usage

```bash
python experiments/enterprise_confidential_ai_tee_gpu_attestation_selection/run.py
python experiments/enterprise_confidential_ai_tee_gpu_attestation_selection/run.py --check
```
