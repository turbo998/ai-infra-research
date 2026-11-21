# Confidential AI & GPU TEE Architecture: NVIDIA H200/Blackwell CC-On, PPCIE Multi-GPU NVLink & Remote Attestation Benchmark

Deterministic quantitative evaluation model for [`docs/components/confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-selection-faq.md`](../../docs/components/confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-selection-faq.md) and [`.en.md`](../../docs/components/confidential-ai-gpu-tee-h200-blackwell-ppcie-attestation-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/confidential_ai_gpu_tee_h200_blackwell_ppcie_attestation/run.py
python experiments/confidential_ai_gpu_tee_h200_blackwell_ppcie_attestation/run.py --check
python -m unittest tests/test_confidential_ai_gpu_tee_h200_blackwell_ppcie_attestation_selection.py
```
