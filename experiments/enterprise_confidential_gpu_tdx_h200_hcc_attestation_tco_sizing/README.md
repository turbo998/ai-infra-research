# Enterprise Sovereign & Confidential AI (`Intel TDX / AMD SEV-SNP + NVIDIA H200 HCC + NRAS Attestation + HSM SKR`) vs. Air-Gapped On-Prem TCO Sizer (`#538`)

- **Chinese FAQ**: [`../../docs/components/enterprise-confidential-gpu-tdx-h200-hcc-attestation-tco-sizing-faq.md`](../../docs/components/enterprise-confidential-gpu-tdx-h200-hcc-attestation-tco-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-confidential-gpu-tdx-h200-hcc-attestation-tco-sizing-faq.en.md`](../../docs/components/enterprise-confidential-gpu-tdx-h200-hcc-attestation-tco-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/confidential-gpu-tdx-h200-hcc-attestation-lab.html`](../../docs/h5/confidential-gpu-tdx-h200-hcc-attestation-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-confidential-gpu-tdx-h200-hcc-attestation-tco-sizing-sizer.html`](../../docs/reports/enterprise-confidential-gpu-tdx-h200-hcc-attestation-tco-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-confidential-gpu-tdx-h200-hcc-attestation-tco-sizing-cards.html`](../../docs/reports/enterprise-confidential-gpu-tdx-h200-hcc-attestation-tco-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_confidential_gpu_tdx_h200_hcc_attestation_tco_sizing/run.py
python -m pytest tests/test_enterprise_confidential_gpu_tdx_h200_hcc_attestation_tco_sizing.py -q
```
