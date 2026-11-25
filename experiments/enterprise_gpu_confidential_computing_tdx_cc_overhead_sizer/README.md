# Enterprise GPU Confidential Computing (`H100/H200/B200 CC-On + TDX/SEV-SNP`) Overhead & Bounce-Buffer Sizer (`#666`)

Reproducible analytical model quantifying why GPU Confidential Computing (`CC-On`) adds only `1.8%–2.1%` overhead on single-GPU inference (since `HBM3e` is inside the trusted silicon boundary and only `18.4 KB` of token IDs cross PCIe `AES-GCM-256` bounce buffers), why naive multi-GPU `TP=8` over PCIe bounce buffers suffers a `28.5%` throughput loss (`56 → 14.8 GB/s`), and how enabling **Protected PCIe / NVLink CC (`PPCIE`) + `FP8` KV Cache** cuts `TP=8` overhead to `4.8%` (`46 → 34× H200`, `-26.1%` GPUs).

```bash
python experiments/enterprise_gpu_confidential_computing_tdx_cc_overhead_sizer/run.py
pytest tests/test_enterprise_gpu_confidential_computing_tdx_cc_overhead_sizer.py -q
```
