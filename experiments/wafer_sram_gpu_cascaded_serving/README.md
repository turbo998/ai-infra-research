# Wafer-Scale / On-Chip SRAM vs GPU HBM Cascaded Speculative Serving (`wafer_sram_gpu_cascaded_serving`)

Deterministic CPU/offline analytical simulator grounded in `vllm-project/vllm` (`commit 973dbf8071f8bdd6dd7eb2d6ee0da7cbe6a93670`, `vllm/v1/spec_decode/suffix_decoding.py` and `vllm/v1/spec_decode/dflash.py`) and published Cerebras WSE-3 / Groq LPU / NVIDIA GB200 NVL72 hardware specifications:
- Quantifies the long-context KV Cache capacity wall (`4K..128K`) on pure on-chip SRAM engines vs HBM3e supernodes.
- Compares GPU HBM3e autoregressive baseline, `SuffixDecodingProposer` (zero-weight CPU suffix-tree speculation), `DFlashProposer` (parallel block cross-attention drafting), pure SRAM with off-wafer KV spill, and heterogeneous `SRAM-Draft (Sliding-8K) + GPU-HBM3e-Target (Full-64K)` cascaded verification.

## Run & Verify

```powershell
python experiments\wafer_sram_gpu_cascaded_serving\run.py
python experiments\wafer_sram_gpu_cascaded_serving\run.py --check
```
