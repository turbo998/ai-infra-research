# Experiment: `flashinfer-qwen35-frost-rubin-moe-ep-backends-sm107-cake-deepgemm-mla-fp8`

Deterministic Python 3.12 standard-library reproduction of **Phase 48 (`priority: 57`) Topic 1**:
1. **FlashInfer Qwen 3.5 Gated Attention Block for Rubin (`sm_107`) via Vendored FROST CuTe-DSL (`flashinfer-ai/flashinfer@449967377a`)**: Models the 6-stage fused block (`W_qkvg` projection, QK-RMSNorm + partial RoPE `(1+w)`, `<128`-row decode paged-KV write-through, `d=256` GQA, `O *= sigmoid(GATE)`, `o_proj`) across 103 vendored cuDNN Frontend 1.32.0 (`d8b0a7ed5`) FROST files with zero `cudnn` runtime dependency (`10.2222x` activation HBM traffic reduction).
2. **MoE-EP Dedicated Backend Split, DeepSeekFp8 Block-Scale Benchmark & SM107 NVCC Fix (`flashinfer-ai/flashinfer@9d3837c44b`)**: Models `flashinfer.moe_ep` (`NVLinkOneSidedAlltoAll`, `NVLinkTwoSidedAlltoAll`, `CakeAlltoAll`), `QuantFormat.DeepSeekFp8` (`1.9394x` payload reduction), and exact SM107 (`GR100`) NVCC flag generation.
3. **Self-Contained Cake DeepGEMM / MLA KV Pack / NVFP4 Packages (`@4d79be4100`, `@595bc5746c`, `@597e3a67ca`, `@c2476a6ea4`, `@edfb3f7f88`)**: Verifies CUPTI cold-L2 measurements on GB300 (`sm_103a`) and GB200 (`sm_100a`).

## Run
```bash
python experiments/flashinfer-qwen35-frost-rubin-moe-ep-backends-sm107-cake-deepgemm-mla-fp8/run.py --check
```
