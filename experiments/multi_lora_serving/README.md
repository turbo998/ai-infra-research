# Multi-LoRA Serverless Serving Simulator (`S-LoRA` + `vLLM Punica`)

Deterministic offline model verifying `1,000+` tenant Serverless Multi-LoRA serving on `8x H100`:
- Shared base GEMM + Triton `SGMV` Shrink (`SPLIT_K=8`) / Expand vs unfused per-adapter batching.
- `S-LoRA` unified paging pool (KV Cache + variable-rank LoRA weights at hidden-dimension stride $H$) vs static max-rank slabs.
- Zipfian (`s = 1.2`) adapter popularity with asynchronous queue-lookahead PCIe Gen5 prefetching.
- Rank break-even crossover $r^* = 64$.
