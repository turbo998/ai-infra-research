# Long-Context Hybrid Attention, FlashMLA KV Formats & Mamba Checkpoint Pool

- **Measurement type**: `own-offline` (deterministic CPU analytical model derived from `deepseek-ai/FlashMLA` and `sgl-project/sglang` source definitions).
- **Clear Boundary**: This experiment does not run GPU or Ascend kernels. Hardware throughput numbers from upstream documentation (`250 -> 410 TFLOPS` on H800 SXM5, `1460 TFLOPS` on B200) are labeled as `author-benchmark` reference context.

## What It Models

1. **Exact Per-Token KV Cache Byte Layouts (`csrc/cuda_kernels/kv_cache_format.h`)**:
   - Standard GQA-8 BF16 (`4096 B/token/layer`, `244.0 GiB` at `1M` tokens across `61` layers).
   - DeepSeek-V3 MLA BF16 (`1152 B/token/layer`, `68.625 GiB` at `1M` tokens).
   - DeepSeek-V3.2 Hopper FP8 MLA (`656 B/token/layer` = `512 B` FP8 + `16 B` FP32 scales + `128 B` BF16 RoPE, `39.0781 GiB` at `1M` tokens).
   - DeepSeek-V4.1 SM100 FP8 MLA (`528 B/token/layer` = `512 B e4m3` + `16 B ue8m0` scales, `31.4531 GiB` at `1M` tokens).
   - DeepSeek-V4.1 SM100 FP4 MLA (`288 B/token/layer` = `256 B e2m1` + `32 B e4m3` scales, `17.1562 GiB` at `1M` tokens).
2. **Hopper SM90 DSM Crossover Dequantization Cycle Model (`docs/20250929-hopper-fp8-sparse-deep-dive.md`)**:
   - Quantifies single-CTA CUDA Core FP8-to-BF16 dequantization (`50.0` cycles/token) vs Tensor Core MMA (`34.0` cycles/token) and 2-CTA Distributed Shared Memory (DSM) Crossover (`25.0` cycles/token, `1.4706x` cycle speedup).
3. **Context Parallelism (`CP=8`) at `1M` Tokens**:
   - Compares Ring Attention P2P (`3584.0 MiB/layer` for GQA-8 vs `1008.0 MiB/layer` for MLA `MQA=1`) against DeepSpeed-Ulysses All-to-All (`7616.0 MiB/layer` and `h_kv=1` KV replication constraint).
4. **SGLang Hybrid Mamba-2 + Attention State Pool (`mamba_checkpoint_pool.py`)**:
   - Compares active `MambaPool` BF16 slots (`46.582 MiB/slot`) vs `Int8CheckpointStore` Radix prefix checkpoints (`24.7852 MiB/slot`, `1.8832x` slot capacity in a `16 GiB` pool) and models prefix truncation replay latency across checkpoint intervals (`2048` to `65536` tokens).

## Usage

```bash
python experiments/long_context_attention/run.py
python experiments/long_context_attention/run.py --check
```
