# FlashAttention-4 SM100 `hdim=256` 2CTA Backward, veRL VeOmni BF16 ReduceScatter, Mooncake Batch C-API, NIXL Telemetry & TE `GroupedTensor`

Reproducible Python 3.12 standard-library model auditing seven first-hand commits across `Dao-AILab/flash-attention`, `volcengine/verl`, `NVIDIA/TransformerEngine`, `kvcache-ai/Mooncake`, `ai-dynamo/nixl`, and `ai-dynamo/dynamo`:

1. **FlashAttention-4 SM100 Unified `hdim=256` 2CTA Backward** (`Dao-AILab/flash-attention` @ `965c9212e6be33c821ee9f2b461564c5c91a2503`, `#2966`): Unifies `hdim=256` backward onto `FlashAttentionBackwardSm100` (`2CTA, tile_m=128, tile_n=64`, serial 2CTA schedule, `512` TMEM columns = `dV 128 | dK 128 | S/P 64 | dP/dS 64 | dQ 128`, `231,424 / 232,448 B` SMEM), reading the peer CTA's `8 KB` contiguous K-block `dS` slice directly from `sdS_dK` without an exchange staging buffer or `dQaccum_empty` wait.
2. **veRL VeOmni FSDP2 `low_precision_reduce_scatter_comm` & TransformerEngine `GroupedTensor` Autograd View Tracking** (`volcengine/verl` @ `fb55b9288560f0c7bf380c1c1c3b45ea170538b0` `#8330`, `NVIDIA/TransformerEngine` @ `2199b7166417a00c76d092b6d79337053f2184cb` `#3701`): Cuts node-local FSDP2 backward ReduceScatter traffic `2.0x` (`FP32 -> BF16` wire format with FP32 accumulation) and restores post-dispatch PyTorch autograd `ViewFunc` + `record_stream` propagation on `GroupedTensor`.
3. **Mooncake $O(1)$ `getBatchTransferStatus` C-API, NIXL 16-Byte `treq` Tag & Single-Shot Telemetry, and Dynamo KVCR Event Filtering** (`kvcache-ai/Mooncake` @ `44094f2e01a05bde80a5a7440caa6d324aa02c12` `#4910`, `ai-dynamo/nixl` @ `f3673f6be3f4f80f6e1e38ff86cebcf9af5750d4` `#2487` & `70badd6aad360a7bbe1f8f06d992ea008f4a8e62` `#2478`, `ai-dynamo/dynamo` @ `8bf042187a8f3f4224e608b1045db14e5f712f30` `#15535`).

## Usage

```bash
python experiments/fa4_sm100_hd256_2cta_bwd_verl_veomni_bf16_rs_mooncake_batch_capi_nixl_dynamo_te/run.py --check
```
