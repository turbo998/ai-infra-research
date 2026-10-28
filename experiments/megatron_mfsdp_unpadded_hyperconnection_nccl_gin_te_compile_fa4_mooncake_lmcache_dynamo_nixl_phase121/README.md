# Phase 121 Track C Experiment: Megatron-LM MFSDP Unpadded Shards & HyperConnection, NCCL GIN/nccl4py, TE Compile Fusion, FA4 Sparse Zero-Fill, Mooncake DMA-BUF, LMCache, Dynamo & NIXL

Reproducible Python standard-library model (`run.py`) and sealed output (`summary.json`) covering 10 upstream commits from `NVIDIA/Megatron-LM`, `NVIDIA/nccl`, `NVIDIA/TransformerEngine`, `Dao-AILab/flash-attention`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, `ai-dynamo/dynamo`, and `ai-dynamo/nixl`:

- `NVIDIA/Megatron-LM@47715bb` & `31c7230`: MFSDP v2 `ShardLayout.UNPADDED` eliminating LCM padding waste across heterogeneous row widths (`18,874,368` padded elements saved on `[(128,6144),(128,4096),(128,14336)]`, reducing total group buffer from `22,020,096` to `3,145,728` elements) and `HyperConnectionTransformerLayer` residual contracts.
- `NVIDIA/nccl@d8347eb` & `373bb82`: GIN Socket backend failing fast (`ncclInternalError`) when `NCCL_GDRCOPY_ENABLE=0` and `nccl4py` `launch_completion_event` (`NcclEventSpec`).
- `NVIDIA/TransformerEngine@45223d7` & `Dao-AILab/flash-attention@86dcb5d`: `torch.compile(fullgraph=True)` forward fusion for `BasicLinear`/`Bias` with frozen-prefix backward support, and FA4 SM100 `zero_fill=True` preventing `0.0 * Inf = NaN` on empty block-sparse Q tiles.
- `kvcache-ai/Mooncake@058c1e8`, `LMCache/LMCache@577cca9`, `ai-dynamo/dynamo@532616d`, & `ai-dynamo/nixl@7bd1d13`: DMA-BUF whole-allocation page-aligned export (`4,206,649 -> 4,210,688` bytes), SGLang MP `cache_salt` isolation, GPT-OSS reasoning channel preservation, and `std::vector<transfer_request_t>` eliminating the 80-byte `postXfer` error-path leak.

## Usage

```bash
python experiments/megatron_mfsdp_unpadded_hyperconnection_nccl_gin_te_compile_fa4_mooncake_lmcache_dynamo_nixl_phase121/run.py --check
```
