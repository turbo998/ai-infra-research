# FlashMLA Absorbed Latent KV Formats & `get_mla_metadata` Variable-Length Split-KV Decode Scheduling

Deterministic offline simulator backed by upstream source verification of `deepseek-ai/FlashMLA` (`commit 0f11e178da1f67becb087a324b400854931ec3e3`):
- `csrc/cuda_kernels/kv_cache_format.h` (`KVCacheFormat<ModelType::V41>`, `KVCacheFormat<ModelType::V41_FP4>`, `KVCachePart<F, 2, CTA>`, `KVFormatPair`)
- `csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu` (`get_mla_metadata_kernel`, `DecodingSchedMeta`, `KU_STG_256`)

## Quick Run

```bash
python experiments/flashmla_paged_decode_scheduling/run.py
python experiments/flashmla_paged_decode_scheduling/run.py --check
```
