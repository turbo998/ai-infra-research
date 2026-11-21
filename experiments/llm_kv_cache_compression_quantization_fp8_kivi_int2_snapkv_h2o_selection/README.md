# LLM KV Cache Compression, Asymmetric Quantization & Sparse Eviction Selection Experiment (`FP8 E4M3` vs. `Asymmetric KIVI INT4` + `DuoAttention` vs. `H2O / SnapKV Eviction`)

Reproducible HBM footprint, batch concurrency, Decode speedup, and multi-turn recall model comparing 4 KV Cache compression architectures on a `128K`-token `Llama-3.3-70B` deployment.

## Run
```bash
python experiments/llm_kv_cache_compression_quantization_fp8_kivi_int2_snapkv_h2o_selection/run.py
python experiments/llm_kv_cache_compression_quantization_fp8_kivi_int2_snapkv_h2o_selection/run.py --check
```
