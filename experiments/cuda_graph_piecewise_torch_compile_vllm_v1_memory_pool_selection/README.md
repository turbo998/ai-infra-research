# CUDA Graph & Dynamic Compilation Selection Experiment (`vLLM V1 Piecewise CUDA Graphs + torch.compile` vs. `Legacy Decode-Only Graphs & Eager Mode`)

Reproducible analytical model comparing 4 CUDA Graph and compilation tiers on `70B-FP8` (`2x H100`, mixed Chunked-Prefill + Decode workload):
- **Tier 1 (`Eager Mode --enforce-eager`)**: `14.8 ms` small-batch TPOT, `22.5 ms` mixed Prefill+Decode step (`320+` separate kernel launches/step).
- **Tier 2 (`Legacy V0 Full-Model Decode-Only CUDA Graphs`)**: `10.6 ms` pure-decode TPOT, **but falls back to slow Eager Mode (`21.8 ms`) whenever a Chunked Prefill request enters the batch**, while consuming `4.8 GB/GPU` of private graph memory pool (`24.0s` capture time).
- **Tier 3 (`Uncached torch.compile + Full Graphs`)**: `9.4 ms` TPOT, `145.0s` cold startup compilation.
- **Tier 4 (`vLLM V1 Piecewise CUDA Graphs + torch.compile Operator Fusion + Persistent Cache`, Recommended)**: **`8.9 ms` small-batch TPOT**, **`14.2 ms` mixed Chunked-Prefill+Decode step (`1.54x` faster than Tier 2)**, **`1.5 GB/GPU` graph pool (`-68.8%` HBM overhead)**, and **`3.8s` warm cached startup**.

## Run & Verify

```bash
python experiments/cuda_graph_piecewise_torch_compile_vllm_v1_memory_pool_selection/run.py
python experiments/cuda_graph_piecewise_torch_compile_vllm_v1_memory_pool_selection/run.py --check
```
