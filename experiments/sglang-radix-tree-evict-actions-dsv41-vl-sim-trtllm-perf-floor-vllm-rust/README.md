# SGLang Radix-Tree Eviction Actions & DS-V4.1 Vision Sim, TRT-LLM History Floor, and vLLM Guardrails

Deterministic Python 3.12 standard-library reproduction of:
- `sgl-project/sglang@5e79333c95` (`#45439`): Unified Radix-Tree eviction `cache_actions` propagation and `BaseEvictionResult.__del__` resource-drain tripwire.
- `sgl-project/sglang@ed268dd320` (`#45628`): DeepSeek-V4.1 vision-biased MoE `SGLANG_SIMULATE_*` routing (`num_routed_topk`, TP-sharded `token_shard_rank`, and `padded_rows_masked = False` reset).
- `NVIDIA/TensorRT-LLM@7f0caf443e` (`#20297`): `gen_only` device-step-time baseline history floor (`1790856543000` epoch ms) eliminating `3x` modal vs expected-bucket false regressions.
- `vllm-project/vllm@65a886c69f` (`#62859`), `35d01660ff` (`#62244`), & `18e8791134` (`#62823`): Multimodal chunked-embedding rejection, XPU speculative-decoding `OnlineAcceptanceEstimator`, and Rust `response_format` vs `structured_outputs` resolution.

## Run

```bash
python experiments/sglang-radix-tree-evict-actions-dsv41-vl-sim-trtllm-perf-floor-vllm-rust/run.py --check
```
