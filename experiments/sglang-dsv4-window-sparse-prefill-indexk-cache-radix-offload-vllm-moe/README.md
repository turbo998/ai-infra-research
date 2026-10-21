# SGLang DeepSeek-V4.1 Request-Window History Deduplication, Cached Index-K Prefill Plan, Restored Sparse Prefill, Ordered Radix-Cache Offloads, and vLLM Monolithic MoE Simulated Routing

Deterministic Python 3.12 standard-library reproduction of:
- `sgl-project/sglang@f7d4f06242` (`#45579`): Precomputing `WindowLayout.history_src` once per batch instead of across 60 layers (`60x` reduction).
- `sgl-project/sglang@bc412213c1` (`#45577`) & `97945831f9` (`#45573`): Gathering DSV4.1 `prefill_k_slots` once per compression ratio, caching `low_ratio_index_plans` (`24x` fewer H2D/plan builds), and restoring sparse prefill with active request windows.
- `sgl-project/sglang@2c7a9e3b50` (`#45559`): Ordering Mooncake/UMBP `UnifiedCacheLinker` offloads behind both the scheduler stream and `forward_stream`.
- `sgl-project/sglang@620718c8ea` (`#45575`) & `vllm-project/vllm@b0e1f8052a` (`#52780`): Keeping DSV4.1 vision router off packed-ids under routing simulation and routing vLLM `RoutingSimulatorRouter` to decomposed MoE kernels (`supports_monolithic_kernels = False`).

## Run

```bash
python experiments/sglang-dsv4-window-sparse-prefill-indexk-cache-radix-offload-vllm-moe/run.py --check
```
