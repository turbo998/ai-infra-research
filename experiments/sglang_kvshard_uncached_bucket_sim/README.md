# SGLang KVShard Page-Interleaved PD Transfer, `sgl-router` Uncached-Token Bucket Routing & Qwen3 MoE EPLB Simulation

Deterministic Python 3.12 standard-library model of `sgl-project/sglang` commits `eb4b167d27`, `df13d5dce3`, and `9db58ab547`:

- **`KVShard` PD Disaggregation (`conn.py`, `prefill.py`, `utils.py`)**: Logical page $l$ maps to rank $l \bmod N$ and local page $\lfloor l / N \rfloor$; `max_bootstrap_input_len` reserves `shard_size * page_size + 3` tokens (`65,536 -> 65,216` for `N=4, page_size=64`), and ` _prepare_send_indices` skips empty non-final chunks (`3` skipped RDMA sends).
- **`sgl-router` Uncached-Token Bucket Routing (`buckets.rs`, `selection.rs`)**: Routes by `E = sequence_tokens - reusable_tokens` (`5,000 <= 8,192 -> short`), enforces `max_context_tokens`, and spills (`idle_spill`) only when a destination DP rank has both `waiting == 0` and `uncached == 0`.
- **Qwen3 MoE EPLB (`qwen3_moe.py`)**: Passes `ExpertLocationDispatchInfo` into `self.topk(...)` to avoid post-rebalance output corruption.

## Usage

```bash
python experiments/sglang_kvshard_uncached_bucket_sim/run.py
python experiments/sglang_kvshard_uncached_bucket_sim/run.py --check
```
