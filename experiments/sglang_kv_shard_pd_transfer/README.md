# SGLang Logical-Page KV Cache Sharding (`KVShard`) & PD Disaggregated Transfer Experiment

Deterministic Python 3.12 standard-library model verifying `sgl-project/sglang` commits `eb4b167d27` (`#38473`), `de487f8039` (`#40102`), and `e7f7447333` (`#37615`):
- `filter_kv_indices_for_shard_rank` modulo-quotient remapping (`l % N == r -> l // N`, exact global `send_positions`) and non-final empty chunk skipping (`len(kv_indices) == 0 and not is_last_chunk`).
- `max_bootstrap_input_len(capacity, page_size=64, shard_size)` admission boundary (`32,576` at `N=2`, `65,216` at `N=4`, `130,496` at `N=8` for `16,384` physical tokens/rank), preventing `300 s` Decode KV-wait timeouts.
- MLA `KVShard` all-to-all PD pull (`target_tp_ranks = [0..N-1]`, `required_prefill_response_num = N`) and `24,576`-token (`1.50x` single-rank pool) dual-needle recovery (`red` page `20` on rank `0`, `blue` page `341` on rank `1`).

## Run & Verify

```powershell
python experiments\sglang_kv_shard_pd_transfer\run.py --check
```
