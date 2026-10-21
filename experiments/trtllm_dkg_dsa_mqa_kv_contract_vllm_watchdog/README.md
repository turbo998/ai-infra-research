# TRT-LLM CuTe DSL Uneven `next_n` DSA MQA Logits, KV Transfer Contract & vLLM Watchdog Experiment

Deterministic Python 3.12 standard-library model reproducing:
1. **`NVIDIA/TensorRT-LLM` PR `#19976` (`033f2ea020`)**: `use_cute_dsl_paged_mqa_logits` on `sm_100f` with `_pick_dsl_split` in-kernel uneven atom splitting (`num_atoms = ceil(next_n / 3)`) and causal stagger `_atom_ctx_offsets` (`[2, 0]` for `next_n=5`, `[4, 2, 0]` for `next_n=7`).
2. **`NVIDIA/TensorRT-LLM` PR `#20117` (`a9000fe10e`) & PR `#20608` (`16d26df5a7`)**: Disaggregated KV `Backend` / `SlotRegion` / `View` / `Outcome` / `Attempt.quiesced()` safety contract and `RemoteMpiCommSessionServer` `SHUTDOWN_ACK` handshake (`PENDING_POLL_SECONDS = 0.1`).
3. **`vllm-project/vllm` PR `#55700` (`1c87991978`)**: `ThreadedWatchDog` one-shot stack dump per stall and `with wd.disable():` idle window suspension in `shm_broadcast.py`.

## Run

```powershell
python experiments\trtllm_dkg_dsa_mqa_kv_contract_vllm_watchdog\run.py --check
```
