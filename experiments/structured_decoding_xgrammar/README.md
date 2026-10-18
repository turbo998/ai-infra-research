# Grammar-Constrained Structured Decoding (`XGrammar` & vLLM V1 Bitmask Pipeline)

Deterministic CPU/offline model for CFG / JSON Schema constrained decoding on a `128,256` vocabulary across `BS = 1..128` and speculative draft trees (`N_tree = 1..64`):

1. **`naive_dynamic_fsm_fp32`**: Per-token dynamic vocabulary regex/FSM scan on CPU + `513,024 B/req` FP32 mask transfer (`6.00 ms` CPU mask build at `BS=64`, `+275.32%` TPOT degradation).
2. **`uncached_pda_fp32_sync`**: Pushdown automaton without `AdaptiveTokenMask` pre-classification (`2.02 ms` exposed overhead at `BS=64`).
3. **`xgrammar_bitmask_sync`**: `AdaptiveTokenMask` precomputes `97.4%` (`124,921`) context-independent tokens into `kAccepted`/`kRejected`/`kAcceptedBitset` and packs masks into `4,008` `int32` words (`16,032 B/req`, **`32.0x`** smaller than FP32).
4. **`xgrammar_async_overlap`**: Overlaps `ThreadPool` CPU bitmask generation for step $t+1$ with GPU decode forward of step $t$, achieving **`37.7 us` (`BS=32`, `+1.76%` TPOT)** and **`56.4 us` (`BS=64`, `+2.26%` TPOT, `3.67x` throughput vs naive FSM)**, plus `TraverseDraftTreeRecursive` DFS `Rollback(1)` verification for EAGLE-3 draft trees.

## Run & Verify

```bash
python experiments/structured_decoding_xgrammar/run.py
python experiments/structured_decoding_xgrammar/run.py --check
```
