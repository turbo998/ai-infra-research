# Phase 110 Track B: SGLang SWA Tombstone Clamp & Non-Blocking Disagg KV Restore, vLLM Hopper MiniMax-M3 & CuteDSL BF16x3 Router & DSV4.1 CG, TensorRT-LLM ADP KV & Logprobs Memory

Deterministic stdlib model of Phase 110 (`priority: 119`) Track B inference runtime and KV cache changes:
- `sglang` (`e7c55f6959`, `b28f4fc065`, `b7ac4c0e72`)
- `vllm` (`8e26dec5f5`, `0076619a94`, `5c1eb11cb3`, `8977b13a1c`, `cebcc5d8c7`)
- `TensorRT-LLM` (`0a03c3ed05`, `8354c4c398`)

Run `python experiments/sglang_swa_tombstone_disagg_kv_vllm_minimax_trtllm_phase110/run.py --check` to verify `summary.json`.
