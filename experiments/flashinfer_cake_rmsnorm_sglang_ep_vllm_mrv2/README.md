# FlashInfer Cake RMSNorm Grad, SGLang gfx1250 W4A8 EP & vLLM MRV2 Spec Async (`Phase 32`)

Deterministic Python 3.12 standard-library simulator verifying:
1. `flashinfer-ai/flashinfer` (`0d062af33b6b42786bc5442acb802dd265242355`): `cake_rmsnorm_train` cooperative persistent backward ring (`cp.async.bulk` + `mbarrier`) at `H=1536` and `H=4096` across `sm_100a`, `sm_103a`, and `sm_107a` (R200 `1.2161x` speedup) with deterministic fixed-order FP32 `dweight` reduction.
2. `sgl-project/sglang` (`89860fbd2e3cf460c0968435670b7fdef6ab5529`): `build_ep_expert_map` (`num_experts + 1` with `-1` sentinels), `gather_src.clamp_(0, N-1)` OOB prevention (`72 -> 0`), and `gate_valid` masking in `aiter_w4a8_gfx1250_forward`.
3. `vllm-project/vllm` (`f1dbfef79973d0f8d05be56e8614db0409e3d7d3`): Scoping `v1_async_sched_incompat` to `not use_v2_model_runner` so MRV2 preserves async scheduling across speculative methods, unpadded drafter batches, and PP>1 (`3/3` preserved).

## Run

```bash
python experiments/flashinfer_cake_rmsnorm_sglang_ep_vllm_mrv2/run.py --check
```
