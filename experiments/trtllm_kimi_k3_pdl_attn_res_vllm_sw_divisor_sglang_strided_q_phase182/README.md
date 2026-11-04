# Phase 182: Inference Serving Deterministic Verification Suite

Validates the quantitative serving engine claims for Phase 182 (`priority: 191`):
- **TensorRT-LLM `#21399` (`c8d72931196a`) & `#20973` (`0160bf7a89f4`)**: Kimi-K3 CuTe DSL PDL decode attention-residual mix (`1.7–2.6 us` vs `3.3–11.0 us` on GB300, up to `4.231x`; `2.124x` on Rubin `M=32, N=9` with folded MoE residual; `-13.28%` step time on Rubin `helix16 b1 mtp7`), plus GPU vision pixel LUT normalization (`-17.14%` steady screenshot TTFT, `-19.27%` HTTP CPU time, `2.0x` H2D pixel bandwidth reduction).
- **vLLM `#60252` (`59519ec0ee0b`), `#64119` (`69d27bf8002a`) & `#59383` (`a92efadb30f5`)**: Sliding-window draft block divisor of primary GDN block (`6592 -> 1648`, `4.0x` LCM reduction, `+27.4` pts prefix-cache hit gain on RTX PRO 5000, `1.305x` speedup on 168-turn DGX Spark replay), ModelOpt CUTLASS block-FP8 IPC weight-cache restarts, and `evict_first` lazy KV offload block return.
- **SGLang `#47822` (`27aedf7f93cc`), `#47553` (`744af7ce724f`), `#47494` (`435418253e5e`) & `#48327` (`fd19cfb660f4`)**: AITER zero-copy strided-Q unified extend (`72.7 -> 44.2 us` = `1.645x` at TP2 window on `gfx1250`), R3 non-output rank expert-capture skip (`8x12 MB -> 1x12 MB` = `8.0x` buffer reduction), single Mamba tracking refresh per DSpark verify step, and `7` in-step finish paths unified in `RequestRetirer`.
- **Dynamo `#17716` (`8b4ad6a1b041`) & `#14775` (`e6b5a99a7a45`)**: `PoolRelay` live pool-state sharing across `2` connection modes and remote prefill cancellation propagation (`0` orphaned remote prefill tasks on compatible SGLang pairs).

## Run
```bash
python experiments/trtllm_kimi_k3_pdl_attn_res_vllm_sw_divisor_sglang_strided_q_phase182/run.py --check
```
