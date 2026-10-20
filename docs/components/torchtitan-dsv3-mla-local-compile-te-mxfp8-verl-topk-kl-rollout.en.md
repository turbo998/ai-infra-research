# TorchTitan DeepSeek-V3 MLA Local Region Compile, TransformerEngine Blackwell Grouped GEMM FP8BS via MXFP8 Broadcast, and veRL Tail-Aware Top-K KL Distillation

> **中文版**: [torchtitan-dsv3-mla-local-compile-te-mxfp8-verl-topk-kl-rollout.md](torchtitan-dsv3-mla-local-compile-te-mxfp8-verl-topk-kl-rollout.md)  
> **Evidence Class**: `implementation` (first-hand upstream commits) + `own-offline` (`experiments/torchtitan_mla_compile_te_mxfp8_verl_tail_kl/run.py --check`)

## 1. First-Hand Source Code & Root-Cause Mechanics

1. **TorchTitan DeepSeek-V3 MLA `@local_compile("mla_qk")` Preventing Masked-Load Pointwise-Cat Lowering** (`pytorch/torchtitan` commit [`21db340cba613c00702ea97730cc3cb79b6d6f50`](https://github.com/pytorch/torchtitan/commit/21db340cba613c00702ea97730cc3cb79b6d6f50), `2026-10-20`):
   - **Code Path**: `torchtitan/models/deepseek_v3/attention.py` (`_apply_rope_to_q_and_k_pe`) and `torchtitan/models/deepseek_v3/model.py`.
   - **Mechanism**: Under real-arithmetic RoPE, TorchInductor's default pointwise-cat lowering fuses `torch.cat([q_nope, q_pe], dim=-1)` (`128 + 64 = 192` head dim) into a single kernel using conditional masked loads (`idx < 128`), which is slower than a plain `cat` DMA copy. Commit `21db340c` wraps the split, RoPE, and concat into `_apply_rope_to_q_and_k_pe` decorated with `@local_compile("mla_qk", batch_invariant=True, options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0})`.

2. **TransformerEngine Blackwell Grouped GEMM FP8 Block Scaling (`FP8BS`) via MXFP8 Broadcast** (`NVIDIA/TransformerEngine` commit [`f4a69dc81eabb5002a662e56a36e8454c9840a92`](https://github.com/NVIDIA/TransformerEngine/commit/f4a69dc81eabb5002a662e56a36e8454c9840a92), `2026-10-19`):
   - **Code Path**: `transformer_engine/pytorch/module/grouped_linear.py`, `transformer_engine/pytorch/ops/basic/grouped_linear.py`, and `tests/cpp/operator/test_cast_float8blockwise_grouped.cu`.
   - **Mechanism**: Native FP8 block-scaling (`128`-element blocks) grouped GEMM only exists on Hopper (`SM90`, cuBLASLt 13.6+). On Blackwell (`SM100+`), commit `f4a69dc8` emulates FP8BS using MXFP8 grouped GEMM (`32`-element blocks with `E8M0` power-of-2 scales) by enforcing `force_pow_2_scales=True`, rejecting non-power-of-2 scales in C++ (`GroupedFP8BlockwiseBlackwell.RejectsNonPow2Scales`), broadcasting each 128-element scale across four 32-element MXFP8 blocks with zero error, and requiring `split_alignment % 128 == 0`.

3. **veRL On-Policy Distillation Tail-Aware Top-K Forward KL (`forward_kl_topk_tail`)** (`volcengine/verl` commit [`5a8dd519ab32bfd36d2509a137f830ba79e89c54`](https://github.com/volcengine/verl/commit/5a8dd519ab32bfd36d2509a137f830ba79e89c54), `2026-10-20`):
   - **Code Path**: `verl/trainer/distillation/fsdp/losses.py` (`tail_aware_kl_divergence`), `verl/trainer/distillation/megatron/losses.py`, and `docs/algo/opd.md`.
   - **Mechanism**: Truncated `forward_kl_topk` only sums $\sum_{i \in \text{Top}K} p_i (\log p_i - \log q_i)$ over the teacher's top-$K$ support, turning negative when the student over-concentrates mass on top-$K$ tokens ($q_i > p_i$) and starves the vocabulary tail. Commit `5a8dd519` collapses all non-top-$K$ tokens into a $(K+1)$-th aggregate tail bucket with `tail_mass_eps=1e-6` and forbids per-entry `log_prob_min_clamp` so the $(K+1)$-class simplex stays valid across TP vocabulary shards.

## 2. Deterministic Verification (`experiments/torchtitan_mla_compile_te_mxfp8_verl_tail_kl`)

| Sub-Track | Metric | Baseline | Patched | Architectural Impact |
|---|---|---:|---:|---|
| TorchTitan `@local_compile("mla_qk")` (`21db340c`) | Masked-load branches / Region latency | `100,663,296` / `198.0 us` | `0` / `112.0 us` | **`1.7679x` speedup** vs default compile (**`2.5446x` vs eager**) |
| TE Blackwell FP8BS via MXFP8 (`f4a69dc8`) | $128 \to 4 \times 32$ `E8M0` broadcast error / Non-pow2 rejected | Unsupported on SM100 | `0.0` / `4/4` rejected | Exact zero-distortion emulation of FP8BS on Blackwell MXFP8 Tensor Cores |
| veRL `forward_kl_topk_tail` (`5a8dd519`) | KL when student over-concentrates top-$K$ ($0.99$ vs $0.80$) | `-0.170396` (negative reward) | `+0.428726` (exact penalty) | Restores proper $(K+1)$-class KL divergence and blocks invalid per-entry clamping |
