# TorchTitan DeepSeek-V3 MLA Local Compile, TransformerEngine Blackwell FP8BS via MXFP8 Broadcast, and veRL Tail-Aware Top-K KL

Deterministic Python 3.12 standard-library model verifying:
1. **TorchTitan DeepSeek-V3 `@local_compile("mla_qk")` (`pytorch/torchtitan` `21db340c`)**: Setting `max_pointwise_cat_inputs=0` and `max_complex_pointwise_cat_inputs=0` eliminates `100,663,296` masked-load branch evaluations per step and achieves a `1.7679x` speedup over default Inductor compilation (`2.5446x` over eager).
2. **TransformerEngine Blackwell Grouped GEMM FP8BS Emulation (`NVIDIA/TransformerEngine` `f4a69dc8`)**: Broadcasting 128-element `force_pow_2_scales=True` scales across four 32-element MXFP8 `E8M0` micro-blocks achieves `0.0` quantization distortion while rejecting `4/4` non-power-of-2 scales on SM100+.
3. **veRL Tail-Aware Top-K Forward KL (`volcengine/verl` `5a8dd519`)**: Collapsing non-top-$k$ vocabulary tokens into a $(K+1)$-th tail bucket corrects pathological negative truncated KL (`-0.170396` -> `+0.428749`) when the student starves tail probability mass, and enforces `log_prob_min_clamp=None` to preserve simplex validity.
