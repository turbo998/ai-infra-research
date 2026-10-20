# FlashInfer Big-Channel Causal Conv1D Backward, TensorRT-LLM SM107 DSpark RMSNorm/RoPE & vLLM PaddleOCR-VL Compile

Deterministic Python 3.12 standard-library model (`own-offline`) verifying:
1. `flashinfer-ai/flashinfer@5f8f8aad` (`#6874`): Host-side `channels_at_least` backward kernel selection (`C >= 512` on `sm_100a`/`sm_103a`, `C >= 1024` on `sm_107a`), capacity-owner workspace sizing (`max(ctas_per_sm)`), and author-reported GB300/B200/R200 speedups (up to `1.2581x` on R200 at `C=8192`).
2. `NVIDIA/TensorRT-LLM@083447cd` (`#20366`): Decoupling `_DSPARK_RMSNORM_ROPE_SM_VERSIONS = {100, 103, 107}` from `_DSV4_DSPARK_ARCH_BY_SM = {100, 103}`.
3. `vllm-project/vllm@0ca531ba` (`#61838`): `@support_torch_compile` on `PaddleOCR-VL` `SiglipEncoderLayer` (`2 -> 29` compiled modules).
