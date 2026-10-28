# Phase 119 Track B: SGLang Diffusion SP/MOSS-Image, MegaMoE, vLLM RL NCCL & HiSparse, and TRT-LLM GVR K=2048

Reproducible stdlib model covering 10 primary upstream commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:
- `sglang#46664` (`5bb7ecc2c2`): Preserve full latents for MOSS-Image sequence parallelism prior to full-image RoPE.
- `sglang#46897` (`357c53885f`): Host-free DSpark draft prep (`extend_start_loc = cumsum(extend_seq_lens) - extend_seq_lens`).
- `sglang#46267` (`15d39bd840`): MegaMoE stops double-dividing DP prefill budget (`2.0x` prefill budget restoration at `attn_dp_size=2`).
- `sglang#46641` (`610a74d648`): MOSS-Image fused joint-block QKV projection in `high` quality (`3 -> 1` GEMMs, strided `_norm_rope_kernel`).
- `sglang#46812` (`54be2010d3`): PD prefill scheduler refreshes `bootstrap_queue[:512]` in `_touch_waiting_prefixes`.
- `vllm#62557` (`2b678ac739`): Per-update source ranks (`0 <= src < rank_offset`) for RL NCCL weight transfer.
- `vllm#62797` (`1c787a25cc`): Reject non-linear expert placement with LoRA + EP.
- `vllm#63819` (`458334f00f`): Close DFlash/DSpark and `OnlineAcceptanceEstimator` (`BLOCK_R >= 256`) JIT warmup gaps.
- `vllm#61659` (`ebaa812475`): HiSparse pads staged prefill KV to `>= 256` rows (`656 B/row`) to prevent `trtllm-gen` sparse MLA OOB reads.
- `TensorRT-LLM#20998` (`9cf4530787`): GVR `K=2048` `reg_clus` with `BLKC=1024`, `HINT_SLOTS=2`, block64 sampling, and parallel rescue.
