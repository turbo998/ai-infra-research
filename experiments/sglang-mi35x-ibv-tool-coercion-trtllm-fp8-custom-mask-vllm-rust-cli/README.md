# SGLang MI35x Mooncake ibv_reg_mr Disaggregation & Tool Coercion, TensorRT-LLM FP8 Custom-Mask Attention, and vLLM Rust CLI Parity

Deterministic Python 3.12 standard-library reproduction for Phase 52 (`priority: 61`) Track 2:

- `sgl-project/sglang@7e0e259435a4cc73df71e096607d5fd219458c7b` (`#45519`), `a2e4ffbcea182496700fd9329d0363ddf9c78985` (`#43941`), & `6b6c61314d0c41d34ef4722854be52d635d0dc61` (`#43698`): MI35x (`gfx950`) `MOONCAKE_DISABLE_HIP_DMABUF=1` fallback to `ibv_reg_mr` to eliminate ROCm stream creation OOM/segfaults, whole-number float string coercion (`"3.0" -> 3`) for integer tool parameters, and Flux ControlNet `id(residual)` deduplicated sequence-parallel sharding.
- `NVIDIA/TensorRT-LLM@5f90e8939d6bab66ee90e3e8555bce3f5a981a54` (`#20387`) & `1c7fadcfbe50911d784658e902d1bd003d89fdd7` (`#20253`): FP8 custom-mask context attention in `TritonCustomMaskFmha` with tile-level `kv_scale_quant_orig` dequantization (zero full-pool BF16 copy) and `[-448.0, 448.0]` clamped `float8_e4m3fn` output scaling, plus warmup switch cleanup.
- `vllm-project/vllm@2916b4b4a6b7e9ae552e661b365fec65117134ac` (`#62822`), `bc4794ef9a9fea5f8a003ef4dc5a49bbe8145241` (`#62825`), & `8a21d0a328bb3cbb9c73ca0ae65d83465768cca4` (`#62416`): Rust server `return_token_ids` propagation, `VLLM_LOG_STATS_INTERVAL` `1ns` floor (`Duration::from_nanos(1)`), `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`, and `ModelArgs::file_sources()` (`--tokenizer`, `--tokenizer-revision`, `--hf-config-path`).

## Run

```bash
python experiments/sglang-mi35x-ibv-tool-coercion-trtllm-fp8-custom-mask-vllm-rust-cli/run.py --check
```
