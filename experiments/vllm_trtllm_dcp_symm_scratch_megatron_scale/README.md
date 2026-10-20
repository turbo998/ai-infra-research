# Experiment: vLLM FlashInfer DCP `trtllm-gen` Prefill, NCCL Symmetric-Memory Scratch Cloning & Megatron-LM `softmax_scale` Precedence (`vllm_trtllm_dcp_symm_scratch_megatron_scale`)

Deterministic Python 3.12 standard-library verification for Phase 34 (`priority: 43`) Topic 1:
- **`vllm-project/vllm@7b569ec895`**: `TrtllmDCPContextWrapper.fits_workspace` (`(num_qo_heads * dcp_world_size) * B * round_up(max_q_len, 256) * 8 + 1 MiB`) and empty local context shard (`seq_len == 0`) zero-init (`torch.zeros_like(q)`) + `mask_dcp_empty_shards_` (`lse = -inf`) to prevent `0 * NaN = NaN` corruption during `cp_lse_ag_out_rs` / `dcp_a2a_lse_reduce`.
- **`vllm-project/vllm@8122776fba`**: `CudaCommunicator._reduce_scatter_symm_mem` (`rs_out`) and `_all_gather_symm_mem` (`ag_out`) `.clone()` on internal symmetric-memory scratch buffers to prevent subsequent same-role collectives from overwriting caller-retained activations (`30.75 -> 0.0` L1 overwrite error).
- **`NVIDIA/Megatron-LM@0156c21373`**: `Attention.get_softmax_scale` 3-tier precedence (`self.softmax_scale` -> `self.config.softmax_scale` -> `d^{-0.5}`) and `MLASelfAttention` YaRN `mscale^2 * config.softmax_scale` composition.

## Run
```bash
python experiments/vllm_trtllm_dcp_symm_scratch_megatron_scale/run.py --check
```
