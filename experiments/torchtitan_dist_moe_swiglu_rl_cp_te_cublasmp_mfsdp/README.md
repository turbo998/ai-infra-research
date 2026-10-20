# TorchTitan Dist-MoE MXFP8 Mega Fused SwiGLU Inference & RL Ulysses CP Per-Token Sharding, TransformerEngine cuBLASMp WGRAD Workspace Reuse & Fullgraph Linear Attention, and Megatron-FSDP Frozen Hook Deferral

Deterministic Python 3.12 standard-library simulator (`own-offline`) verifying:
1. `pytorch/torchtitan@ed3c713`: Dist-MoE MXFP8 Mega kernel with fused SwiGLU (`interleaved_fc13=True`, `dgrad_scale.numel() == 0`) achieving `1.53x–1.72x` speedup vs BF16 (`1.72x` at `R=512` and `R=2048` on B200) and `1.10x` E2E DAPO-Math RL step speedup.
2. `pytorch/torchtitan@9371a2f`: `shard_per_token_tensors` for RL GRPO/DAPO `loss_kwargs` under Ulysses Context Parallelism (`2.0e-7` relative loss diff on 4x GB300, peak memory `4.47 -> 3.92 GiB`, `-12.3%`).
3. `NVIDIA/TransformerEngine@1aaede2` & `5eeb62a`: cuBLASMp 0.11.0+ (`nvte_cublasmp_version() >= 1100`) workspace view reuse in WGRAD (`Linear` backward AllGathers `1 -> 0`, `LayerNormMLP` `2 -> 1`) and `LinearAttention.__init__` operator caching for `torch.compile(fullgraph=True)`.
4. `NVIDIA/Megatron-LM@d37b3a8`, `LMCache/LMCache@64b05b8`, and `ai-dynamo/dynamo@2ae8a46`: Frozen Megatron-FSDP `_delayed_post_backward_callbacks` deferral when `grad_input` is `None`, LMCache `PDBackendAsync` local-request `transfer_spec is None` guard, and Dynamo TRT-LLM `EngineDeadError` teardown bypass.

## Run

```bash
python experiments/torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp/run.py --check
```
