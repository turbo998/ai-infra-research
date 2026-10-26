# Phase 92 Track 2: SGLang MiniMax/AMD/Diffusion, vLLM MRV2/DSv4.1/KV Offload & TensorRT-LLM Unified KV

Deterministic Python 3.12 standard-library simulator covering 10 upstream commits (`sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`):

1. `sgl-project/sglang@a9bf8c76a0` (`#45925`): Support MiniMax-M3.1-Flash (`MSA` monokernel plugin, DSPark uncommitted slot mask before `req_pool_ptr` dereference, and partial RoPE rotation over `ROT` dims).
2. `sgl-project/sglang@d96c386335` (`#45531`): Pass `softcap=layer.logit_cap` instead of `0` in AMD ROCm `aiter_backend.forward_decode`.
3. `sgl-project/sglang@ce77d5e93d` (`#46057`): Clean up rejected and cancelled `/v1/videos` temporary upload directories via `AsyncExitStack` and enforce cloud upload before reporting completion.
4. `sgl-project/sglang@042dc54cbf` (`#46249`): Route `ReleaseRealtimeSessionReq` to the owning DP replica via `_session_key`.
5. `vllm-project/vllm@4135590ffe` (`#63431`): Skip adaptive verification cost profiling during MRV2 `capture_model(profile_only=True)` so throwaway cached activations do not shrink the KV cache budget.
6. `vllm-project/vllm@9744ada582` (`#63576`): Await `krdy` and load `a["klen"]` with `CM_DEV` in ROCm DeepSeek-V4.1 monokernel `stage_combine`.
7. `vllm-project/vllm@14aefcd4f9` (`#62929`): Hold KV Offload tiering promotions after a cache reset until in-flight pre-reset transfer jobs report.
8. `vllm-project/vllm@8a4b018f1c` (`#63628`): Roll back failed or superseded P2P NIXL peer memory registrations in KV Offload.
9. `NVIDIA/TensorRT-LLM@fce28f0726` (`#20178`): Add `BufferConfig::coalescingGroup` to Unified KV Cache Manager V2 (`CoalescingKey = std::pair<int, size_t>`).
10. `NVIDIA/TensorRT-LLM@732fd63463` (`#20915`): Rebuild each live beam's true hypothesis token history via `start_beam_histories_callable` before running logits processors under beam search.
