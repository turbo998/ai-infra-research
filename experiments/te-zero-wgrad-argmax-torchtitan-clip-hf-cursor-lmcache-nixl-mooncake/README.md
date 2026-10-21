# TransformerEngine Zero-Storage wgrad & NaN-Safe Argmax, TorchTitan GradientClipper & HF Cursor, LMCache Affinity RPC, NIXL UCX CPU Proxy & Mooncake Store Safety

Deterministic Python 3.12 standard-library reproduction for Phase 52 (`priority: 61`) Track 3:

- `NVIDIA/TransformerEngine@fc48751409dd65da99d92cd83ba736e618ddcf43` (`#3670`), `39a8b879dfbfea0d23bc3862dc83ec1d6f970b49` (`#3671`), & `e6afa9ddfa813e39295572385b2fc9fdc77c067b` (`#3606`): Zero-storage dummy `wgrad` via `torch._efficientzerotensor` + per-device `(*shape, dtype, device)` cache, NaN-safe tie-breaking `return_argmax=True` in `parallel_cross_entropy`, and `GroupedLinear` shared backward state holder.
- `pytorch/torchtitan@9698a733623ca997ef824a03cb1f44706ee631e7` (`#5394`) & `c0eb136092402581bc1b184434bd2e46fea90bbc` (`#5501`): Optimizer-owned `GradientClipper` configurable extension point and `_HuggingFaceCursorIterator._pending_hf_state` preserving restored cursor bytes across pre-read packer checkpoints.
- `LMCache/LMCache@b28250e765c80fe67a0f97f050fb8bb6e562f79a` (`#5803`), `fcee51cc5e017dae453a15fb17038b90435fca00` (`#5903`), `ai-dynamo/dynamo@4f42be7bebd96ea0199088840549397809f80a92` (`#16974`), `ai-dynamo/nixl@3f6e1a2716e728b3a9e16b1556a9b51efbebd19b` (`#2327`), & `kvcache-ai/Mooncake@631e1019cf` (`#4622`), `1a6d00d0f4` (`#4688`), `3cf058db6d` (`#4673`), `d782a5998f` (`#4755`): Client-affinity plugin RPCs (`server_module_affinity_call`), `pickle` ban (`TID251`), Dynamo SGLang 2s idle HTTP timeout, NIXL `nixl::ucx::proxyEngine` (`device_proxy=true`), and Mooncake lease/birth-time/partial-bucket/DFS-shard invariants.

## Run

```bash
python experiments/te-zero-wgrad-argmax-torchtitan-clip-hf-cursor-lmcache-nixl-mooncake/run.py --check
```
