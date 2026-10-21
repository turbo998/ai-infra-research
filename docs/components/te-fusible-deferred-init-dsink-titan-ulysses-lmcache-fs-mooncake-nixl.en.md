# TransformerEngine Fusible-Op Deferred Init & THD Learnable dSink Guard, TorchTitan Ulysses NCCL Graph-Register Workaround & Non-Special Added Tokens, LMCache `fs_native` Byte Read Budget & Isolated IPC / Mamba Edits, Mooncake EFA Lock-Free Handshake RPC & etcd Lease TTL, and NIXL Redis / GPUNetIO Hardening

> **Phase 55 (`priority: 64`) · First-Hand Commit-Level Deep Dive**
> - `NVIDIA/TransformerEngine@53e241c5a1c978a56c784b37ee794db739c04df8` (`#3327`) & `c3081a5f2db9de31eab51689a25c70ea48068aa6` (`#3470`)
> - `pytorch/torchtitan@8899cee5bcf520cbeca62ec7b320af8d6b112428` (`#5393`) & `5e27dc87b8cc8f0f1dd0904bec60a704d9781ad8` (`#5470`)
> - `LMCache/LMCache@741b61305902e7d70fd2cf560ad025b251a19f3d` (`#5078`), `786010f4d680653ffe2bf7c88a71684c2fab9495` (`#5693`), & `32f8aa90b7dca6e15a6d7e51c4dff6e2d029b26b` (`#5702`)
> - `kvcache-ai/Mooncake@7ff6bf7f72f15aba7289bc25c87cf491e04efc6f` (`#4704`) & `292e8391b7df0a385092474374660f599958f6c4` (`#4706`)
> - `ai-dynamo/nixl@d4f72ffe6a34be610bcd6578dc0a4d83c3e176b8` (`#2419`) & `8882e2d3484676aa56502fb3d3535c6477159e14` (`#2422`)

---

## 1. TransformerEngine Fusible-Op Deferred Initialization & THD Learnable `dSink` Guard (`#3327`, `#3470`)

- **Explicit Deferred Init in `te.ops` (`#3327`)**: Adds `reset_parameters()` across all fusible ops (`BasicLinear`, `Bias`, `LayerNorm`, `GroupedLinear`, `Sequential`), updates quantized `BasicLinear` weights in-place via `self.weight.quantize_(weight)`, and raises an error on first forward if parameters remain on `meta` instead of silently materializing them.
- **Narrowed Pre-cuDNN 9.26 THD Learnable `dSink` Guard (`#3470`)**: Preserves specialized kernels for `head_dim_v ∈ {64, 128, 256}` on `cudnn_runtime_version < 92600`, rejecting only the generic value-head-dimension kernel.

---

## 2. TorchTitan Ulysses `NCCL_GRAPH_REGISTER=0` & Non-Special Added Tokens (`#5393`, `#5470`)

- **Ulysses CP Graph-Registration Workaround (`#5393`)**: Sets `NCCL_GRAPH_REGISTER=0` prior to process-group initialization for normal Ulysses on `NCCL < 2.32.3` (`NVIDIA/nccl#2362`), preventing incomplete CUDA-graph IPC mapping reuse across buffer views.
- **Preserving `AddedToken.special=False` (`#5470`)**: Registers added tokens via `add_tokens` rather than `add_special_tokens`, preserving Qwen3 `<think>` / `<tool_call>` delimiters during default decoding while still stripping true special tokens (`<|im_end|>`).

---

## 3. LMCache `fs_native` Byte Budget / Isolated IPC / Mamba Edits & Mooncake / NIXL Hardening (`#5078`, `#5693`, `#5702`, `#4704`, `#4706`, `#2419`, `#2422`)

- **LMCache Storage & IPC (`#5078`, `#5693`, `#5702`)**: Decouples `fs_native` read depth from `num_workers` with a byte-sized in-flight budget (`kDefaultReadMaxBytesInFlight = 32 MiB`), supports dim-0-padded KV pool views under isolated IPC (`#5693`), and adds Ascend split `conv_state` / `ssm_state` Mamba KV group edits (`#5702`).
- **Mooncake EFA Handshake Lock Release & etcd Lease TTL (`#4704`, `#4706`)**: Releases `RWSpinlock` across EFA handshake RPCs (`active_setup_mutex_` + `generation_` counter) to eliminate bilateral active setup deadlocks (`#4704`), and makes `--ha_leader_lease_ttl_sec` configurable in `2..60s` (`#4706`).
- **NIXL Redis `EVLOOP_NO_EXIT_ON_EMPTY` & GPUNetIO Bounded Parsing (`#2419`, `#2422`)**: Keeps the Redis libevent loop alive until pool shutdown so disconnected commands complete with `NIXL_ERR_BACKEND` (`0` hung futures), and validates GPUNetIO metadata/notifications via `std::from_chars` while preserving embedded `NUL` bytes.

---

## 4. Deterministic Verification

Run `python experiments/te-fusible-deferred-init-dsink-titan-ulysses-lmcache-fs-mooncake-nixl/run.py --check` to verify all invariants.
