# TransformerEngine Deferred Init & THD dSink, TorchTitan Ulysses NCCL & Tokenizer, LMCache FS/IPC/Mamba, Mooncake EFA/etcd & NIXL Redis/GPUNetIO Verification

Deterministic Python 3.12 standard-library verification of:
- `NVIDIA/TransformerEngine@53e241c5a1c978a56c784b37ee794db739c04df8` (`#3327`) & `c3081a5f2db9de31eab51689a25c70ea48068aa6` (`#3470`): Deferred initialization in fusible ops (`reset_parameters()` + meta-device guard) and pre-cuDNN 9.26 THD learnable `dSink` kernel preservation for `d_v in {64, 128, 256}`.
- `pytorch/torchtitan@8899cee5bcf520cbeca62ec7b320af8d6b112428` (`#5393`) & `5e27dc87b8cc8f0f1dd0904bec60a704d9781ad8` (`#5470`): Ulysses `NCCL_GRAPH_REGISTER=0` on NCCL `< 2.32.3` and preserving `AddedToken.special=False` via `add_tokens`.
- `LMCache/LMCache@741b61305902e7d70fd2cf560ad025b251a19f3d` (`#5078`), `786010f4d680653ffe2bf7c88a71684c2fab9495` (`#5693`), & `32f8aa90b7dca6e15a6d7e51c4dff6e2d029b26b` (`#5702`): `fs_native` byte-sized read budget (`32 MiB`), isolated IPC dim-0-padded KV views, and Ascend Mamba KV group edits.
- `kvcache-ai/Mooncake@7ff6bf7f72f15aba7289bc25c87cf491e04efc6f` (`#4704`) & `292e8391b7df0a385092474374660f599958f6c4` (`#4706`): EFA bilateral active handshake lock release + generation counter and configurable etcd leader lease TTL (`2..60s`).
- `ai-dynamo/nixl@d4f72ffe6a34be610bcd6578dc0a4d83c3e176b8` (`#2419`) & `8882e2d3484676aa56502fb3d3535c6477159e14` (`#2422`): Redis `EVLOOP_NO_EXIT_ON_EMPTY` and GPUNetIO bounded metadata/notification parsing.

## Run
```bash
python experiments/te-fusible-deferred-init-dsink-titan-ulysses-lmcache-fs-mooncake-nixl/run.py --check
```
