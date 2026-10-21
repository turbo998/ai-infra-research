# Megatron-LM GDN Single-Update GDP Prefill CUDA Graph Replay, TorchTitan Optimizer-Owned GradientClipper & LoRA SPMD Typing, LMCache Watermark Prefetch Admission & VMM Padded IPC, and Mooncake/NIXL/Dynamo Transport Telemetry

Deterministic Python 3.12 standard-library reproduction of:
- `NVIDIA/Megatron-LM@29e1a7b669` (`#8353`): GDN dynamic-inference prefill modeled as single-update GDP (`num_householder = 1`, `GDP_CHUNK_SIZE = 64`) for bit-exact CUDA graph replay.
- `pytorch/torchtitan@9698a73362` (`#5394`) & `a242475253` (`#5390`): Optimizer-owned `GradientClipper` extension point and `_LinearFunction.spmd_typecheck` `einsum` rule for TP RowParallel LoRA (`Partial` passthrough).
- `LMCache/LMCache@a6accee6d8` (`#5889`), `21aa6ac06c` (`#5855`), & `45486917f1` (`#5929`): Dynamic `--l2-prefetch-inflight-watermark=0.5` admission (`4x` higher small-request concurrency without L1 OOM), VMM dim-0 padded KV pool IPC (`3584 B` span vs `2048 B`), and primary CUDA context binding before `cuStreamWaitValue64` / `cuStreamWriteValue64`.
- `kvcache-ai/Mooncake@7a05855674` (`#4410`), `ai-dynamo/nixl@03d9b74648` (`#2479`), & `ai-dynamo/dynamo@5004f3fa80` (`#16947`): CPU-initiator peer-GPU TCP capability (`dram_to_gpu = true`), NIXL 64-bit unsigned decimal `nixl.xfer_request_id` & `nixl.xfer_execution` correlation, and Dynamo multimodal image token error propagation (`Result<usize>`).

## Run

```bash
python experiments/megatron-gdn-cudagraph-titan-gradclip-lmcache-prefetch-vmm-mooncake-nixl/run.py --check
```
