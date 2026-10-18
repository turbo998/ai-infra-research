# 4-Tier KV Cache Hierarchy (`HBM3e -> Host DDR5 -> CXL/Mooncake -> NVMe GDS`)

Deterministic `own-offline` simulator for **Multi-Turn Long-Context Agentic KV Cache Offloading and Recompute vs Swap Breakeven**, grounded in `LMCache/LMCache` (`StorageManager`, `GdsBackend`, `NixlStorageBackend`) and `vllm-project/vllm` (`MooncakeStoreConnector`, `SimpleCPUOffloadConnector`).

## Key Findings
- **Swap-in vs Recompute Breakeven ($L_{\text{breakeven}}$)**: On `8x H100 TP=8` (`Llama-3-70B-FP8`, `160 KiB/tok` node KV), `64K` prefill recompute takes **`3,077.12 ms`** due to quadratic causal attention. Swapping `64K` KV from pinned Host DDR5 (`448 GB/s`) takes **`24.32 ms`** (`126.53x` faster, $L_{\text{breakeven}} = 25\text{ tokens}$), from CXL 3.0 / Mooncake RDMA pool (`360 GB/s`) takes **`30.48 ms`** (`100.96x` faster, $L_{\text{breakeven}} = 46\text{ tokens}$), and from 8x NVMe RAID0 via GPUDirect Storage `cuFile` (`52 GB/s`) takes **`208.29 ms`** (`14.77x` faster, $L_{\text{breakeven}} = 123\text{ tokens}$), whereas legacy POSIX buffered disk (`9.5 GB/s`) requires $L_{\text{breakeven}} = 6,940\text{ tokens}$.
- **1,000-Session Agentic Workload**: Coordinating all 4 tiers eliminates `87.2%` HBM eviction recomputes (`0.0%` recompute rate), slashes P99 TTFT by **`18.91x`** (`4,387.91 ms -> 231.99 ms`, mean TTFT `44.18 ms`), and reduces per-1M-token serving cost by **`66.6%`**.

## Verification
```powershell
python experiments\tiered_kv_cache_hierarchy\run.py --check
```
