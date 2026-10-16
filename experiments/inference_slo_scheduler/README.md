# vLLM V1 统一 Token-Budget 调度、KV 块预分配与 P/D 分离 SLO 算例

本目录提供与 [Prefill/Decode 分离、Continuous Batching 与 TTFT/TPOT/吞吐权衡](../../docs/components/inference-slo.md) 配套的**原创标准库 CPU/offline 合成算例**（证据状态：`own-offline`）。

## 1. 验证的源码不变量

基于固定版本 `vllm-project/vllm@25f560a62c4f955672e2c6080b17ab3a48f96201`（`v0.8.2`）`vllm/v1/core/`：

1. **无阶段区分的统一 `token_budget` 调度（`scheduler.py:106-318`）**：
   - V1 `Scheduler.schedule()` 不再把 step 切分为互斥的 Prefill 阶段或 Decode 阶段，而是维护全局 `token_budget = max_num_batched_tokens`；
   - 先遍历 `self.running`（每个在跑 Decode 请求消耗 `1` 个或 `1 + spec` 个 token 预算），仅当本步**未发生任何抢占（`not preempted_reqs`）**且 `token_budget > 0` 时，才从 `self.waiting` 头部切入新请求或 Chunked Prefill。
2. **Prefix Cache 100% 命中时的整块强制重算不变量（`scheduler.py:265-275`）**：
   - 当等待请求的 Prompt 长度恰好被 `block_size` 整除且全部块命中 Prefix Cache（`num_new_tokens == 0`）时，由于 `allocate_slots()` 要求 `num_tokens > 0` 且假定 `num_computed_tokens` 为 `block_size` 的整数倍，调度器执行 `num_computed_tokens -= block_size; num_new_tokens = block_size; computed_blocks.pop()`，强制弹出并重算最后一个完整块（`16` tokens）。
3. **待驱逐命中块（`ref_cnt == 0`）扣减与预分配（`kv_cache_manager.py:168-221`、`block_pool.py:211-244`）**：
   - `KVCacheManager` 默认设置 `num_preallocate_tokens = 64`（`block_size = 16` 时每次多预分配 `4` 个空块以降低每步修改 `free_block_queue` 的开销）；
   - 在检查 `num_new_blocks` 是否可分配时，必须从 `block_pool.get_num_free_blocks()` 中扣减 `new_computed_blocks` 里处于 `ref_cnt == 0`（仍在 `free_block_queue` 待驱逐队列中）的块数 `num_evictable_computed_blocks`，防止将即将被 `touch()` 救活的缓存块重复计作可用新块；
   - 释放请求块时（`kv_cache_manager.py:239-256`），若启用 Prefix Caching，按**逆序（`reversed(blocks)`）**归还至 `free_block_queue` 尾部，确保序列尾部块优先被驱逐而公共前缀头部块保留最久。
4. **同机混部 Chunked Prefill vs Prefill/Decode（P/D）分离的 TTFT/TPOT 边界**：
   - 大 `max_num_batched_tokens = 2048` 虽将 8K Prompt 压缩为 5 个 chunk（`TTFT = 294.8 ms`），但混部步长把在跑 Decode 请求的 `TPOT` 拉高至 `67.6 ms`（违背 `35 ms` SLO）；
   - 缩小 `max_num_batched_tokens = 512` 虽保住 `TPOT = 29.2 ms`，却将 8K Prompt 切碎为 19 个 chunk（`TTFT = 546.8 ms`，违背 `350 ms` SLO）；
   - P/D 分离在 Prefill 节点单步完成算力密集型 Prefill 并经 `50 GB/s` RDMA 传输 `1 GiB` KV Cache（`21.47 ms`），同时满足 `TTFT = 244.27 ms` 与 Decode 节点 `TPOT = 18.0 ms`。

## 2. 复现与校验命令

```bash
python experiments/inference_slo_scheduler/run.py --check
```

## 3. 限制声明

- 本算例仅验证 vLLM V1 核心调度状态机不变量与解析延迟边界，不包含真实 GPU FlashAttention/GEMM Kernel 执行或 RDMA 网卡实测耗时。
