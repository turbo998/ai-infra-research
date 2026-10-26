# FlashAttention-4 SM90 块稀疏 Tile 适配与 SM100 核内 TMA Padding、Megatron-LM 设备发起 GroupedLinear Paged Stash 与 DSA Top-K/Indexer FLOPs、Mooncake 按目标聚合传输/HA 校验和保留/通配符 Rail 映射/无锁 DFS 拓扑、NIXL Context Span 及 Dynamo Mocker 指标对齐解析（Phase 90）

> **English Version**: [fa4-sparse-mla-tma-megatron-stash-dsa-flops-mooncake-nixl-dynamo-phase90.en.md](fa4-sparse-mla-tma-megatron-stash-dsa-flops-mooncake-nixl-dynamo-phase90.en.md)  
> **证据等级**：`implementation`（上游固定 commit 源码审查）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/fa4_sparse_mla_tma_megatron_stash_dsa_flops_mooncake_nixl_dynamo_phase90/run.py`）。未运行真实 GPU/RDMA 硬件 benchmark。

## 1. 一手源码证据与关键实现路径

1. **FlashAttention-4 SM90 前向 Tile 适配显式块稀疏粒度与 SM100 Sparse MLA `1..128` Q Heads 核内 TMA Padding**（`Dao-AILab/flash-attention` PRs `#2903` `d9c4d15f5c`, `#2883` `d15f1531a4`）：
   - 在 `flash_attn/cute/interface.py` 中新增 `_fit_sm90_fwd_tile_to_block_sparsity`，当 `head_dim=192` 默认启发式 `tile_n=80` 无法整除显式块大小 `sparse_block_size_kv=128` 时，自动收缩至可整除的 `tile_n=64`，避免抛出 `ValueError`；
   - 在 SM100 Sparse MLA 前向/反向路径中，通过核内 TMA 视图零填充支持任意 `1..128` 个 Q heads（如 TP 切分后的 64 Q heads），无需在全局显存中分配和拷贝填充至 128 heads 的副本（全局显存拷贝开销为 $0\text{ B}$）。
2. **Megatron-LM 设备发起 GroupedLinear Paged Stash 与 DSA Top-K + Lightning Indexer FLOPs 核算**（`NVIDIA/Megatron-LM` PRs `#6828` `f9d606dc47`, `#7043` `705274093a`）：
   - 支持在 `--moe-flex-dispatcher-backend hybridep` 与设备发起（device-initiated）`GroupedLinear` 下使用 **Paged Stash**（将路由专家前向激活打包存入分页缓冲供反向使用，全程零主机同步）；
   - 新增 `megatron/training/logging/dsa_flops.py`，精确核算 DeepSeek Sparse Attention (DSA) 的因果 Top-K 有效序列长度 $\sum_{i=0}^{S-1}\min(i+1, K_{\text{top}})$ 及 Lightning Indexer 开销。在 $S=32768, K_{\text{top}}=2048$ 下，DSA 注意力有效键值比仅为稠密因果注意力的 $12.11\%$，消除了将稀疏训练误算为稠密 FLOPs 导致的虚高 MFU。
3. **Mooncake 按目标聚合多缓冲传输、HA 晋升保留对象 Checksum、TENT 通配符目标保留直连 Rail 与无全局锁读取 DFS 分片拓扑**（`kvcache-ai/Mooncake` PRs `#4799` `b46254dea9`, `#4889` `83f8f87749`, `#4923` `58e243fb29`, `#4770` `ad546de5e0`）：
   - 多缓冲 `put/upsert/get` 按目标端点聚合成单次 Transfer Engine 批次（模型中 256 个小 key 分布于 4 个目标节点时批次提交次数降低 $64\times$），且 Ascend 本地拷贝使用 6 条异步流并发；
   - `StandbyObjectMetadata` 新增 `struct_pack::compatible<uint64_t, 3> object_checksum`，并在 `MasterService::PublishRestoredObjects` 中写回主元数据，确保主备切换（HA promotion）后端到端数据校验和不丢失；
   - TENT RDMA `RailMonitor::findBestRemoteDevice` 在遇到无远端 NUMA 偏好的通配符目标时保留默认直连映射的 peer rail；
   - `ShardAllocator` 消除热路径 `std::atomic_load_explicit(std::shared_ptr)`（在 `libstdc++` 中所有调用竞争同一把全局哈希桶互斥锁与引用计数缓存行），改为无全局锁读取不可变分片拓扑快照。
4. **NIXL `beginContextSpan` 精确 Span ID 绑定与 Dynamo Native Mocker 指标对齐**（`ai-dynamo/nixl` PR `#2508` `c8b46078e4`, `ai-dynamo/dynamo` PR `#17200` `59d7ef4e68`）：
   - NIXL 新增 `beginContextSpan(name, kind, context, request_id)`，直接使用传入 `context` 标识的 `span_id` 作为当前 span ID，其余子事件均以该 `span_id` 为 `parent_span_id`，打通跨进程 NVTX/Chakra 追踪树；
   - Dynamo Rust `mocker` 引擎将原生 Prometheus 指标名与标签同 vLLM、SGLang、TRT-LLM 全面对齐，支持统一路由与压测大盘。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：当 $S=32768, K_{\text{top}}=2048$ 时，因果 DSA 有效 KV 对总数是多少？占稠密因果注意力 $\frac{S(S+1)}{2}$ 的比例是多少？
  - 答：前 $2048$ 个 token 为 $\frac{2048 \times 2049}{2} = 2,098,176$，后 $30720$ 个 token 为 $30720 \times 2048 = 62,914,560$，合计 $65,012,736$；占稠密因果总数 $536,887,296$ 的 $12.1092\% \approx 12.11\%$。
- **可证伪假设**：在高并发 `GetReplicaList`/`BatchGet` 下，`libstdc++` 的 `std::atomic_load_explicit(shared_ptr)` 会因 16 桶全局互斥锁同址碰撞成为多线程吞吐瓶颈。
