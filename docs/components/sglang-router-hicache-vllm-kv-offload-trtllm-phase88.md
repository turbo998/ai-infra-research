# SGLang Confirmed-First 冷前缀路由与 HiCache Mooncake L3 布局暴露、vLLM RL ParamMeta 与 KV Offload 五重加固、以及 TensorRT-LLM Mamba/KDA 缓存模块化与 Beam 假设重构（Phase 88）

> **Language / 语言**: **中文 (Chinese)** | [English (`sglang-router-hicache-vllm-kv-offload-trtllm-phase88.en.md`)](sglang-router-hicache-vllm-kv-offload-trtllm-phase88.en.md)

- **关联队列题（Queue ID）**：`sglang-router-hicache-vllm-kv-offload-trtllm-phase88`（`priority: 97`）
- **关联证据 ID（Evidence ID）**：`ev-sglang-router-hicache-vllm-kv-offload-trtllm-phase88-2026`
- **验证状态（Verification Status）**：`own-offline`（上游一手源码核验 + Python 3.12 标准库离线复现 [`experiments/sglang_router_hicache_vllm_kv_offload_trtllm_phase88/run.py`](../../experiments/sglang_router_hicache_vllm_kv_offload_trtllm_phase88/run.py)）

---

## 1. 核心架构问题与结论速览

在大规模分布式推理与 RL 采样集群中，KV 缓存路由、多级卸载（GPU $\leftrightarrow$ CPU $\leftrightarrow$ P2P/Object L3）与混合 Mamba/Attention 状态管理在 2026-10-26 的最新演进中解决了六类生产级故障与性能退化：
1. **冷前缀突发（Cold-Prefix Burst）导致的单实例拥塞**：当并行采样或 Agent 扇出同时发送 64 个共享全新前缀的请求时，路由器若无上限地将所有后续请求绑定到第一个未确认（pending）前缀所在的 Worker，会导致单节点严重排队而其余节点空闲。
2. **外部 KV 路由器缺乏 Mooncake L3 对象键视图**：HiCache 虽将 KV 卸载至 Mooncake L3，但外部全局路由器无法感知各 DP Rank 的 L3 对象键布局，导致跨实例 L3 命中无法被路由层利用。
3. **vLLM 多批次 LRU CPU Offload 的前缀驱逐反转（Prefix Eviction Inversion）**：长请求跨多个批次写入 CPU 缓存时，若缓存策略不知晓每个 key 在请求内的绝对 token 位置，可能先驱逐正在运行请求的前缀块而保留尾部块，破坏前缀连续性契约。
4. **vLLM Tiered KV Offload 的控制面/数据面竞态**：P2P ZMQ DEALER 在对端死锁时阻塞 `send()` 并在 `Context.destroy()` 时永久挂起；缓存重置（cache reset）期间后台飞行的二级提升（promotion）任务完成后污染新缓存；NIXL 远端注册失败未回滚残留句柄；OBJ 存储层未按 `blocks_per_chunk` 隔离命名空间。
5. **TensorRT-LLM 投机解码下的 Mamba/KDA 状态重放推进错误与 Blackwell TMA 越界映射不足**：若按全部草稿 token 数而非实际接受数（含 bonus token）推进 GDN/KDA 缓存重放状态，下一步解码将静默读到错误循环状态；同时 Blackwell 原生稀疏 FMHA 的无界行域要求 staging pool 地址后方至少映射 $128\text{ KiB}$。
6. **Beam Search 下 Logits Processor 误读槽位历史**：在束搜索中，每个束槽位保存的并不是回溯重组后的完整假设路径（beam hypothesis），导致重复惩罚或引导解码作用在错误的 token 序列上。

### 一手源码对标矩阵

| 上游仓库与提交 | 关键模块 / 路径 | 核心架构机制 | 定量收益 / 验证结果 |
|---|---|---|---|
| `sgl-project/sglang` `#46505` (`eabb11a4c1`, 2026-10-26) | `sgl-router/src/config/{cli,types}.rs` | 新增 `--cache-pending-prefix-consult-below <ratio>`（Confirmed-First 模式）与 `--cache-pending-prefix-max-followers <cap>` 追随上限 | 64 请求冷前缀突发在 8 Worker（`max_followers=7`）下最大单机负载从 `64` 降至 `8`（过载降低 `8.0x`） |
| `sgl-project/sglang` `#40467` (`64fc5f0bdb`, 2026-10-26) | `python/sglang/srt/entrypoints/http_server.py` | 新增 `/get_hicache_l3_cache_layout` 端点，聚合各 DP Rank 的 Mooncake L3 对象键布局供外部 KV 路由器查询 | 实现外部 KV 路由器对 Mooncake L3 缓存的端到端亲和感知 |
| `sgl-project/sglang` `#33250` (`ef11eb3b76`), `#45713` (`6a28c96a49`), `#46308` (`9841183827`) | `moe_transport.py`, `kda_decode_gfx950.cu`, `ipc_a2a_multi.py` | 修复 DeepSeek-V4 非 EP TBO 在 `attn_tp > 1` 下的子批次填充；MI350X/MI355X（`gfx950`）默认启用 HIP 融合 KDA 解码核；Ulysses 隐藏本卡行拷贝 | `gfx950` KDA 解码将 5 个阶段融合为单次启动（启动开销减少 `5.0x`）；Ulysses 零额外本卡拷贝气泡 |
| `vllm-project/vllm` `#63256` (`50a184f435`) & `#61636` (`14d593115d`) | `docs/training/weight_transfer/`, `vllm/lora/model_manager.py` | 在 `ParamMeta` 上声明 RL 权重所有权与分片布局（避免 `nccl_m2n` 调 `full_tensor()`）；修复 3D MoE LoRA 双重 pin_memory 与 EP dummy LoRA 尺寸 | 3D MoE LoRA 锁页主机内存开销降低 `2.0x`，EP 预热按全局专家数对齐 |
| `vllm-project/vllm` `#61050` (`e113018f50`), `#63708` (`51d2a85594`), `#62929` (`14aefcd4f9`), `#63628` (`8a4b018f1c`), `#63655` (`8c6e770a1c`) | `vllm/v1/kv_offload/` | `CachePolicy.insert()` 传入 `req_context` 保护运行中请求前缀；ZMQ DEALER `_SNDHWM=1000` 非阻塞发送；重置期间冻结 promotion；NIXL 失败回滚；OBJ 按 `blocks_per_chunk` 隔离 | 多批次 LRU 前缀保留率达 `100%`；ZMQ 死锁挂起 `0` 次；NIXL 泄漏句柄 `0` 个 |
| `NVIDIA/TensorRT-LLM` `#18843` (`ba6c1fc87d`), `#21027` (`e7d3441e77`), `#20915` (`732fd63463`) | `tensorrt_llm/_torch/attention/backends/sparse/` & `pyexecutor/sampler/beam_search.py` | 模块化 Mamba/GDN/KDA 缓存重放管理器并强制 `_TMA_MIN_BACKING_BYTES = 128 KiB`；`PendingBeamHistories` 在前向传播前异步重组并拷贝每束真实假设 | 100% 消除投机重放步进错位与 Blackwell 稀疏 TMA 越界故障；束搜索 logits 处理器读取精确假设 |

---

## 2. 关键实现路径与协议机制深挖

### 2.1 SGLang Router Confirmed-First 冷前缀流控与 Mooncake L3 布局暴露（PR `#46505` & `#40467`）

在 `sgl-router` 的 `--policy cache_aware` 下，`--cache-pending-prefix-ttl-ms` 会在请求刚被路由但 Worker 尚未通过 KV 事件确认（confirm）缓存落盘前，在路由器本地 Radix Tree 上记下待确认前缀（pending stake）。
- **问题**：默认策略在每次查询时取 `max(pending_depth, confirmed_depth)` 且无追随者上限。当 64 个并发请求携带相同冷前缀到达时，第 1 个请求在 Worker 0 立下 pending stake，随后 63 个请求全部因深度匹配追随至 Worker 0，造成 8 卡集群出现 **8.0x 负载倾斜**；或者当已确认前缀已覆盖 $80\%$ 提示词时，仅因某个繁忙 Worker 上有 $85\%$ 的未确认 stake 就把请求引向繁忙节点。
- **修复（PR `#46505`）**：
  1. `--cache-pending-prefix-max-followers <cap>`：限制单个 pending stake 在 TTL 内最多吸引 `cap` 个追随请求；超过后该 stake 对后续请求不可见，回退至已确认树与负载均衡策略。
  2. `--cache-pending-prefix-consult-below <ratio>`（Confirmed-First 模式）：仅当已确认前缀覆盖率低于 `ratio` 且 pending stake 严格更深时才参考 pending stake。
- **Mooncake L3 布局暴露（PR `#40467`）**：在 `http_server.py` 新增 `GET /get_hicache_l3_cache_layout`，调用 `tokenizer_manager.get_hicache_l3_cache_layout()` 并通过 `merge_l3_layouts(layouts)` 归并所有 DP Rank 的 Mooncake L3 对象键布局，使外部全局路由器能直接把请求发往拥有可复用 L3 块的实例。

### 2.2 vLLM KV Offload 五重鲁棒性加固（PRs `#61050`, `#63708`, `#62929`, `#63628`, `#63655`）

```
[Request Prefix Keys: k0, k1, ..., k7] ---> CPUOffloadingManager._record_request_cache_access()
  |-- passes req_context (get_offload_key_position(key)) to CachePolicy.insert()
  |-- calls policy.on_request_keys(req_context, new_keys)
  +-- Evicts higher-position tail keys first -> 100% running prefix preservation (#61050)

[Tiering & P2P Control/Data Planes]
  |-- ZmqConnection: _SNDHWM = 1000 + non-blocking send() + bounded linger (#63708)
  |-- Cache Reset: _promotions_held = True until in-flight TransferJobs report (#62929)
  |-- NIXL P2P: add_remote_peer_async rolls back partial agent handles on error (#63628)
  +-- OBJ Tier: FileMapper(blocks_per_file=offloading_spec.blocks_per_chunk) (#63655)
```

### 2.3 TensorRT-LLM Mamba/KDA 投机重放缓存与 Beam Search 假设重组（PRs `#18843`, `#20915`, `#21027`）

1. **GDN / Kimi-KDA 投机重放（PR `#18843`）**：在 `FusedRecurrentGatedDeltaRuleCachedReplayUpdate` 中，每步投机解码结束后必须先对本卡所有 GDN 层调用 `commit_gdn_cached_replay_history_layers`，再按**实际接受的草稿 token 数加上 bonus token** 调用 `_advance_replay_state`。同时，在 `csa2/metadata.py` 的 `_allocate_staging_pool` 中强制保证分配字节数不小于 `_TMA_MIN_BACKING_BYTES = 128 * 1024`（$131,072\text{ B}$），满足 Blackwell TMA 硬件对无界行域尾部的映射要求。
2. **Beam Search 异步假设重组（PR `#20915` & `#21027`）**：在 `DecoderRunner._start_beam_histories` 中，仅在最后一个流水线并行级（`self._mapping.is_last_pp_rank()`）于模型前向传播启动前，调用 `_gather_beam_path` 重建每个束的完整 token 路径并发起异步 D2H 拷贝（记录 `copied: torch.cuda.Event`），使前向传播与 D2H 传输完全重叠，供主机端 logits post-processors 精确消费。

---

## 3. AI Infra 创新推理平台对比（Fireworks / Together / Cerebras / Groq）

| 维度 | SGLang + vLLM + TensorRT-LLM（开源全栈引擎） | Fireworks AI | Together AI | Groq LPU | Cerebras WSE-3 |
|---|---|---|---|---|---|
| **冷前缀突发与全局 KV 路由** | `sgl-router` Confirmed-First + `max_followers` 限流 + Mooncake L3 布局暴露 | 会话粘性路由 + 跨副本分布式前缀索引，自动拆散热点前缀突发 | 分层缓存感知路由，结合投机解码动态调整并发批次 | 无共享动态 KV 池，按确定性流水线槽位顺序调度请求 | 单晶圆超大片上内存集中服务，无跨节点 KV 路由倾斜问题 |
| **多级 KV 卸载与故障隔离** | CPU LRU 位置感知前缀保护、ZMQ 非阻塞 HWM、NIXL 注册回滚、OBJ 分块隔离 | NVMe/DRAM/HBM 多级热温冷缓存自动降级，单节点故障秒级切换 | 高速 RDMA 远程 KV 池与本地页表校验，防止中断污染 | 全片上 SRAM，不依赖主机内存或外部 RDMA 存储卸载 | 全片上 SRAM，权重与 KV 零片外卸载 |
| **混合 SSM/Mamba 与投机解码** | TRT-LLM 模块化 GDN/KDA 重放缓存 + SGLang `gfx950` HIP 融合 KDA 解码核 | 原生融合投机草稿验证与状态回滚，支持多分支树状验证 | 针对混合线性注意力定制块级状态快照与重放 | 超高单流 token 速率（数百 tok/s）配合确定性多 token 投机 | 晶圆级宽数据流原生加速多 token 并行草稿验证 |

---

## 4. 架构师练习与可证伪问题

1. **计算练习（冷前缀突发限流）**：当 $B = 64$ 个相同冷前缀请求同时到达 $W = 8$ 台 Worker、单请求预填充耗时 $T_{\text{prefill}} = 120\text{ ms}$ 时：
   - 若不设 `max_followers`，64 个请求全部排队在 Worker 0，最坏排队等待时间为 $64 \times 120\text{ ms} = 7,680\text{ ms}$；
   - 设置 `--cache-pending-prefix-max-followers 7` 后，每个 stake 最多汇聚 $1 + 7 = 8$ 个请求，8 台 Worker 并行分摊，单机最大排队时间降至 $8 \times 120\text{ ms} = 960\text{ ms}$（降低 **8.0x**）。
2. **后续可证伪问题**：
   - 在 vLLM `ZmqConnection` 中将 `_SNDHWM` 设为 `1000` 后，在大规模 P2P 集群瞬时重平衡（例如 128 节点同时广播上千条块失效控制消息）时，是否会误触发健康节点的非阻塞发送异常？
