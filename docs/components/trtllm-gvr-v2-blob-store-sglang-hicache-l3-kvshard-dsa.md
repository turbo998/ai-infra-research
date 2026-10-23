# TensorRT-LLM GVR v2 Top-K 解码、Disaggregation Blob Store 驱动边界、SGLang HiCache PD L3 预取承诺契约、KVShard 页分片 DSA 注意力与 vLLM PP 多模态塔跳过深度解析

- **研究日期**：2026-10-24（Phase 59 · Priority 68 · Track B）
- **上游证据链**：
  - `NVIDIA/TensorRT-LLM@4ccdca1e84` (`#20106`, *perf(dsa): GVR top-k v2 — large-batch / small-seqlen optimization*)
  - `NVIDIA/TensorRT-LLM@9f445c2a86` (`#20741`, *feat(auto_deploy): establish canonical identities for CUTLASS MoE implementations*)
  - `NVIDIA/TensorRT-LLM@032ea6d414` (`#20217`, *feat(disagg): add blob store driver boundary and package layout*)
  - `sgl-project/sglang@77bc8cab18` (`#43548`, *fix(hicache): promise prefill only L3 KV that decode prefetch took*)
  - `sgl-project/sglang@cc811ff0c3` (`#44280`, *feat(kv_shard): support DSA attention with page-sharded KV cache*)
  - `sgl-project/sglang@8cd3c494a3` (`#45538`, *fix(hicache): skip linker lookup on decode in PD disagg mode*)
  - `sgl-project/sglang@178febef5d` (`#45214`, *perf(jit_kernel): pack 2 head_dim=128 rows per warp in fused_qknorm*)
  - `vllm-project/vllm@168a86ab8f` (`#60238`, *[MRV2] Skip initializing encoder towers on later PP stages*)
  - `vllm-project/vllm@ef5bcaa387` (`#41482`, *[Bugfix][LoRA] Prevent redundant LoRA loading when load_inplace is True*)
- **证据等级**：`implementation` + `author-benchmark` + `own-offline`（`experiments/trtllm-gvr-v2-blob-store-sglang-hicache-l3-kvshard-dsa/run.py` 确定性验证）

---

## 1. 架构背景与核心问题

在 Blackwell 大规模分离式推理（Disaggregated Prefill-Decode）、稀疏注意力（DSA）与多模态流水线并行（PP）中，Phase 59 Track B 解决了五个核心系统难题：

1. **Blackwell GVR Top-K 解码在大 Batch / 短序列下的粗分桶过早回退与直方图重扫开销（`#20106`）**：`gvr_topk_decode_self_sampling` 的寄存器驻留路径此前仅设置 `QC = 96` 个 crossing-bin 槽位，在大 batch 下极易溢出并退化为全序列二次扫描；且当 `n < 2*k` 且 `b > num_sms`（如 `n=2048, k=2048, b=256`）时，投机单趟 `CURE` 路径必然溢出，白白浪费一次 1024-bin 直方图归约。
2. **分离式 KV 存储缺乏强内存安全契约与版本化键隔离（`#20217`, `#20741`）**：远端 KV 传输驱动若将“逻辑操作完成（`poll()`）”与“RDMA/DMA 物理内存访问结束（`access_ended()`）”混为一谈，会在取消或超时后引发 GPU/Host 缓冲区 Use-After-Free；同时 CUTLASS MoE 缺乏显式 `MoEImplBase` 家族树与 `can_implement` 契约。
3. **SGLang 分离式 PD + HiCache 下 Decode 向 Prefill 虚假承诺未接管的 L3 KV 前缀（`#43548`, `#45538`）**：`DecodePreallocQueue` 此前先调用 `_pre_alloc` 将 `L1+L2+L3` 全长写入 `req.AdjustExtendLen`，再调用 `_start_hicache_prefetch`。一旦 L3 命中段短于 `prefetch_threshold` 或预取抛错被跳过，Prefill 就会跳过该前缀计算，而 Decode 既未预取也未收到传输，导致读出未初始化的垃圾 KV 页！此外，`UnifiedRadixCache` 在 PD Decode 节点上还会误调无回加载路径的 `linker.match`。
4. **SGLang 页分片 `KVShard` 与 DeepSeek 稀疏注意力（DSA）的逻辑/物理页表冲突（`#44280`）及 `head_dim=128` QKNorm 半 Warp 空闲（`#45214`）**：在 `KVShard` 循环页分片下，Attention Top-K 索引需要 0-based 逻辑页表，而 Indexer Gather 需要物理池/暂存区页表；且 `fused_qknorm_warp` 在 `head_dim=128`（256B/行）下用 32 个 lane 处理单行，每 lane 仅加载 8B，损失了 16B 向量化带宽。
5. **vLLM Model Runner V2 非首 PP Stage 冗余分配多模态编码器塔（`#60238`）与 `AsyncLLM` `load_inplace=True` 每步重复读盘（`#41482`）**：多模态编码器仅在 PP rank 0 执行，但非首 stage 仍在加载权重并分配 GPU 显存；而 `AsyncLLM` 缓存 `lora_request` 时未重置 `load_inplace=False`，导致后续每个 decode 步都从磁盘重新加载同一 LoRA 适配器。

---

## 2. 核心实现机制与源码走读

### 2.1 TensorRT-LLM GVR v2 Top-K 解码与 Disagg Blob Store 驱动边界（`#20106`, `#20217`, `#20741`）

1. **GVR v2 寄存器驻留路由与 `CURE` 自适应门控（`#20106`）**：
   - 将寄存器路径 crossing-bin 容量从 `QUADC = 96` 提升至 `QC = 160`（`+67%`），消除短序列 `n <= 8192` 的回退重扫；
   - 新增编译期 `CURE` 开关：当 `n < 2*k and b > num_sms`（即每个元素平均都进 top-k 且 CTA 跨多波次调度）时设 `CURE = false`，跳过必败的单趟投机直方图；
   - 将 `scan_cross_w` 的 32-bin 阈值扫描改为 `uint4`（每次 4 个 `int`）向量化加载，并在 `4K < n <= 8K` 下细分寄存器阶梯：`b <= 148` 用 `(512, 3, 1)` (`k=2048, n<=6144`) 或 `(1024, 2, 1)` (`2*NB` bins)；`b > 148` 用 `(512, 3, 2)` (`n<=6144`) 或 `(512, 4, 2)`（`b <= 296` 用 `2*NB` bins，`b > 296` 用 `NB=1024` bins）。
2. **Disaggregation Blob Store 驱动契约（`#20217`）与 CUTLASS MoE 家族树（`#20741`）**：
   - `encode_key(prefix, name)` 强制校验 `[A-Za-z0-9_-]{1,64}` 与非空 `bytes`，编码为 `"<prefix>/v1/<hex>"`。
   - 严格正交化 `StoreDriver.poll(op)`（逻辑结果就绪）与 `StoreDriver.access_ended(op)`（认证驱动/NIC 永不再触碰调用方 `BufferView` 内存），并规定抛出 `DriverRejected` 时证明 `BufferView` 从未逃逸调用栈。

### 2.2 SGLang HiCache PD L3 预取承诺契约与 `KVShard` DSA 页分片（`#43548`, `#45538`, `#44280`, `#45214`）

1. **Decode 仅向 Prefill 承诺实际立项的 L3 KV 前缀（`#43548`）与跳过 Decode Linker 查询（`#45538`）**：
   - `DecodeHiCachePreallocMixin.pop_preallocated` 改为**先执行 `_start_hicache_prefetch`，后执行 `_pre_alloc`**：若 `_start_hicache_prefetch` 返回空字符串 `""`（因 L3 增量 `< prefetch_threshold` 或异常被捕获），立即调用 `req.hicache_match_result.clear_l3_hit()` 将 `l3_prefix_len` 归零，并通过 `total_prefix_len` 屬性返回 `prefix_match.decode_prefix_len`（而非包含未接管 L3 的 `total_l3_prefix_len`）。
   - `UnifiedRadixCache._match_tree` 在 `get_disagg().disaggregation_mode == "decode"` 时跳过 `self.linker.match(...)`，避免无加载通道的虚假命中。
2. **`KVShard` 页分片 DSA 注意力与双目标写入（`#44280`）**：
   - 区分 `req_to_token_pool.indexer_logical_page_table`（提供给稀疏 Top-K 索引选择器的 0-based 逻辑页号）与物理池/暂存区页表；
   - 在 `extend` 阶段构造 `write_targets = [(self.k_buffer, page_indices), (local_k_buffer, local_slots)]` 双目标写入（第一趟向 scratch 写入全量激活供当前层 gather，第二趟将本 rank 拥有的分片页持久化至本地池并仅在最后一趟执行量化），同时在 `_kv_shard_pool is not None` 时关闭 `supports_mha_one_shot`。
3. **Half-Warp 双行打包 `head_dim=128` QKNorm（`#45214`）**：
   - `fused_qknorm_half_warp` 将每个 32-lane warp 拆为两个 16-lane 半 warp（`row_lane = lane_id & 15, row_slot = lane_id >> 4`），每 lane 单次加载 16B（8 个 BF16/FP16），通过 `width=16` 的 `__shfl_xor_sync` 蝶形归约实现与 32-lane 单行内核**逐比特一致（bit-identical）**的输出；遇到奇数行（如 `T=4099, H_q+H_k=7 -> 28,693` 行）时高半 warp 重算末行但不写回，遇到仅 8B 对齐的 `head_padding` 奇数行时自动回退至单行内核。

### 2.3 vLLM V2 PP 多模态编码器塔跳过与 AsyncLLM LoRA 原位加载修复（`#60238`, `#41482`）

- **`_mark_tower_model` 非首 PP Stage 跳过（`#60238`）**：在 `vllm/model_executor/models/Initialization_utils.py` 中拦截 `SupportsPP` 多模态模型的视觉/音频编码器塔，在 `not get_pp_group().is_first_rank` 时将塔模块权重替换为空占位并跳过 GPU 显存分配（`PP=4` 下单模型可节省 `3 x 1.85 GiB = 5.55 GiB` 显存）。
- **`AsyncLLM` 重置 `lora_request.load_inplace = False`（`#41482`）**：在首次加载或更新缓存前将 `lora_request.load_inplace` 复位为 `False`，使 500 步生成中的磁盘重载次数从 `500` 次降至 `1` 次。

---

## 3. 架构师练习与可证伪问题

1. **定量计算练习**：当 `tokens = 4099, H_q = 5, H_k = 2, head_dim = 128` 时，计算总行数 `total_rows`、传统单行/warp 步数与 `fused_qknorm_half_warp` 双行/warp 步数（答案：`total_rows = 28,693`，单行需 `28,693` 步，双行仅需 `14,347` 步，步数缩减 `1.9999x`）。
2. **可证伪假设**：在分离式 PD + HiCache 架构中，若 Decode 在 `_start_hicache_prefetch` 之前就按 `total_l3_prefix_len` 向 Prefill 发送 `AdjustExtendLen`，任何低于 `prefetch_threshold` 的 L3 命中都会导致 Prefill 跳过对应 token 且 Decode 无法从 L3 拉取，最终在注意力的 KV 槽位中读到未初始化显存。
