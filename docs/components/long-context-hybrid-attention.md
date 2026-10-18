# 超长上下文混合注意力：FlashMLA / DSA 稀疏算子、Ring vs Ulysses CP 与 SGLang Mamba-2 状态检查点池

## 1. 核心架构问题与结论（Executive Summary）

当 LLM 上下文窗口扩展至 **1,048,576 (`1M`) Tokens** 时，传统稠密自回归注意力与按 Token 分页的 Paged KV Cache 面临三重物理墙：
1. **HBM 容量与解码带宽墙**：61 层标准 GQA-8 (`8` KV heads, `d_head = 128`, BF16) 在 `1M` 上下文下单请求需 **`244.0 GiB`** 显存，单步 Decode 扫描全部历史 KV 需读取 `249,856.0 MiB`，在 `6.4 TB/s` 有效 HBM 带宽下仅 KV 读取就耗时 **`40.9364 ms`**。
2. **量化反量化指令吞吐倒挂（Dequantization-Bound）**：将 MLA KV Cache 压缩至 FP8 后，NVIDIA Hopper (`H800`) 单个 SM 的 Tensor Core 仅需 **`34.0` cycles/token** 即可完成 64 个 Query Heads 的 MMA 计算，但 CUDA Core 执行 `float8_e4m3 -> half -> float32 -> bfloat16` 与 Scale 乘法却需要 **`50.0` cycles/token**，导致算子被反量化卡死。
3. **混合线性注意力（Hybrid Mamba-2 / GDN）的不可任意截断性**：Mamba-2 / 线性注意力的循环状态（Recurrent State $S_t$）在数学上随 Token 递推更新（复杂度 $O(1)$），不保存逐 Token 历史状态。一旦在 Radix Prefix Cache 中发生任意位置的分支命中（Branch Hit）或投机解码回退（Speculative Retract），若无中间检查点，系统必须从头或从最近的 Checkpoint 重新执行 Prefill（Replay）。

基于 `deepseek-ai/FlashMLA`（commit `0f11e178da1f67becb087a324b400854931ec3e3`）与 `sgl-project/sglang`（commit `4ec19ca27f08f7ce44bbe1da70ce964731940eb2`）一手源码核验与 `experiments/long_context_attention/run.py` 离线建模，核心量化结论如下：
- **FlashMLA 精确到字节的 KV 布局演进**：从 DeepSeek-V3 BF16 MLA（`1152 B/token/layer`，`1M` 占 **`68.625 GiB`**）演进到 DeepSeek-V3.2 Hopper FP8 MLA（`656 B/token/layer` = `512 B` FP8 + `16 B` FP32 scales + `128 B` BF16 RoPE，占 **`39.0781 GiB`**），再到 DeepSeek-V4.1 SM100 全量化 FP8 MLA（`528 B/token/layer` = `512 B e4m3` + `16 B ue8m0` scales，占 **`31.4531 GiB`**）与 FP4 MLA（`288 B/token/layer` = `256 B e2m1` + `32 B e4m3` scales，占 **`17.1562 GiB`**，相对 GQA-8 BF16 压缩 **`14.2222x`**）。结合 DeepSeek Sparse Attention (`DSA`, `topk = 2048`)，单步 Decode 的 KV HBM 读取降至 **`62.9062 MiB`**（V4.1 FP8）与 **`34.3125 MiB`**（V4.1 FP4），HBM 传输耗时分别降至 **`0.0103 ms`** 与 **`0.0056 ms`**。
- **Hopper 2-CTA Distributed Shared Memory (`DSM`) Crossover**：利用 MLA 解码等价于 MQA（`h_kv = 1`，同一 Token 的 `128` 个 Query Heads 共享同一组 KV）的不变量，FlashMLA 将 2 个各负责 64 个 Query Heads 的 CTA 组成 Cluster (`NUM_CTAS = 2`)，每个 CTA 仅加载并反量化 **一半（256 维）** KV，并通过 `st.async` 写入对端 CTA 的 Distributed Shared Memory，将每 CTA 反量化周期从 **`50.0` cycles** 砍半至 **`25.0` cycles**（小于 Tensor Core 的 `34.0` cycles），彻底消除反量化瓶颈，实测 H800 SXM5 稀疏解码吞吐从 **`250 TFLOPS` 跃升至 `410 TFLOPS`**（`author-benchmark`）。
- **Context Parallelism (`CP = 8`) 的 MLA/MQA 拓扑陷阱**：在 `1M` Tokens、`CP = 8` 下，Ring Attention 通过点对点（P2P）环状流水传递 KV，当模型采用 MLA (`h_kv = 1`, `576` dims) 时每层每卡仅需通信 **`1008.0 MiB`**（GQA-8 需 `3584.0 MiB`）且完全被 `7505.9994 ms` 的注意力计算隐藏；而 DeepSpeed-Ulysses 依赖按注意力头切分的 4 次 `All-to-All`，当 `CP = 8 > h_kv = 1` 时必须跨 Rank 复制 KV 头，导致每层每卡通信膨胀至 **`7616.0 MiB`** 且暴露在关键路径上。
- **SGLang `MambaCheckpointPool` 的 `int8` 对称检查点存储**：SGLang 将运行期全精度 `MambaPool`（每槽位 **`46.582 MiB`**）与 Radix 树持有的 `MambaCheckpointPool` 解耦，在 `Int8CheckpointStore` 中沿 `d_v` 轴归约计算每个 `(head, k-channel)` 的对称 `int8` 缩放因子（契合递推衰减矩阵 $\text{diag}(\alpha)$），将每检查点体积压缩至 **`24.7852 MiB`**，在 `16 GiB` 缓存预算下把可缓存前缀节点数从 **`351` 个提升至 `661` 个（`1.8832x`）**。

---

## 2. 一手源码深度剖析（Implementation & Hardware Alignment）

### 2.1 `deepseek-ai/FlashMLA`：从 Hopper 656B 到 SM100 V4.1 528B / 288B KV 格式

在 `csrc/cuda_kernels/kv_cache_format.h`（commit `0f11e178da1f67becb087a324b400854931ec3e3`）与 `docs/20250929-hopper-fp8-sparse-deep-dive.md` 中，FlashMLA 明确定义了两代量化 KV 缓存结构：

1. **DeepSeek-V3.2 Hopper (`SM90`) 656-Byte 格式**（`docs/20250929-hopper-fp8-sparse-deep-dive.md:7-12`）：
   - `head_dim_k = 576`，`head_dim_v = 512`，`h_kv = 1`（MQA 模式，`h_q = 128`）。
   - 前 `512` 维（NoPE 潜变量）采用 $1 \times 128$ 细粒度 Tile 量化：存储 `512` 个 `float8_e4m3`（`512 B`）与 `4` 个 `float32` 缩放因子（`16 B`）。
   - 后 `64` 维（RoPE 旋转位置编码部分）对精度极度敏感，在 V3.2 中保持未量化的 `bfloat16`（`64 × 2 = 128 B`）。
   - 单 Token 单层总字节数：
     $$\text{Bytes}_{\text{V3.2}} = 512 + 16 + 128 = 656\text{ Bytes}$$

2. **DeepSeek-V4.1 Blackwell (`SM100`) 528-Byte (`FP8`) 与 288-Byte (`FP4`) 格式**（`csrc/cuda_kernels/kv_cache_format.h:7-20` & `flash_mla/flash_mla_interface.py:112-118`）：
   ```cpp
   template<ModelType MT>
   struct KVCacheFormat {
       static constexpr ModelType MODEL_TYPE = MT;
       static constexpr bool IS_FP4 = MT == ModelType::V41_FP4;
       static constexpr int D_QK = 512;
       static constexpr int D_ROPE = 64;
       static constexpr int D_NOPE = D_QK - D_ROPE;
       static constexpr int D_FP4 = IS_FP4 ? D_QK : 0;    // Dimensions stored as fp4 e2m1
       static constexpr int D_FP8 = IS_FP4 ? 0 : D_QK;    // Dimensions stored as fp8 e4m3, needing dequant
       static constexpr int QUANT_TILE_SIZE = IS_FP4 ? 16 : 32; // Dimensions sharing one scale
       static constexpr int NUM_SCALES_EACH_TOKEN = D_QK / QUANT_TILE_SIZE; // 16 (V41) / 32 (fp4)
       static constexpr int QUANT_BYTES = D_FP4 / 2 + D_FP8;
       static constexpr int BYTES_PER_TOKEN = QUANT_BYTES + NUM_SCALES_EACH_TOKEN; // 528 / 288
   };
   ```
   - **V4.1 FP8 (`528 Bytes/token`)**：`D_QK = 512`（包含 `64` 维 RoPE 全部量化为 `float8_e4m3`，彻底移除 `bfloat16` 尾部），每 `32` 个连续元素共享 1 个 `float8_e8m0` 指数缩放因子（共 `512 / 32 = 16 B`）。
   - **V4.1 FP4 (`288 Bytes/token`)**：`512` 维全部量化为 `e2m1`（每字节打包 2 个值，占 `256 B`），每 `16` 个连续元素共享 1 个 `float8_e4m3` 缩放因子（共 `512 / 16 = 32 B`）。在混合精度稀疏解码中，主缓存 `k_cache` 采用 `528 B` FP8，而扩展缓存 `extra_k_cache` 可挂载 `288 B` FP4（`is_valid_kv_format_pair(KV, EXTRA_KV)`）。
   - **稀疏扁平寻址契约（`flash_mla_interface.py:84-105`）**：在 `SM100` 稀疏解码中，`indices` 形状为 `(batch_size, seq_len_q, topk)`，直接编码物理偏移 `page_block_index * page_block_size + offset_in_block`（忽略 `block_table` 参数）。同时要求整个 `k_cache` 底层虚拟地址空间必须连续有效（Contiguously Valid），且唯一合法的无效索引标记为 `-1`（内核不做上界检查，任何越界正整数会直接触发 TMA 非法内存访问 IMA）。

### 2.2 Hopper DSM Crossover：破解 CUDA Core 反量化瓶颈

在 `docs/20250929-hopper-fp8-sparse-deep-dive.md:15-46` 中，DeepSeek 团队给出了极具代表性的微架构周期核算：
- **Tensor Core MMA 周期**：H800 单 SM 每周期可执行 `4096` 次 BF16 MMA FLOPs（`989 TFLOPS / 1830 MHz / 132 SMs`）。每个 CTA 处理 `64` 个 Query Heads，每消费 1 个 KV Token 的 QK 与 PV 矩阵乘共需：
  $$\text{Cycles}_{\text{MMA}} = \frac{64 \times (576 + 512) \times 2}{4096} = 34.0\text{ cycles}$$
- **CUDA Core FP8 反量化周期**：由于 Hopper SM90 缺乏原生 `cvt.rn.bf16x2` 直接转换指令（注：`README.md:80` 指出 CUDA 13.2+ 在 SM100 上提供了原生转换），H800 必须依次执行 `e4m3 -> half`（吞吐 `64/cycle`）、`half -> float32`（`64/cycle`）、`float32 -> bfloat16`（`16/cycle`）以及与 `float32` scale 相乘（`256/cycle`）。反量化 `512` 个元素至少需要：
  $$\text{Cycles}_{\text{Dequant, 1-CTA}} = \left(\frac{1}{64} + \frac{1}{64} + \frac{1}{16} + \frac{1}{256}\right) \times 512 = 50.0\text{ cycles}$$
  由于 `50.0 > 34.0`，算子完全受限于 CUDA Core 反量化（Dequantization-Bound）。
- **2-CTA Cluster DSM Crossover**：将处理同一 Query Token 的 `128` 个 Query Heads 拆分为大小等于 2 的 CTA Cluster（`CTA0` 处理前 64 头，`CTA1` 处理后 64 头）。每个 CTA 仅从 HBM 加载并反量化 `256` 个元素（耗时降至 **`25.0` cycles**），随后一边写入本地 Shared Memory，一边利用 Hopper 硬件 `st.async` 指令异步写入同伴 CTA 的 Distributed Shared Memory，并通过 Cluster Transaction Barrier 同步。反量化耗时从 `50.0` cycles 降至 `25.0` cycles，使流水线重新回到 `34.0` cycles 的 Tensor Core MMA 主导态（理论周期压缩比 `50 / 34 = 1.4706x`，实测 H800 稀疏解码算力从 `250 TFLOPS` 提升至 `410 TFLOPS`，增益 `1.64x`）。

### 2.3 SGLang `MambaSlotAllocator` 与 `MambaCheckpointPool` 混合状态管理

在 `sgl-project/sglang`（commit `4ec19ca27f08f7ce44bbe1da70ce964731940eb2`）中，SGLang 针对 Qwen3-Next / DeepSeek-V4 / Granite-Hybrid 等混合线性注意力架构实现了三层解耦：
1. **按请求分配的 `MambaSlotAllocator`（`python/sglang/srt/mem_cache/allocator/mamba.py:30-88`）**：
   - 明确声明 **不继承 `BaseTokenToKVPoolAllocator`**：因为 Mamba 每个活跃请求只需维护 1 个固定大小的全层递推状态槽（Slot），而非按 Token 分配页索引；其中 `Slot 0` 固定保留为 Padding Token 的哑写入目标（Dummy Write Target）。
2. **`MambaCheckpointPool` 与 `Int8CheckpointStore`（`python/sglang/srt/mem_cache/mamba_checkpoint_pool.py:15-155`）**：
   - 活跃请求在 `MambaPool` 中保持 `bfloat16` 全精度递推；而缓存在 Radix Prefix Tree 节点上的历史状态则通过 `store_from_active` 转存入 `Int8CheckpointStore`：
     - `qdata`：`[L, num_slots, H, d_v, d_k]`，类型为 `torch.int8`。
     - `scale`：`[L, num_slots, H, 1, d_k]`，沿 `d_v`（`dim = -2`）取 `amax / 127`。之所以选择 `int8` 而非 `fp8_e4m3`，且沿 `d_v` 归约，是因为线性注意力的状态衰减项 $\text{diag}(\alpha)$ 按 `k-channel` 作用，且缓存命中时只执行 **单次反量化（Quantize-Once on Store, Dequantize-Once on Hit）** 拷贝回新的活跃槽位（Copy-on-Write），绝不进入循环量化误差累积回路。
3. **混合池协同驱逐（`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:57-75`）**：
   - 当全注意力 KV、滑动窗口注意力（SWA）或 Mamba 状态槽任一资源不足时，分别通过 `_evict_swa_for_device_alloc`（`EvictParams(swa_num_tokens=shortfall)`）或 `_evict_mamba_for_device_alloc`（`EvictParams(mamba_num=shortfall)`）驱动 `UnifiedRadixCache` 回收前缀树叶节点。

---

## 3. 原创 CPU 离线实验结果（`experiments/long_context_attention/`）

运行 `python experiments/long_context_attention/run.py` 生成的确定性离线对比数据（`measurement_type: "own-offline"`）如下：

### 3.1 `1M` Token 下不同 KV 格式的显存与单步 Decode HBM 延迟对比（61 层）

| KV Cache 格式 (`id`) | 单 Token 每层字节数 | `1M` 上下文 KV 显存 (`GiB`) | Decode 每步关注 Token 数 | Decode 每步读取量 (`MiB`) | Decode KV HBM 延迟 (`ms`, `@6.4 TB/s`) | 相对 GQA-8 显存压缩比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gqa8_bf16_dense` | `4096 B` | `244.0000 GiB` | `1,048,576` | `249,856.00 MiB` | `40.9364 ms` | `1.0000x` |
| `deepseek_v3_mla_bf16_dense` | `1152 B` | `68.6250 GiB` | `1,048,576` | `70,272.00 MiB` | `11.5134 ms` | `3.5556x` |
| `deepseek_v3_2_mla_fp8_dsa` | `656 B` | `39.0781 GiB` | `2,048` (`DSA`) | `78.1562 MiB` | `0.0128 ms` | `6.2439x` |
| `deepseek_v4_1_mla_fp8_dsa` | `528 B` | `31.4531 GiB` | `2,048` (`DSA`) | `62.9062 MiB` | `0.0103 ms` | `7.7576x` |
| `deepseek_v4_1_mla_fp4_dsa` | `288 B` | `17.1562 GiB` | `2,048` (`DSA`) | `34.3125 MiB` | `0.0056 ms` | `14.2222x` |

### 3.2 `1M` Token Prefill 下 Ring Attention vs DeepSpeed-Ulysses (`CP = 8`)

| 并行模式 (`id`) | 通信原语 | KV 结构 | 每层每卡通信量 (`MiB`) | 每层计算耗时 (`ms`) | 每层通信耗时 (`ms`, `@360 GB/s`) | 每层暴露总耗时 (`ms`) | 零 KV 头复制？ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `ring_cp8_gqa8` | Ring P2P | GQA-8 | `3584.0 MiB` | `7505.9994 ms` | `10.4392 ms` | `7505.9994 ms` | 是 (`True`) |
| `ring_cp8_mla_mqa1` | Ring P2P | MLA (`h_kv=1`) | `1008.0 MiB` | `7505.9994 ms` | `2.9360 ms` | `7505.9994 ms` | 是 (`True`) |
| `ulysses_cp8_gqa8` | 4x All-to-All | GQA-8 | `7616.0 MiB` | `7505.9994 ms` | `22.1832 ms` | `7528.1826 ms` | 是 (`True`) |
| `ulysses_cp8_mla_mqa1` | 4x All-to-All | MLA (`h_kv=1`) | `7616.0 MiB` | `7505.9994 ms` | `22.1832 ms` | `7528.1826 ms` | **否 (`False`, 8x KV 复制)** |

### 3.3 SGLang Hybrid Mamba-2 检查点间隔与分支截断重算（Replay）权衡

对于 `45` 层 Mamba-2 + `15` 层 Attention 的混合模型，单个活跃 BF16 状态槽占 **`46.5820 MiB`**，而 `Int8CheckpointStore` 压缩后占 **`24.7852 MiB`**（`16 GiB` 池可容纳 **`661` vs `351`** 个节点，容量提升 **`1.8832x`**）。当请求在 `Token 130,000` 处发生非对齐前缀分支时：

| Checkpoint 间隔 (`tokens`) | 恢复的最近检查点位置 | 需 Replay 重算的 Token 数 | 分支首包重算惩罚 (`ms`, `@32K tok/s`) | `1M` 序列检查点总数 | `1M` 序列 `int8` 检查点总显存 (`GiB`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `2,048` | `129,024` | `976` | `29.7852 ms` | `512` | `12.3926 GiB` |
| `8,192` | `122,880` | `7,120` | `217.2852 ms` | `128` | `3.0981 GiB` |
| `32,768` | `98,304` | `31,696` | `967.2852 ms` | `32` | `0.7745 GiB` |
| `65,536` | `65,536` | `64,464` | `1967.2852 ms` | `16` | `0.3873 GiB` |

---

## 4. 解决方案架构师设计指南与可证伪问题

1. **架构师选型法则**：
   - 在 **MLA (`h_kv = 1`)** 模型上做超长上下文并行时，优先采用 **Ring Attention / Striped Ring CP**（传输压缩后的 `576` 维潜变量），避免在 `CP > 1` 时直接使用纯 Ulysses Head Sharding 导致 KV 潜变量被全量复制；若需跨节点结合，应采用 **Hybrid Ulysses (节点内 Q 头切分) + Ring (跨节点 P2P KV 传递)**。
   - 在 **Hybrid Mamba-Transformer** Agentic 多轮工具调用场景中，将 `MambaCheckpointPool` 间隔设为 **`2,048 ~ 8,192` Tokens** 并开启 `int8` 检查点压缩与 Host HiCache 分层卸载，可将分支重算 TTFT 惩罚控制在 `30 ~ 220 ms` 以内，同时将 `1M` 序列的 Mamba 检查点显存控制在 `3.1 ~ 12.4 GiB`。
2. **后续可证伪实验问题**：
   - 在 DeepSeek-V4.1 的双缓存模式（主缓存 `528 B` FP8 + 辅助缓存 `288 B` FP4）下，如何根据 Token 的历史注意力累积分数动态将冷 Page 降级（Demote）至 `extra_k_cache` FP4 而不触发 `FlashMLASchedMeta` 的重新初始化开销？
