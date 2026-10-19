# DeepSeek MLA 吸收潜变量 KV 布局 (`V41` FP8 `528B` / FP4 `288B`) 与 `FlashMLA` SM90/SM100 变长 Split-KV 2-CTA 负载均衡调度剖析

> **证据等级声明**：本文对 `KVCacheFormat<ModelType::V41>`、`KVCacheFormat<ModelType::V41_FP4>`、`KVCachePart<F, 2, CTA>`、`KVFormatPair` 及 `get_mla_metadata_kernel` (`DecodingSchedMeta` / `KU_STG_256`) 的分析基于 `deepseek-ai/FlashMLA`（commit `0f11e178da1f67becb087a324b400854931ec3e3`）源码文件 `csrc/cuda_kernels/kv_cache_format.h`（blob `e730426a72f1d1cec4d063c834a479c541b2c299`）与 `csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`（blob `2307c47f3dbd731510eae2ada4f9a502e67eb420`）的一手核验（`implementation`）。矩阵吸收算术强度 Roofline、2-CTA 分半非对称 TMA 传输字节分解以及偏斜变长批次下的 `get_mla_metadata` 波次失衡消除数据来自确定性离线仿真算例 `experiments/flashmla_paged_decode_scheduling/run.py`（`own-offline`），不声称真机 SM90/SM100 硬件实测。

---

## 1. 为什么矩阵吸收（Matrix Absorption）能把 Decode Attention 变成高算术强度 GEMM？

在传统多头注意力（MHA）或分组查询注意力（GQA）中，自回归 Decode 阶段（每请求仅生成 $q_{\text{len}} = 1$ 个新 Token）受困于极低的算术强度（Arithmetic Intensity, AI）。以 $H_q = 128$、$d_{\text{qk}} = 128$、$d_v = 128$ 为例：
- **标准 MHA ($H_{\text{kv}} = 128$, BF16)**：每历史 Token 需存储 $128 \times (128 + 128) \times 2 = 65,536\text{ B}$ KV Cache，而与单个 Query Token 计算仅产生 $2 \times 128 \times (128 + 128) = 65,536\text{ FLOPs}$，算术强度仅为 **$1.0\text{ FLOP/B}$**；
- **标准 GQA-8 ($H_{\text{kv}} = 8$, BF16)**：8 个键值头各自被 $G = H_q / H_{\text{kv}} = 16$ 个查询头共享，每 Token KV Cache 降至 $4,096\text{ B}$，算术强度升至 **$16.0\text{ FLOP/B}$**，但仍远低于 H100 SXM (`~295 FLOP/B`) 与 Blackwell B200 (`~281 FLOP/B` FP16 / `~562 FLOP/B` FP8) 的 Tensor Core 算术强度拐点（Ridge Point）。

**Multi-Head Latent Attention (MLA)** 通过低秩联合压缩与**矩阵吸收（Matrix Absorption）**彻底改写了这一物理边界：
1. **未吸收形态（Unabsorbed Prefill 形态）**：将输入隐藏状态投影为低维潜向量 $c_t^{KV} \in \mathbb{R}^{d_c}$（DeepSeek-V2/V3 中 $d_c = 512$）与解耦旋转位置编码键 $k_t^R \in \mathbb{R}^{d_r}$（$d_r = 64$）。若在 Decode 时显式调用 $W^{UK}, W^{UV}$ 将 $c_t^{KV}$ 还原为 128 个头的 $K, V$，则每 Token 需在片上展开 $81,920\text{ B}$，重新退化回 $\text{AI} = 1.0\text{ FLOP/B}$。
2. **矩阵吸收形态（Absorbed Decode 形态）**：利用线性结合律，将键上投影矩阵 $W^{UK} \in \mathbb{R}^{(H_q \cdot d_{\text{nope}}) \times d_c}$ 吸收进查询投影 $q_t^{C} = (q_t^{\text{nope}} W^{UK}) \in \mathbb{R}^{H_q \times d_c}$，并将值上投影矩阵 $W^{UV}$ 吸收进注意力输出投影 $O = (\text{AttnScores} \cdot C^{KV}) W^{UV}$。此时**所有 $H_q = 128$ 个查询头共享同一个潜变量向量 $c_t^{KV}$（等效 $H_{\text{kv}} = 1, G = 128$）**！

$$\text{AI}_{\text{MLA-Absorbed}} = \frac{2 \cdot H_q \cdot (d_{\text{qk}} + d_v)}{\text{Bytes\_Per\_Token}(c_t^{KV}, k_t^R)}$$

当 $H_q = 128$ 时：
- 在 DeepSeek-V3 BF16 吸收形态（$d_{\text{qk}} = 576, d_v = 512, \text{Bytes} = 1,152\text{ B}$）下，每历史 Token 提供 $278,528\text{ FLOPs}$，算术强度跃升至 **$241.78\text{ FLOP/B}$**（比 GQA-8 高 **$15.1\times$**）；
- 在 `FlashMLA` 最新 `ModelType::V41` (FP8, `528 B/tok`) 与 `ModelType::V41_FP4` (FP4, `288 B/tok`) 下（$d_{\text{qk}} = 512, d_v = 512$，每 Token 计算量 $262,144\text{ FLOPs}$），算术强度进一步飙升至 **$496.48\text{ FLOP/B}$** 与 **$910.22\text{ FLOP/B}$**，首次让单步 Decode Attention 跨越 Hopper FP8 与 Blackwell B200 Tensor Core 的计算受限（Compute-Bound）拐点！

---

## 2. 一手源码走读：`csrc/cuda_kernels/kv_cache_format.h` 与 2-CTA 潜变量分片

在 `deepseek-ai/FlashMLA`（commit `0f11e178da1f67becb087a324b400854931ec3e3`）的 `csrc/cuda_kernels/kv_cache_format.h`（blob `e730426a72f1d1cec4d063c834a479c541b2c299`，共 70 行）中，明确定义了 `ModelType::V41`（FP8 块缩放潜变量）与 `ModelType::V41_FP4`（FP4 块缩放潜变量）的字节级内存布局及 Blackwell `cta_group::2` 双 CTA 协同切分契约：

```cpp
// Source: deepseek-ai/FlashMLA csrc/cuda_kernels/kv_cache_format.h
template<ModelType MT>
struct KVCacheFormat {
    static constexpr ModelType MODEL_TYPE = MT;
    static constexpr uint32_t D_QK = 512;
    static constexpr uint32_t D_ROPE = 64;
    static constexpr uint32_t D_NOPE = D_QK - D_ROPE;

    static constexpr uint32_t QUANT_TILE_SIZE = MT == ModelType::V41 ? 32 : 16;
    static constexpr uint32_t NUM_SCALES_EACH_TOKEN = D_QK / QUANT_TILE_SIZE;

    using scale_t = std::conditional_t<MT == ModelType::V41, cutlass::float_ue8m0_t, cutlass::float_e4m3_t>;
    static constexpr uint32_t QUANT_BYTES = MT == ModelType::V41 ? D_QK : D_QK / 2;
    static constexpr uint32_t BYTES_PER_TOKEN = QUANT_BYTES + NUM_SCALES_EACH_TOKEN * sizeof(scale_t);
};
```

源码揭示了三个极其关键的硬件协同设计：
1. **细粒度块缩放潜变量（`528 B` vs `288 B`）**：
   - `ModelType::V41`（FP8）：将 $D_{\text{QK}} = 512$（包含 $D_{\text{NOPE}} = 448$ 与 $D_{\text{ROPE}} = 64$）按 `QUANT_TILE_SIZE = 32` 切分为 16 个子块，每个子块配 1 个 1 字节无符号纯指数缩放因子 `cutlass::float_ue8m0_t`，单 Token 总字节数为 `512 + 16 × 1 = 528 B`（相比 GQA-8 BF16 的 `4,096 B` 压缩 **$7.758\times$**）。
   - `ModelType::V41_FP4`（FP4）：将 $D_{\text{QK}} = 512$ 按 `QUANT_TILE_SIZE = 16` 切分为 32 个子块，潜变量尾数压缩为 `256 B`（`0.5 B/elem`），每个 16 元素子块配 1 个 `cutlass::float_e4m3_t`（含 3-bit 尾数，與 NVFP4 一致）缩放因子（共 `32 B`），单 Token 总字节数压至 **`288 B`**（相比 GQA-8 BF16 压缩 **$14.222\times$**）。
2. **2-CTA Cluster 非对称 TMA 载荷切分（`KVCachePart<F, 2, CTA>`）**：
   在 Blackwell `SM100` 的 `tcgen05.mma.cta_group::2` 模式下，两个相邻 CTA（`CTA 0` 与 `CTA 1`）共享同一个 $D_{\text{QK}} = 512$ 潜变量的不同特征半区（每个 CTA 负责 `PART_D_QK = 256` 维）：
   ```cpp
   template<typename F, int CTA>
   struct KVCachePart<F, 2, CTA> {
       using Format = F;
       static constexpr uint32_t PART_D_QK = F::D_QK / 2;
       static constexpr uint32_t PART_QUANT_BYTES = F::QUANT_BYTES / 2;
       static constexpr uint32_t PART_NUM_SCALES = F::NUM_SCALES_EACH_TOKEN / 2;
       static constexpr uint32_t START_OFFSET = CTA == 0 ? 0 : F::QUANT_BYTES / 2;
       static constexpr uint32_t SCALE_START_OFFSET = F::QUANT_BYTES + (CTA == 0 ? 0 : F::NUM_SCALES_EACH_TOKEN / 2);
       static constexpr uint32_t RAW_BYTES_PER_TOKEN = PART_QUANT_BYTES + (CTA == 0 ? 0 : F::NUM_SCALES_EACH_TOKEN * sizeof(typename F::scale_t));
       static constexpr bool SEPARATE_SCALES = CTA == 0;
   };
   ```
   なぜ `CTA 0` 与 `CTA 1` 的 `RAW_BYTES_PER_TOKEN` 不同？因为在内存布局中，每个 Token 的 `512 B`（或 `256 B`）量化潜变量在前，`16 B`（或 `32 B`）缩放因子紧随其后排在尾部！因此：
   - **`CTA 1`（右半特征 `256..511`）**：其量化数据 `[QUANT_BYTES/2 .. QUANT_BYTES)` 与尾随的缩放因子 `[QUANT_BYTES .. BYTES_PER_TOKEN)` 在物理地址上**完全连续**，只需单次連続 TMA 传输即可同时取回右半量化数据与全量缩放因子（`V41` 下 `CTA 1` 传输 `256 + 16 = 272 B`，`V41_FP4` 下传输 `128 + 32 = 160 B`，`SEPARATE_SCALES = false`）；
   - **`CTA 0`（左半特征 `0..255`）**：其量化数据位于 `[0 .. QUANT_BYTES/2)`，与尾部的缩放因子不相邻，因此 `RAW_BYTES_PER_TOKEN` 仅计入左半量化字节（`V41` 为 `256 B`，`V41_FP4` 为 `128 B`），标记 `SEPARATE_SCALES = true` 单独加载其 `PART_NUM_SCALES` 缩放因子。
3. **混合精度双格式对（`KVFormatPair<V41, V41_FP4>`）**：支持在同一推理实例中对近期活跃块或高敏感层保留 `V41` FP8 (`528 B/tok`)、对长程历史块采用 `V41_FP4` (`288 B/tok`)。当采用 1:2 混合配比时，平均每 Token 仅占 **`368 B`**（相比 GQA-8 BF16 压缩 **$11.13\times$**，算术强度达 **`712.35 FLOP/B`**）。

| KV Cache 格式 | 每 Token 字节数 | `CTA 0` / `CTA 1` 字节切分 | 相比 GQA-8 压缩比 | Decode 算术强度 (`FLOP/B`) | H100 可达有效吞吐 (`TFLOPS`) | B200 可达有效吞吐 (`TFLOPS`) |
|---|---:|---:|---:|---:|---:|---:|
| **`MHA_128_BF16`** | `65,536 B` | `65,536 / 0` (1-CTA) | `0.062x` | `1.00` | `2.85` | `6.88` |
| **`GQA_8_BF16`** | `4,096 B` | `4,096 / 0` (1-CTA) | `1.000x` | `16.00` | `45.56` | `110.08` |
| **`MLA_UNABSORBED_BF16`** | `81,920 B` | `81,920 / 0` (1-CTA) | `0.050x` | `1.00` | `2.85` | `6.88` |
| **`MLA_ABSORBED_BF16_D576`** | `1,152 B` | `1,152 / 0` (1-CTA) | `3.556x` | `241.78` | `688.46` | `1,663.43` |
| **`FLASHMLA_V41_FP8_528B`** | **`528 B`** | **`256 B / 272 B`** (2-CTA) | **`7.758x`** | **`496.48`** | **`1,154.40`** | **`2,880.00`** |
| **`FLASHMLA_V41_FP4_288B`** | **`288 B`** | **`128 B / 160 B`** (2-CTA) | **`14.222x`** | **`910.22`** | **`1,154.40`** | **`2,880.00`** |
| **`FLASHMLA_HYBRID_V41_FP8_FP4`** | **`368 B`** | **`171 B / 197 B`** (2-CTA) | **`11.130x`** | **`712.35`** | **`1,154.40`** | **`2,880.00`** |

---

## 3. 一手源码走读：`get_mla_metadata_kernel` 变长 Split-KV 持久波次均衡

在真实的 Continuous Batching 生产负载中，同一 Decode 批次内的序列长度往往高度偏斜（例如 4 个长文档 Agent 请求达 `64K` tokens，其余 60 个短对话请求仅 `512–4,096` tokens）。如果按“每请求分配 1 个 CTA/Part”或固定切分数（Uniform Split-K）发射 Kernel，负责 `64K` 请求的 SM 将独自迭代 `1,024` 个 `block_size=64` 的 KV 块，而其余 73 个 SM Pair 早已空闲等待，造成严重的 **Wave Quantization 尾波空转**。

`deepseek-ai/FlashMLA` 在 `csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`（blob `2307c47f3dbd731510eae2ada4f9a502e67eb420`）中实现了一个单 Warp（`__launch_bounds__(32, 1, 1)`）的设备端精确工作窃取分区规划器 `get_mla_metadata_kernel`：

```cpp
// Source: deepseek-ai/FlashMLA csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
for (int i = lane_id; i < b; i += 32) {
    int num_blocks = ceil_div(seqlens_k_ptr[i], block_size_n);
    total_num_blocks += num_blocks + fixed_overhead_num_blocks;
    seqlens_k_smem[i] = num_blocks;
}
#pragma unroll
for (int i = 16; i >= 1; i /= 2) {
    total_num_blocks += __shfl_xor_sync(0xffffffff, total_num_blocks, i);
}
if (threadIdx.x == 0) {
    int payload = ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;
    int now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
    for (int i = 0; i < num_sm_parts; ++i) {
        intrem = payload;
        // Greedily pack whole short sequences and split long sequences across SM parts...
```

### 3.1 关键调度不变量与 256-Bit 向量化元数据写出

1. **计入固定收尾开销的代价模型（`fixed_overhead_num_blocks = 5`）**：
   每在某个 SM Part 上开启一段新的序列分片（Split），除了遍历 `num_blocks` 个 KV 页外，还必须付出 Epilogue 写回局部部分和（Partial Output + Log-Sum-Exp `LSE`）以及最后 `flash_mla_combine` 跨分片归约的固定延迟，等效于 `fixed_overhead_num_blocks = 5` 个 KV 块的耗时。因此单序列总代价建模为 `num_blocks + 5`，全批次通过 Warp 蝶形归约 `__shfl_xor_sync` 求和后，计算每个 SM Part 的目标载荷配额：
   $$\text{payload} = \left\lceil \frac{\sum_{i=0}^{B-1} (\text{blocks}_i + c_{\text{fixed}})}{N_{\text{sm\_parts}}} \right\rceil + c_{\text{fixed}}$$
   其中在 Hopper H100 (`132 SMs`, 1-CTA/part) 上 $N_{\text{sm\_parts}} = 132$；在 Blackwell B200 (`148 SMs`, `cta_group::2` 即 2 SMs/part) 上 $N_{\text{sm\_parts}} = 74$。
2. **短请求打包（Packing）与长请求跨 SM 切分（Splitting）的统一游标状态机**：
   - 若当前请求剩余块数 `remain_blocks + fixed_overhead <= rem + (rem == payload)`，则将该请求剩余部分完整装入当前 SM Part，游标推进至下一请求 `now_idx++`；
   - 若剩余预算 `rem > fixed_overhead_num_blocks`，则从当前长序列切出 `rem - fixed_overhead_num_blocks` 个块交给当前 SM Part，并将该请求的 `now_n_split_idx++` 记录至 `num_splits_ptr` 前缀和数组中，供后续 `Combine` Kernel 精确索引。
3. **Blackwell `KU_STG_256` 32 字节对齐写出**：
   每个 SM Part 的任务描述符 `DecodingSchedMeta` 恰好封装 8 个 `int32` 字段（`begin_req_idx, end_req_idx, begin_block_idx, end_block_idx, begin_n_split_idx, _, _, _` = `32 Bytes`），在 `SM100a` 上直接调用 `ku::st_global_256_v2(&tile_scheduler_metadata_ptr[i], *reinterpret_cast<ku::b256_t*>(&meta))` 单指令落盘。

### 3.2 定量实验核验：偏斜 `B=64` 批次 (`512..65,536` Tokens) 下的 `6.0x` 加速

在 `experiments/flashmla_paged_decode_scheduling/run.py` 中，我们对 `B=64` 混合负载（4 条 `65,536`、8 条 `16,384`、16 条 `4,096`、36 条 `512` tokens，总计 `472,576` KV tokens = `7,384` 个 64-token 块）进行了精确复现：

| 调度策略 | 目标硬件单元数 `num_sm_parts` | 总分片数 `total_splits` | 最大单单元负载 (`blocks`) | 平均单元负载 (`blocks`) | 波次失衡率 (`Imbalance %`) | 端到端 Kernel 延迟 (`us`) | 有效算力 (`TFLOPS`) | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`STATIC_REQ_PER_PART_74`** (静态 1请求/Part) | `74` (B200 2-CTA) | `64` | `1,029` | `104.11` | `89.88%` | `91.090 us` | `1,360.01` | `1.000x` |
| **`UNIFORM_SPLIT4_74`** (固定 4-Split) | `74` (B200 2-CTA) | `256` | `307` | `117.62` | `61.69%` | `37.768 us` | `3,280.10` | `2.412x` |
| **`FLASHMLA_GET_MLA_METADATA_SM100_74`** | **`74` (B200 2-CTA)** | **`131`** | **`110`** | **`108.635`** | **`1.24%`** | **`15.182 us`** | **`8,159.86`** | **`6.000x`** |
| **`FLASHMLA_GET_MLA_METADATA_SM90_132`** | **`132` (H100 1-CTA)** | **`188`** | **`64`** | **`63.061`** | **`1.47%`** | **`15.448 us`** | **`8,019.35`** | **`5.897x`** |

---

## 4. 解决方案架构师行动指南与可证伪问题

1. **何时启用 `V41_FP4` (`288 B/tok`) vs 混合 `KVFormatPair<V41, V41_FP4>`？**
   - 对于纯吞吐优先或超长上下文（`64K–128K`）Agent 推理，`V41_FP4` 将单卡 `80 GiB / 192 GiB` HBM 的可容纳并发上下文容量比 FP8 MLA 再翻 **`1.83x`**（比 GQA-8 BF16 提升 **`14.22x`**）；
   - 对于代码生成或数学推理等对近期注意力头相位敏感的任务，使用 `KVFormatPair<V41, V41_FP4>` 将最近 `4K–8K` Sliding Window 保持在 `V41` FP8 (`528 B/tok`)，历史归档块转为 `V41_FP4` (`288 B/tok`)。
2. **CUDA Graph 捕获注意事项**：
   `get_mla_metadata_kernel` 的输出直接写入固定形状的 `tile_scheduler_metadata_ptr[num_sm_parts]` 与 `num_splits_ptr[batch_size + 1]` 缓冲区，主 Decode Kernel 通过指针间接读取 `DecodingSchedMeta`，因此 **`get_mla_metadata_kernel + flash_mla_decode + flash_mla_combine` 三阶段流水线可完整捕获进同一张 CUDA Graph**，无需任何 Host-Device 同步！
