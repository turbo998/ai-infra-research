# FlashMLA Blackwell SM100 DeepSeek-V4.1 融合 `Norm+RoPE+Attn+RoPE+Cast` 内核与零 Bank 冲突 `KVBlockDequantizer`

> **English Version**: [flashmla-sm100-v41-fused-norm-rope-attn-dequant-kv.en.md](flashmla-sm100-v41-fused-norm-rope-attn-dequant-kv.en.md)  
> **证据等级 (Evidence Class)**: `implementation`（一手源码核验） + `own-offline`（确定性标准库仿真）  
> **上游仓库与提交**: `deepseek-ai/FlashMLA` commits [`07a1089857b63e74e3133630c02b083b75e8d4b2`](https://github.com/deepseek-ai/FlashMLA/commit/07a1089857b63e74e3133630c02b083b75e8d4b2) (2026-09-10) & [`ba89a3466e9470ad08ab39738d4e7bb66989e1e7`](https://github.com/deepseek-ai/FlashMLA/commit/ba89a3466e9470ad08ab39738d4e7bb66989e1e7) (2026-09-15)  
> **关联实验**: [`experiments/flashmla_sm100_v41_fused_dequant_sim/run.py`](../../experiments/flashmla_sm100_v41_fused_dequant_sim/run.py) (`--check` 通过)

---

## 1. 架构背景与核心工程矛盾

在 DeepSeek-V4 / V4.1（`d_qk = d_v = 512, h_kv = 1, d_rope = 64`）Token 级精确 Top-K 稀疏注意力中，若将 `q_b_proj` 输出至 `wv_proj` 输入之间的计算拆分为独立的 `RMSNorm`、`Q RoPE`、`Core Attention`、`O RoPE` 与 `O FP8 Quantization` 5 个内核，每次内核边界都会引入额外的 HBM 读写往返与启动开销。同时，在解码阶段混合读取 `V41` (FP8) 主 KV 缓存与 `V41_FP4` (FP4 E2M1 + E4M3 缩放) 辅助 KV 缓存时，TMA `gather4` 收集至共享内存的原位反量化极易遭遇 4 路共享内存 Bank 冲突与 `ptxas` 跨格式寄存器生命周期膨胀。

---

## 2. 一手源码机制拆解 (`deepseek-ai/FlashMLA`)

### 2.1 四代分页量化 KV 缓存格式与 `TMA_K_STRIDE >= 256` 约束 (`csrc/kernels/kv_cache_format.h`)

`KVCacheFormat<MT>` 将每个分页块存储为两个按行对齐的字节数组（量化数据区 `[page_block_size, TMA_K_STRIDE]` 与缩放因子区 `[page_block_size, NUM_SCALES_EACH_TOKEN * SCALE_BYTES]`）：

| 模型格式 (`ModelType`) | `D_QK` | `D_FP8` / `D_FP4` / `D_BF16` | `QUANT_TILE_SIZE` | `NUM_SCALES` × `SCALE_BYTES` | `TMA_K_STRIDE` | `BYTES_PER_TOKEN` | 相对 V3.2 压缩比 |
|---|---:|---|---:|---|---:|---:|---:|
| `V32` | 576 | `512 / 0 / 64` | 128 | `4 × 4B (fp32 内联)` | `656 B` | **`656 B`** | `1.0000x` |
| `V4` | 512 | `448 / 0 / 64` | 64 | `8 (7+1 pad) × 1B (ue8m0)` | `576 B` | **`584 B`** | `1.1233x` |
| `V41` | 512 | `512 / 0 / 0` | 32 | `16 × 1B (ue8m0)` | `512 B` | **`528 B`** | `1.2424x` |
| `V41_FP4` | 512 | `0 / 512 / 0` | 16 | `32 × 1B (e4m3)` | `256 B` | **`288 B`** | **`2.2778x`** |

- **`TMA_K_STRIDE >= 256` 硬件不变量**：`V41_FP4` 的每 Token 量化数据区恰好为 `512 / 2 = 256 B`（`TMA_K_STRIDE = 256 B`），确保 `int32` TMA 行坐标在不溢出的前提下覆盖完整超长 KV 缓存池（相对 `V41` FP8 的 `528 B` 进一步获得 **`1.8333x`** 压缩）。

### 2.2 零 Bank 冲突 Warpgroup 反量化与 `e2m1 × e4m3` 精确 BF16 乘法 (`csrc/kernels/sm100/dequant_utils.cuh`, `helpers.h`)

1. **`for_each_kv_block<Orig, Extra>` 双循环隔离**：当 `Orig` (`V41` FP8) 与 `Extra` (`V41_FP4` FP4) 格式不同时，显式拆分为 `[begin, min(num_orig_blocks, end))` 与 `[max(begin, num_orig_blocks), end)` 两个 `CUTE_NO_UNROLL` 循环，防止 `ptxas` 将两种反量化路径的寄存器活跃区间合并。
2. **`KVBlockDequantizer` 的 `LDS.32` 四分位错位填充 (`RAW_TOKEN_SMEM_STRIDE / 4 % 16 == 8`)**：
   - 128 线程 Warpgroup 按 `GROUP_SIZE = 8` 分为 `NUM_GROUPS = 16` 组，每组 8 个线程每步处理 `ELEMS_PER_STEP = 64` 个元素（FP4 下每线程读取 `4 B` = 8 个 E2M1，转为 `16 B` = 8 个 BF16）。
   - 若不填充，1-CTA (`256 B`) 或 2-CTA (`128 B`) 下 `gather4` 写入的连续 4 行满足 `stride / 4 % 16 == 0`，同 Warp 内 4 组线程的 `LDS.32` 会落入完全相同的共享内存 Bank（**`4-way` Bank 冲突**）。
   - 通过让 TMA Box 宽度比数据宽 `+32 B`（1-CTA `288 B`、2-CTA `160 B`，超出部分由 TMA 自动补零），满足 `RAW_TOKEN_SMEM_STRIDE / 4 % 16 == 8`，使 `gather4` 组内的 4 行分别起始于 32 个 Bank 的 4 个不同四分位（0、8、16、24），实现 **`1-way` 零冲突 `LDS.32`**。
   - 在 `fused_norm_rope_attn_rope_cast_fwd` 的 `LDS.128` 通用反量化路径中，每行填充 `+16 B`（`RAW_TOKEN_SMEM_STRIDE = 272 B / 144 B` (FP4) 或 `528 B / 272 B` (FP8)），使 `RAW_TOKEN_SMEM_STRIDE / 16` 为奇数（`17 / 9 / 33 / 17`），同样消除 `LDS.128` 冲突。
3. **`SW128` K-major 异或交织与 `e2m1 × e4m3` 无损 BF16 精确性**：
   - 写出偏移 `dst_offset = (group_idx / 8) * 1024 + row_in_atom * 128 + (idx_in_group ^ row_in_atom) * 16` 将 16 B 通道号与 `row_in_atom = group_idx % 8` 异或，使 8-lane Wavefront 的 `STS.128` 100% 无冲突。
   - `e2m1`（至多 2 位有效尾数）与 `e4m3`（至多 4 位有效尾数）相乘最多产生 `2 + 4 = 6` 位有效位，完全落入 BF16 的 8 位有效尾数（`1 + 7` 位）内：遍历全部 `2,032` 对非 NaN 有穷 `(e2m1, e4m3)` 组合，IEEE-754 FP32 乘积的低 16 位恒为 `0x0000`（**`0` 舍入误差**）。

### 2.3 SM100 全融合 `Norm+RoPE+Attn+RoPE+Cast` 的 TMEM 布局、权重重排与 `st.async` LLP64 修复

- **TMEM 512 列紧致布局 (`core_attn/config.h:tmem_cols`)**：
  - `H_Q = 64` (`CLUSTER_SIZE = 1, FOLD_FACTOR = 2, B_TOPK = 64, NUM_MRGEMM_RAILS = 2, NUM_P_BUFS = 2`)：`O = [0, 256)`、`Q = [256, 384)`、`P_0 = [384, 448)`、`P_1 = [448, 512)`，总计恰为 **`512` 列**；由于双缓冲 `P` 满足 `P[i] <- O[i-2] <- S[i-2] <- P[i-2] free` 因果链，`NEED_TP_EMPTY_BAR = 0`（省去 P 空闲屏障）。
  - `H_Q = 128` (`CLUSTER_SIZE = 2, FOLD_FACTOR = 2, B_TOPK = 96, NUM_P_BUFS = 1`)：`O = [0, 256)`、`Q = [256, 384)`、`P_0 = [384, 480)`，总计 **`480` 列**（`<= 512`，`NEED_TP_EMPTY_BAR = 1`）。
- **输入/输出权重预重排双射 (`permute_q_b_proj`, `permute_wv_proj`)**：
  - `permute_q_b_proj_kernel` 将 `row_idx = head_idx * 512 + head_dim_idx` 映射至 `out_row_idx = (head_dim_idx % 16) + head_idx * 16 + (head_dim_idx // 16) * (h_q * 16)`（在 `65,536` 行上构成严格双射），使 `q_b_proj` GEMM 直接输出跨 Head 交错的 16 元素块。
  - `permute_wv_proj_kernel` 在每个 `wv_group` (`wv_group_size = 8, D_O = 512`) 内将 32 元素块 `(head_idx, head_dim_chunk_idx)` 映射至 `head_dim_chunk_idx * 8 + head_idx`（`128` 块严格双射），使融合内核输出的 `out_fp8` 无需转置即可直接送入 `wv_proj`。
- **128-bit `st.async` 的 MSVC LLP64 截断修复 (`ba89a3466e`, `intrinsics.cuh`)**：将 `reinterpret_cast<const long2*>(&data)` 修正为 `longlong2`，消除 Windows MSVC LLP64 下 `sizeof(long) == 4` (`sizeof(long2) == 8 B`) 造成的高 64 位静默丢失。

---

## 3. 确定性离线实验验证 (`flashmla_sm100_v41_fused_dequant_sim`)

运行 `python experiments/flashmla_sm100_v41_fused_dequant_sim/run.py --check`，核心验证指标如下：

| 验证维度 | 基线 / 未优化配置 | FlashMLA SM100 V4.1 优化配置 | 核验结果 |
|---|---|---|---|
| KV 缓存单 Token 字节数 | `V32`: `656 B` / `V4`: `584 B` / `V41`: `528 B` | `V41_FP4`: **`288 B`** (`TMA_K_STRIDE = 256 B`) | **`2.2778x`** (vs V3.2) / **`1.8333x`** (vs V41 FP8) |
| `KVBlockDequantizer` `LDS.32` Bank 冲突 | 未填充 (`256 B` / `128 B`): **`4-way`** 冲突 | `+32 B` 填充 (`288 B` / `160 B`): **`1-way`** | **消除 4 路共享内存 Bank 冲突** |
| `e2m1 × e4m3 -> bf16` 精度 | 2,032 对有穷非 NaN 组合 | 低 16 位尾数全为 `0x0000` (`inexact = 0`) | **100% 位精确 (`0` 舍入误差)** |
| 融合内核 TMEM 列占用 | `H_Q=64` (`B_TOPK=64`, 双缓冲 `P`) / `H_Q=128` (`B_TOPK=96`) | **`512` 列** (`NEED_TP_EMPTY_BAR=False`) / **`480` 列** | 严格满足 SM100 `<= 512` 列上限 |
| 权重重排 & `st.async` | `permute_q_b_proj` (`65,536` 行) / `permute_wv_proj` (`128` 块) | 双射校验 `True`；`longlong2 = 16 B` vs `long2 = 8 B` | 零转置开销 + 消除 LLP64 截断 |
