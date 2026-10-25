# FlashInfer SM103 QSA NVFP4 两波持久化遍历与单块融合归约、SM120 mma.sync 解码、Hidden-4096 4-Route Mega-mHC、SM120 拓扑加权 PCIe CE 流水环、Kimi-K3 L2::evict_first 与 TileLang 屏障修复

> **English Version**: [flashinfer-sm103-qsa-persistent-mhc4096-sm120-piped-kimi-k3-tilelang.en.md](flashinfer-sm103-qsa-persistent-mhc4096-sm120-piped-kimi-k3-tilelang.en.md)  
> **关联问题 ID**: `flashinfer-sm103-qsa-persistent-mhc4096-sm120-piped-kimi-k3-tilelang` (Phase 74, Priority 83)  
> **关联可复现实验**: [`experiments/flashinfer_sm103_qsa_persistent_mhc4096_sm120_piped_kimi_k3_tilelang_sim/run.py`](../../experiments/flashinfer_sm103_qsa_persistent_mhc4096_sm120_piped_kimi_k3_tilelang_sim/run.py)  
> **证据分类**: `implementation` / `author-benchmark` (上游源码与 PR 实测) + `own-offline` (确定性标准库仿真器)

---

## 1. 背景与架构痛点

在 Blackwell 数据中心与工作站推理（`sm_100a` B200、`sm_103a` GB300、`sm_107a` Rubin 与 `sm_120a` RTX PRO 6000 Blackwell Server Edition）中，稀疏注意力（CAKE QSA NVFP4）、超连接残差（Mega-mHC）、单机 PCIe 集合通信（PCIe CE Ring All-Reduce）、LatentMoE 专家流水与 TileLang 算子编译器在 `2026-10-22` 至 `2026-10-25` 期间推进了六项底层微架构演进：

1. **SM103 两波 Grid 尾延迟与 `tcgen05` 融合归约开销**（[`flashinfer@131f85942edb`](https://github.com/flashinfer-ai/flashinfer/commit/131f85942edb00c9258bc6e393ccbe57b88b55bc)）：在 GB300（`152` SMs）上运行 `rows=128`、2 KV heads、`2,051` 选中 token（`33` 个 64-token 块）时，传统 `plain_splits=1` 启动 `256` 个 CTA，跨 `152` SMs 形成 2 波（第二波仅 `104` SMs 活跃，`48` SMs 闲置），最长路径需串行跑完 `2 × 33 = 66` 个块。同时，小批量（`rows=1,2`）的 `tcgen05` 融合模式在 `grid.sync()` 后由 32 条线程每次仅处理 8 列（4 次迭代），浪费了共享内存带宽。
2. **SM120 无 TMEM/`tcgen05` 下的 QSA NVFP4 解码与切分过密**（[`flashinfer@09a77503f004b8f36a38d23074fc5da6877de811`](https://github.com/flashinfer-ai/flashinfer/commit/09a77503f004b8f36a38d23074fc5da6877de811)）：`sm_120a`（`188` SMs，每 SM 可驻留 2 个 `99 KiB` 共享内存的 decode CTA，共 `376` 个协同 CTA）不具备 `tcgen05` 与 TMEM；若沿用通用切分公式在 `rows=4,8,16,32` 产生 `33,23,11,5` 个 split，切分归约开销远超并行收益。
3. **GLM-5.3-Flash Hidden-4096 4-Route Mega-mHC 推理与 32 位步幅溢出**（[`flashinfer@847172573ad1d5a929a8e7fb67aa7c072f8f7dcb`](https://github.com/flashinfer-ai/flashinfer/commit/847172573ad1d5a929a8e7fb67aa7c072f8f7dcb)）：此前 Mega-mHC 仅支持 `H ∈ {1280, 2560, 3072}`，缺少 `H=4096`（`K = 4 × 4096 = 16,384`）持久化算子，且 TMA 描述符构造中的 `rows * stride` 乘法在 32 位整数下会在 `T >= 131,072` 时发生无符号溢出。
4. **SM120 双路 CPU 下跨 Socket PCIe 慢跳限制 Ring All-Reduce 吞吐**（[`flashinfer@68e3e09ad3a52cbbd6b39d159d2fad7d1653073d`](https://github.com/flashinfer-ai/flashinfer/commit/68e3e09ad3a52cbbd6b39d159d2fad7d1653073d)）：8 卡 SM120 跨 2 颗 CPU Socket 时，`3→4` 与 `7→0` 两跳经过跨 Socket 互连（`NVML SYSTEM`），等尺寸单分片环被最慢跨 Socket 跳卡住。
5. **Kimi-K3 Stable LatentMoE 解码权重冲刷 L2 激活与 TP1 尾部 GEMM 负载不均**（[`flashinfer@1c0d2be434d6d57ca16a428d439c433b8b1e6f81`](https://github.com/flashinfer-ai/flashinfer/commit/1c0d2be434d6d57ca16a428d439c433b8b1e6f81)）：`T <= 128` 时流式权重块会驱逐 L2 中跨专家复用的 token 激活；`T >= 1024` 时 TP1 尾部的持久化 tile-walk 内核跨 SM 负载不均。
6. **TileLang 跨地址空间部分屏障 ID 冲突死锁与 Top-K 共享内存越界**（[`tilelang@efa1ffc1e3dc`](https://github.com/tile-ai/tilelang/commit/efa1ffc1e3dc547b11fee5acf671daff1f0f94cb)、[`tilelang@4da574c01326`](https://github.com/tile-ai/tilelang/commit/4da574c01326eab5da5345af0aec1d19ce473c6a)）：`ThreadPartialSyncRewriter` 对 `"shared"` 与 `"shared.dyn"` 两次遍历间清空 `used_barrier_ids_`，导致不同 warpgroup 复用同一 `bar.sync 3, 128` 死锁；`tl_topk_impl` 的固定 `[2, 4096]` 共享内存队列在等值分数超过 4,096 时越界触发 `CUDA_ERROR_ILLEGAL_ADDRESS`。

---

## 2. 核心实现机制与源码走读

### 2.1 SM103 QSA NVFP4 两波持久化遍历与单块 `tcgen05` 融合归约 (`131f85942edb`)

在 `flashinfer/experimental/cake_qsa_nvfp4/cake_backend.py` 与 `cake_qsa_nvfp4_kernels.cuh` 中：

- **两波持久化遍历 (`decode_tcgen05_persistent`)**：当架构属于 `WALK_ARCHES = ("sm_103a",)` 且平铺切分满足
  $$1 < \left\lceil \frac{\text{rows} \cdot H_{kv} \cdot S_{\text{plain}}}{\text{SM}_{\text{count}}} \right\rceil \le \text{WALK\_MAX\_WAVES} = 2$$
  并且单 CTA 最大遍历块数小于两波串行块数时，启用持久化遍历。在 GB300（`152` SMs）的 `rows=128`（`256` 个 `(row, kv_head)` 对，每对 `33` 个块，共 `8,448` 个块槽位）下，启动 `152` 个持久 CTA（1 CTA/SM），每个 CTA 遍历连续的 `55..56` 个 `(pair, tile)` 槽位（小于两波 `2 × 33 = 66` 块），每对仅产生 `walk_splits = 2` 个部分槽位供归约读取。
- **单块融合归约 (`tcgen05` one-tile fused merge)**：在 `rows=1,2` 的融合模式中，`grid.sync()` 后不再分 4 次每次合并 8 列，而是由 32 条线程同时在 `33 × 32` 共享内存表上归约全部 32 组 8 列（`256` 维 head_dim），单次遍历完成 `partial_out` 合并。

### 2.2 SM120 `mma.sync` QSA 解码与 `SPLIT_OVERRIDES` (`09a77503f004b8f36a38d23074fc5da6877de811`)

- **无 TMEM 架构的双缓冲寄存器累加**：`sm_120a` 采用 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`，通过 `cp.async.cg` 双缓冲 64-token NVFP4 块，在共享内存中用 `cvt.rn.bf16x2.e2m1x2` 配合 UE8M0 块指数反量化为 BF16。
- **实测切分表 `SPLIT_OVERRIDES`**：将 `rows=4,8,16,32` 的切分数从公式值 `33,23,11,5` 覆盖为 `16,11,5,3`（`rows=1,2` 保持 `33`），消除过度切分带来的归约带宽放大。

### 2.3 Hidden-4096 4-Route Mega-mHC 算子与 `CheckedMul` 溢出保护 (`847172573ad1d5a929a8e7fb67aa7c072f8f7dcb`)

- 新增 5 个 `H=4096`（`K=16,384`）持久化算子：`bf16`、`tailpipe`、`tailpipe_wsmem`、`fp8_group128` 与 `nvfp4`，单核融合后混合、`BF16x3` split-K 映射、20 轮 Sinkhorn 迭代、折叠与 RMSNorm。
- 在 `csrc/cake_deepgemm_mega_mhc_jit_binding.cu` 中引入 `CheckedMul`（调用 `__builtin_mul_overflow`），当 TMA 步幅或元素总数超过 32/64 位上限时立即抛出异常，杜绝截断回绕。

### 2.4 SM120 拓扑加权流水化 PCIe CE Ring All-Reduce (`68e3e09ad3a52cbbd6b39d159d2fad7d1653073d`)

- `decide_ring_slow_hops` 通过 NVML 检测跨 CPU Socket 的 `NVML SYSTEM` 链路（8 卡双 Socket 下为 `3→4` 与 `7→0`，掩码 `0x88 = 136`）。
- 在 7 步 reduce-scatter 与 7 步 all-gather 中，不经过慢跳的 chunk 获得更大权重（更多元素），并在每卡数据量 `>= 4 MiB` 时切分为 `2..4` 个 pipeline piece，使 Copy Engine (CE) DMA 传输与 SM 归约在流水线上重叠。

### 2.5 Kimi-K3 Stable LatentMoE `L2::evict_first` 与认领项驻留 TP1 尾部 GEMM (`1c0d2be434d6d57ca16a428d439c433b8b1e6f81`)

- **`DECODE_A_EVICT_FIRST = True`**：在 `T <= 128` 时对流式读取的专家权重和 down-proj 权重附加 `L2::evict_first` 提示，保留 L2 中跨活跃专家复用的 token 激活。
- **认领项驻留 `256×512` 2-CTA TP1 尾部 GEMM**：在 `T >= 1024` 时，每个 `256×512` 2-CTA 集群通过原子队列认领输出块，并在寄存器内融合 RMSNorm 后直接写出。

### 2.6 TileLang 跨地址空间部分屏障追踪与全局 Top-K 工作区 (`efa1ffc1e3dc`, `4da574c01326`)

- `ThreadPartialSyncRewriter` 将 `used_barrier_ids_` 提升为跨 `"shared"` 与 `"shared.dyn"` 两次遍历的持久集合，确保 `"shared"` 分配 `bar.sync 3, 128` 后，`"shared.dyn"` 分配 `bar.sync 4, 128`。
- `tl_topk_impl` 将桶内候选队列从共享内存 `[2, 4096]` 移至全局显存 `(batch, 2, seq_len)`，彻底消除分数扎堆时的越界崩溃。

---

## 3. 架构师定量表格与实验验证

运行 `python experiments/flashinfer_sm103_qsa_persistent_mhc4096_sm120_piped_kimi_k3_tilelang_sim/run.py --check` 可验证以下定量指标：

| 子系统 / 算子 | 硬件平台 | 优化前基线 | 优化后实测 | 加速比 / 收益 | 证据性质 |
|---|---|---|---|---|---|
| QSA NVFP4 `rows=128` 两波持久化遍历 | GB300 (`sm_103a`, 152 SMs) | `71.89 us` (256 CTAs, 2 waves) | `67.74 us` (152 CTAs, 55-56 tiles) | **1.061x** (CG **1.068x**) | `author-benchmark` + `own-offline` |
| QSA NVFP4 `rows=1` 单块融合归约 (read flush) | GB300 (`sm_103a`) | `11.70 us` | `10.02 us` | **1.168x** | `author-benchmark` + `own-offline` |
| QSA NVFP4 `rows=4` `mma.sync` + `SPLIT_OVERRIDES` | RTX PRO 6000 (`sm_120a`, 188 SMs) | `28.59 us` (33 splits) | `16.13 us` (16 splits) | **1.772x** | `author-benchmark` + `own-offline` |
| QSA NVFP4 `rows=128` HND `mma.sync` (CUDA Graph) | RTX PRO 6000 (`sm_120a`) | `321.59 us` | `169.10 us` | **1.902x** | `author-benchmark` + `own-offline` |
| Mega-mHC `H=4096` 4-Route (`T=1024`) | B200 (`sm_100a`) / GB300 (`sm_103a`) | `39.095 us` / `35.510 us` | `27.424 us` / `25.608 us` | **1.426x** / **1.387x** | `author-benchmark` + `own-offline` |
| PCIe CE Ring Piped All-Reduce (`96 MiB/rank` BF16) | 8×SM120 双路 PCIe Gen5 | `4.4753 ms` (单分片 Memop) | `4.0287 ms` (3 pieces 拓扑加权) | **1.111x** | `author-benchmark` + `own-offline` |
| Kimi-K3 LatentMoE Front `L2::evict_first` (`T=64`) | GB300 (`sm_103a`) | `53.07 us` | `44.18 us` | **1.201x** | `author-benchmark` + `own-offline` |
| Kimi-K3 LatentMoE TP1 驻留尾部 GEMM (`T=1024`) | GB200 (`sm_100a`) | `120.87 us` | `107.81 us` | **1.121x** | `author-benchmark` + `own-offline` |

---

## 4. 反证、边界条件与后续可证伪问题

1. **为什么 B200 (`sm_100a`, 148 SMs) 不启用 `decode_tcgen05_persistent`？**  
   上游实测表明，在 B200 上两波 Grid 的持久化遍历与普通切分持平（`walk` 带来的动态槽位除法和 `walk_splits=2` 合并开销抵消了波尾收益），因此 `WALK_ARCHES` 严格限定为 `("sm_103a",)`。
2. **为什么 `rows=1,2` 在 SM120 上保留 `33` 个切分而不降为 `16`？**  
   在 `rows=1,2` 时，188 个 SM 完全空闲，若将切分从 33 降为 16，L2 read-flush 场景下会损失约 `7%` 的延迟，因此 `SPLIT_OVERRIDES` 仅对 `rows ∈ {4, 8, 16, 32}` 生效。
3. **后续可证伪问题**：当 Rubin (`sm_107a`) 的 SM 数量与 L2 分区拓扑确定后，`WALK_MAX_WAVES=2` 的持久化遍历阈值是否应扩展到三波 Grid？
