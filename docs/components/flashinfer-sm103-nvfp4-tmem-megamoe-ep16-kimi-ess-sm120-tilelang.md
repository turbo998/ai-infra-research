# FlashInfer SM103 NVFP4 TMEM Attention、EP16 MXFP8 MegaMoE 松弛屏障/Warp 派发、Kimi-K3 TP12 ESS 路由、SM120 FP8 Varlen Attention 与 TileLang 编译器演进

> **关联研究问题 (`question_id`)**：`flashinfer-sm103-nvfp4-tmem-ep16-megamoe-sm120-fp8-varlen-kimi-tp12-tilelang`（Phase 76，`priority: 85`）  
> **先修依赖 (`depends_on`)**：`flashinfer-cake-norm-combine-w4a16-moe-minimax-nsplit-frost-fp8-tilelang`  
> **关联证据 ID**：`ev-flashinfer-sm103-nvfp4-ep16-megamoe-kimi-sm120-tilelang-2026`  
> **本地可复现实验**：`experiments/flashinfer_sm103_nvfp4_tmem_megamoe_ep16_kimi_ess_sm120_tilelang/run.py`  
> **英文版文档**：[`flashinfer-sm103-nvfp4-tmem-megamoe-ep16-kimi-ess-sm120-tilelang.en.md`](flashinfer-sm103-nvfp4-tmem-megamoe-ep16-kimi-ess-sm120-tilelang.en.md)

---

## 1. 核心架构问题与背景

在 NVIDIA Blackwell Ultra (`sm_103a` / GB300)、Blackwell (`sm_100a` / GB200)、GeForce/Workstation Blackwell (`sm_120a` / RTX 5090 & RTX PRO 6000) 以及昇腾 (Ascend) / AMD ROCm 多后端算子栈中，2026 年 10 月下旬的上游提交集中解决了以下五个底层瓶颈：

1. **GB300 (`sm_103a`) NVFP4 Attention 的共享内存与 Tile 边界停顿**：在 `cake_nvfp4_attention`（BF16 `[B, H, S, 128]` 量化为 NVFP4 E2M1 + E4M3 块缩放因子，非因果，`S % 512 == 0`）中，若注意力概率矩阵 $P$ 经由共享内存 (SMEM) 中转并在 Tile 边界串行等待，会限制 `tcgen05` 流水线吞吐。
2. **16-Rank NVL72 `CakeMxfp8MegaMoeEp16` 的跨卡屏障与 Token 派发开销**：在 16×GB300 NVL72 域上，每次跨 Rank 屏障若对 16 个目标 Rank 顺序执行 16 次系统作用域 `release` 原子加（两次屏障共 32 次），会串行化 NVLink 写刷出；同时全 CTA 协同派发为每个专家路由重复从全局内存加载 `6 KiB` BF16 激活行。
3. **Kimi-K3 TP12 LatentMoE Tail 在中等批次 (`256 <= M <= 960`) 的早切换退化**：`cake_kimi_k3_tp12_tail` 原本在 `M >= 256` 即从 Early-Shared-Scatter (`K1_twoshot_ess:grouped` + `K3-ESS`) 切换到 `cp.async.bulk` 持久化流水线 (`k3_persist_bulk`)，导致 `M=256..512` 区间承受不必要的持久化调度开销。
4. **SM120 (`sm_120a`) MiniMax-H3 FP8 Varlen Attention 与 NVFP4 Out-Projection 的主机同步空泡**：`minimax_h3_sm120_varlen_attention_fp8` 存在独立统计量 kernel 与 K/V 混合重填瓶颈；而 `minimax_h3_nvfp4_out_proj` 每次调用触发 2 次 Host-Device 同步 (`stream.synchronize()`) 以读取标量缩放因子计算 `alpha`，在连续 kernel 之间制造了 `186.9–223.3 us` 的 GPU 空闲空泡。
5. **TileLang 昇腾 L0 部分切片步长与 ROCm FP4 常量生成**：Ascend `copy_l1_to_l0` 处理 L1 N 维部分切片时将 L0 步长误算为切片宽度而非完整 L0 宽度；ROCm HIP 后端在生成 `float4_e2m1fn` 标量常量时缺少 `__tl_float_to_fp4(...)` 转换与头文件启用标志。

---

## 2. 第一手源码与提交审计

| 仓库与提交 | 合并日期 | 证据层级 | 核心源码路径与改动重点 |
|---|---|---|---|
| `flashinfer-ai/flashinfer@d08a1f732193` (PR `#7328`) | `2026-10-25` | `implementation` + `author-benchmark` | `flashinfer/experimental/nvfp4_attention/csrc/cake_nvfp4_attention_aeef857fb82e9e29a633_kernel.cu`：将 `TMEM_NCOLS` 从 `432` 扩展至 `480`，新增 `TMEM_TMEM_P0_LO/HI_OFFSET` (`448`/`456`) 与 `P1_LO/HI_OFFSET` (`464`/`472`)，将 $P$ 驻留在 TMEM 并重叠 Tile 边界；GB300 (`sm_103a`) 提速 **`1.02x–1.05x`**。 |
| `flashinfer-ai/flashinfer@a4dc7d17fb6a` (PR `#7299`) | `2026-10-25` | `implementation` + `author-benchmark` | `csrc/cake_mxfp8_megamoe_ep16_f8a1e691688d06811749_kernel.cu`：用 **1 次 `fence.acq_rel.sys` + 16 次 `atomicAdd.relaxed.sys`** 替代 16 次顺序 `release` 原子操作；新增 `cake_megamoe_dispatch_rows` 按 Warp 认领 Token 并单次复制 `6 KiB` BF16 行；16×GB300 提速 **`1.13x–1.18x`** (`377.5 -> 320.5 us`)。 |
| `flashinfer-ai/flashinfer@06da6cc640a0` (PR `#7310`) & `506fac6203ae` (PR `#7317`) | `2026-10-25` | `implementation` + `author-benchmark` | `flashinfer/experimental/kimi_k3_tp12_tail/cake_backend.py`：将 `K3_PERSIST_MIN_TOKENS` 从 `256` 提升至 `961`（`9 <= M <= 960` 走 `k3_ess:grouped`，`M > 8` 启用 cuBLAS），GB300/GB200 在 `M=256, 512` 提速 **`1.060x–1.080x`**；`gemm_base.py` 用 `@functools.cache` 缓存 cuDNN 版本检查（耗时 `-91.97%`）。 |
| `flashinfer-ai/flashinfer@ba12a8322385` (PR `#7289`) & `4796d2effd9e` (PR `#7309`) | `2026-10-25` | `implementation` + `author-benchmark` | `minimax_h3_sm120_varlen_attention_fp8` 融合统计量、单位缩放 MMA 与分离 K/V 重填（RTX 5090 提速 **`1.244x–1.404x`**，RTX PRO 6000 提速 **`1.005x–1.104x`**）；`minimax_h3_sm120_quant_out_proj` 在设备端以 FP64 计算 `alpha` 消除 2 次主机同步，GPU 空泡从 **`186.9–223.3 us` 降至 `3.1–3.6 us`（`~60x` 缩减）**。 |
| `tile-ai/tilelang@0bd33f953cf5` (`#3511`) & `9253431f2a17` (`#3548`) | `2026-10-25` | `implementation` | `src/transform/ascend_lower_opaque.cc` + `ascend_sync_insert.cc`：为 L1->L0 部分拷贝显式传入 `l0_pitch` 并精确记录 L1 写入范围；`codegen_hip.cc`：为 `float4_e2m1fn` 发射 `__tl_float_to_fp4(...)` 并置位 `enable_fp4_`。 |

---

## 3. 关键实现机制、公式与架构推导

### 3.1 SM103 NVFP4 TMEM 驻留概率矩阵 $P$
在 `cake_nvfp4_attention` (`sm_103a`) 中，输入 BF16 $[B, H, S, 128]$ 被量化为 NVFP4 (`E2M1` + `E4M3` 块缩放因子)，配备 `NUM_K_PIPE_STAGES = 4` 与 `NUM_V_PIPE_STAGES = 5`。更新后的 kernel 将 Tensor Memory (`TMEM`) 列数从 `432` 扩展到 `480`：
$$\text{TMEM\_NCOLS}: 432 \longrightarrow 480, \quad \text{Offsets}(P_{0,\text{lo}}, P_{0,\text{hi}}, P_{1,\text{lo}}, P_{1,\text{hi}}) = (448, 456, 464, 472)$$
这使得在线 Softmax 量化后的注意力权重 $P$ 直接保留在 TMEM 内供 PV `tcgen05.mma` 消费，并在 Tile 边界提前发起下一轮 K/V 预取，在 GB300 上实现：
- $(4, 8, 4096)$：`0.1145 ms -> 0.1094 ms` (**`1.05x`**)；
- $(1, 8, 32768)$：`1.4432 ms -> 1.4201 ms` (**`1.02x`**)；
- $(8, 32, 8192)$：`2.8392 ms -> 2.7562 ms` (**`1.03x`**)。

### 3.2 `CakeMxfp8MegaMoeEp16` 的单 Fence + 松弛系统原子屏障与单次 6 KiB 行拷贝
对于 16-Rank NVL72 (`H=3072, I=5120, E=512, top_k=8`)：
1. **跨卡 NVLink 屏障优化**：原先每个屏障发出 $16$ 次 `atomicAdd.release.sys`，每次 release 原子操作都会强制排空此前全部跨卡写请求；PR `#7299` 改为：
   $$\text{1}\times \texttt{fence.acq\_rel.sys} \;+\; 16\times \texttt{atomicAdd.relaxed.sys}$$
   两次屏障共将 32 次串行 release 原子操作收敛为 2 次系统级屏障加 32 次无序松弛原子加。
2. **Per-Warp Dispatch (`cake_megamoe_dispatch_rows`)**：在 C72 网格中，每个源 Rank 使用 $9\text{ CTAs} \times 12\text{ warps/CTA} = 108\text{ warps}$ 通过每调用原子计数器动态认领输入 Token 行。Warp 内 `lane 0..7` 分别对应该 Token 的 `top_k=8` 个路由目标，使用松弛系统原子操作认领目标 Rank 上的专家本地行槽位并通过 warp shuffle 广播；随后整个 32-lane warp **仅从全局内存读取一次 `6 KiB` BF16 激活行**，分 2 轮、每 lane 每轮拷贝 6 个 16-byte (`uint4`) 字：
   $$\text{Bytes/Token} = 32\text{ lanes} \times 6\text{ words} \times 16\text{ B} \times 2\text{ passes} = 6144\text{ B} = 6\text{ KiB}$$
3. **16×GB300 NVL72 实测结果（Bitwise-Identical）**：
   - `16 tokens/rank (256 total)`：balanced `355.4 -> 312.6 us` (**`1.14x`**)，hotset50 `355.3 -> 313.8 us` (**`1.13x`**)；
   - `32 tokens/rank (512 total)`：balanced `362.7 -> 320.5 us` (**`1.13x`**)，hotset50 `368.4 -> 318.5 us` (**`1.16x`**)；
   - `64 tokens/rank (1024 total)`：balanced `376.1 -> 322.0 us` (**`1.17x`**)，hotset50 `377.5 -> 320.5 us` (**`1.18x`**)。

### 3.3 Kimi-K3 TP12 Tail 的 Early-Shared-Scatter (`K3-ESS`) 阈值扩展与 SM120 设备端 `alpha`
- **Kimi-K3 TP12 Tail (`#7310`)**：将 `K3_PERSIST_MIN_TOKENS` 从 `256` 提高到 `961`，路由分段为：
  - $M \le 4$: `("k1_oneshot_ess", "k23:c4")`（融合 GEMM+Tail）；
  - $5 \le M \le 8$: `("k1_oneshot_ess", "k23:c8")`（单次 BF16 舍入，避免二次舍入误差）；
  - $9 \le M \le 16$: `("k1_oneshot_ess", "k3_ess:grouped")` + cuBLAS；
  - $17 \le M \le 960$: `("k1_twoshot_ess:grouped", "k3_ess:grouped")` + cuBLAS；
  - $M \ge 961$: `("k1_twoshot:pinned", "k3_persist_bulk:pinned")` + cuBLAS。
  在 `M=256` 与 `M=512` 下，GB300 分别从 `41.74 -> 39.39 us` (**`1.060x`**) 和 `60.59 -> 56.10 us` (**`1.080x`**)，GB200 分别从 `43.17 -> 39.96 us` (**`1.080x`**) 和 `61.53 -> 57.22 us` (**`1.075x`**)。
- **SM120 Device-Side `alpha` (`#7309`)**：`minimax_h3_nvfp4_out_proj` 直接将设备指针 `act_scale_d` 与 `weight_scale_d` 传入 GEMM kernel，在设备端按 FP64 计算：
  $$\alpha = \text{ float32}\!\left(\frac{1.0}{\text{double}(\text{act\_global\_scale}) \times \text{double}(\text{o\_weight\_global\_scale})}\right)$$
  彻底消除了每次调用的 2 次 `cudaStreamSynchronize`，使连续调用间的 GPU 空闲间隙从 **`186.9–223.3 us` 降至 `3.1–3.6 us`（降低约 `60x`）**，并在 CUDA Graph 捕获下无需预计算固定 `alpha`。

---

## 4. 架构师设计练习与可证伪问题

1. **计算练习（EP16 派发带宽节省）**：当 `tokens_per_rank = 64`、`top_k = 8`、`H = 3072` (BF16) 时，若每个专家路由独立从源 Rank 全局内存读取激活行，每 Rank 需读取 $64 \times 8 \times 6\text{ KiB} = 3072\text{ KiB}$；改用 `cake_megamoe_dispatch_rows` 单次读取后，源端全局内存读取降至 $64 \times 6\text{ KiB} = 384\text{ KiB}$（减少 `87.5%`）。
2. **后续可证伪问题**：在 `M > 960` 时，为什么 `k3_ess:grouped`（每 Token 每半列 1 个 CTA，即 $2M$ 个 CTA）开始劣于 `k3_persist_bulk:pinned`？当 $M > 960$ 时，$2M > 1920$ 个 CTA 在 GB200/GB300 (`148–160` SM) 上跨越超过 12 个调度波次，此时 `cp.async.bulk` 持久化流水线可摊销 CTA 启动与多播同步开销。
