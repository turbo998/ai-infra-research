# FlashInfer SM103 QSA Persistent-Walk 解码与 tcgen05 Merge、SM120 QSA mma.sync 188-SM Split Override、Hidden-4096 4-Route mHC Sinkhorn 融合、MiniMax-H3 半宽尾块、SM120 PCIe Copy-Engine Ring Piped All-Reduce、Kimi-K3 L2 Evict-First 与 2-CTA 常驻尾 GEMM，以及 TileLang 跨存储域 Partial-Sync 命名屏障去重

> **语言 / Language**: [中文](flashinfer-sm103-qsa-walk-mhc4096-sm120-ce-ring-kimi-k3-tilelang.md) | [English](flashinfer-sm103-qsa-walk-mhc4096-sm120-ce-ring-kimi-k3-tilelang.en.md)
> **阶段 / Phase**: Phase 73 (`priority: 82`, Track A)
> **关联问题 / Question ID**: `q-phase73-flashinfer-sm103-qsa-walk-mhc4096-sm120-ce-ring-kimi-k3-tilelang`
> **证据等级 / Evidence Tiers**: `implementation` + `author-benchmark` + `own-offline` (`mastery: not-assessed`)

---

## 1. 概述与核心架构洞察

在 2026-10-25 的上游提交中，`flashinfer-ai/flashinfer` 与 `tile-ai/tilelang` 针对 Blackwell 数据中心（SM100 B200/GB200、SM103 B300/GB300）与工作站/边缘服务器（SM120 RTX PRO 6000 Blackwell Server Edition，188 SMs）以及 Hopper 流水线稀疏注意力落盘了七项底层算子与编译器不变量修复：

1. **SM103 `cake_qsa_nvfp4` Persistent-Walk 解码与 `tcgen05` 单 Tile 融合 Merge**（`flashinfer@131f85942e`）：在 GB300（`152` SMs）上，当 `rows=128`、`num_kv_heads=2`、`plain_splits=1` 时，普通 split-KV 网格启动 `256` 个 CTA（每 CTA 走 `33` 个页面 Tile），恰好跨越 `ceil(256 / 152) = 2` 个调度波次（Wave），导致第二波仅有 `104` 个 CTA 运行而 `48` 个 SM 处于尾波空转。新引入的 `decode_tcgen05_persistent` 仅在 `sm_103a` 的两波区间（`1 < ceil(rows * num_kv_heads * plain_splits / sm_count) <= WALK_MAX_WAVES = 2`）启用，启动 `152` 个常驻 CTA 均匀遍历 `256 * 33 = 8448` 个 `(row, kv_head, tile)` 单元（每 CTA 走 `55–56` 个 Tile），将每个 `(row, kv_head)` 拆分为最多 `walk_splits = 2` 个部分槽位交由 `merge_tcgen05` 归约；同时将 `merge_tcgen05` 从 4 个 `16x256` sub-tile 合并为单个 `64x256` `tcgen05` 累加器 Tile，消除 3 次 `tcgen05.Wait` 停顿。在 GB300 上实现 `rows=1..4` **`1.07x–1.15x`**、以及 `rows=128` **`1.06x`（Eager NHD `71.89 -> 67.74 us`、HND `72.11 -> 67.79 us`）与 `1.07x`（CUDA Graph HND `72.37 -> 67.74 us`）** 端到端提速。
2. **SM120 `cake_qsa_nvfp4` `mma.sync` 解码与 188-SM Split Overrides**（`flashinfer@09a77503f0`）：针对无 `tcgen05`/TMEM 的 `sm_120a`（RTX PRO 6000 Blackwell Server Edition，`188` SMs，每 SM 最多同时驻留 2 个 `69,632 B` SMEM / `192` 线程 CTA，总驻留容量 `376` CTAs），新增 `mma.sync.aligned.m16n8k16` plain/cooperative decode 与 merge 核心，并在 `_SPLIT_OVERRIDES[188]` 中将 `rows=4, 8, 16, 32` 的默认拆分 `[33, 23, 11, 5]`（会产生 `264–368` CTAs 逼近单波驻留上限）降为 `[16, 11, 5, 3]`（`128–192` CTAs，每 SM 仅驻留 1 个 CTA，消除同 SM 资源争抢并缩短 Merge 列表一半）。在 RTX PRO 6000 上相对 FlashInfer 参考实现取得 **`1.27x–1.89x` Eager 加速**（`rows=128` HND `320.93 -> 169.62 us` **`1.89x`**，NHD `301.26 -> 161.36 us` **`1.87x`**）与 **`1.21x–1.90x` CUDA Graph 加速**。
3. **Hidden-4096 4-Route mHC 推理融合与 TMA Stride `CheckedMul` 64 位溢出防护**（`flashinfer@847172573a`）：在 `cake_deepgemm_mega_mhc` 中新增 `hidden=4096, routes=4`（20 轮 Sinkhorn 双随机归一化）的 5 个 SM100/SM103 专用内核（`bf16`, `tailpipe`, `tailpipe_wsmem`, `fp8_group128`, `nvfp4`），通过 `VARIANT_WSMEM_REGIME_SPLITS = 8` 与 `VARIANT_PLAIN_MIN_WAVES = 7` 路由解码/中批与大批次路径，并在生成主机端 TMA TensorMap 绑定代码时引入 `CheckedMul(int64_t, int64_t)` 防止大批次 `tokens * row_stride` 在 32 位有符号乘法下溢出。相对 vLLM `MHCFusedPostPreOp.forward_cuda` 在 B200 上取得 **`1.038x–1.426x`**（`T=1024`: `39.095 -> 27.424 us`）、在 GB300 上取得 **`1.041x–1.387x`**（`T=1024`: `35.510 -> 25.608 us`）加速。
4. **MiniMax-H3 `out_proj_nvfp4` N-Split 半宽尾块 (`256x128`)**（`flashinfer@b4125442a5`）：当 $M \ge 32$ 且 `256x256` 全宽 Tile 的尾波 CTA 余数 $\le \text{sm\_count}/2$（SM100 在 10 波内、SM103 在 48 波内）时，切换到 `256x128` 半宽输出列 Tile 将尾波填满整张 GPU，同时将每 CTA SMEM 从 `187 KiB` 降至 `122 KiB`。在 GB300 上达到 **`1.003x–1.041x`**（`P=8, M=4816`: `82.2 -> 79.0 us`），在 GB200 上达到 **`1.017x–1.052x`**（`P=8, M=4816`: `98.2 -> 93.4 us`）。
5. **SM120 `IpcVariant.COPY_ENGINE_RING_PIPED` 跨 Socket 慢跳感知流水环**（`flashinfer@68e3e09ad3`）：在 `flashinfer.comm` 中为 SM120 4/8 卡 PCIe 拓扑新增第 8 种 All-Reduce 变体 `COPY_ENGINE_RING_PIPED`。当单分片 `chunk >= 4 MiB`（4 卡总输入 $\ge 16\text{ MiB}$，8 卡总输入 $\ge 32\text{ MiB}$）时，把单分片拆为 2–3 个 Piece 使 DMA Copy Engine 的 HBM P2P 推送与 SM Reduce-Add 内核在双流（`stream` + `ce_stream`）上重叠，并通过 `decide_ring_slow_hops` 构建 `ce_slow_hops` 位掩码识别跨 CPU Socket 边界的慢跳以做非均匀分片配重。在 8×SM120 PCIe Gen5 上，`48 MiB/rank` 提速 **`1.050x`**（`2.1832 -> 2.0792 ms`），`96 MiB/rank` 提速 **`1.111x`**（`4.4753 -> 4.0287 ms`）。
6. **Kimi-K3 Latent MoE `L2::evict_first` 解码权重流式读取与 2-CTA 常驻 TP1 尾 GEMM**（`flashinfer@1c0d2be434`）：在 $T \le 128$ 解码区间将只读一次的权重矩阵标记为 `L2::evict_first` 以把激活保留在 L2 中，使 Front 内核提速 **`1.079x–1.201x`**、TP1 Tail 提速 **`1.156x–1.191x`**；在 $T \ge 1024$ 预填充区间将 TP1 Tail 改为基于共享内存队列（`land` + `free_word`）的常驻 2-CTA 集群（10 阶 TMA 流水线）并内联 RMSNorm，在大 Token 数（152 SMs 上 $T \ge 4096$、148 SMs 上 $T \ge 8192$）切换至 `256x512` 宽 Tile（使 `B_2` 跨 `K=2048` 的 TMA 加载减半），取得 **`1.028x–1.121x`** 加速（GB200 $T=4096$: `244.7 -> 218.2 us` **`1.121x`**）。
7. **TileLang `ThreadSync` 跨存储域 Partial-Sync 命名屏障去重**（`tilelang@efa1ffc1e3`）：`ThreadPartialSyncRewriter::Rewrite` 会依次对 `"shared"` 和 `"shared.dyn"` 两个存储域独立运行。此前每次运行都从 `ReservedNamedBarriers::kFirstUsedBarrier` 重新编号，导致 `deepseek_v32/sparse_mla_fwd_pipelined.py` 中 `"shared"` 域的 Producer warpgroup `[256, 384)`（128 线程）与 `"shared.dyn"` 域的 Consumer warpgroup `[0, 128)`（128 线程）被分配到同一个硬件屏障指令 `bar.sync 3, 128`，使两个不相关的线程组互相提前唤醒对方并引发 H100 数值损坏。修复后通过 `PostOrderVisit` 预先收集 IR 中已存在的 `tvm_storage_sync` 屏障 ID 填入 `used_barrier_ids_`，确保跨存储域屏障 ID 全局唯一（Consumer 分配到 `barrier_id = 4`）。

---

## 2. 数学公式、数据布局与硬件同步边界

### 2.1 SM103 QSA Persistent-Walk 网格与波次边界

设请求行数 $R = \text{rows}$，KV 头数 $H_{kv} = 2$，每头选定页面数 $P_{sel} = 1 + N_{\text{topk}} = 33$，普通拆分因子 $S_{\text{plain}}$。普通网格 CTA 数为：

$$G_{\text{plain}} = R \cdot H_{kv} \cdot S_{\text{plain}}, \quad W_{\text{plain}} = \left\lceil \frac{G_{\text{plain}}}{\text{SM}_{\text{count}}} \right\rceil$$

当满足以下条件时，SM103（$\text{SM}_{\text{count}} = 152$）激活 `decode_tcgen05_persistent`：

$$\text{arch} = \text{sm\_103a} \;\land\; S_{\text{plain}} < P_{sel} \;\land\; 1 < W_{\text{plain}} \le \text{WALK\_MAX\_WAVES} \;(=2)$$

此时常驻网格大小固定为 $G_{\text{walk}} = \min(\text{SM}_{\text{count}}, R \cdot H_{kv} \cdot P_{sel}) = 152$，总 Tile 任务数 $N_{\text{work}} = R \cdot H_{kv} \cdot P_{sel} = 8448$。每个 CTA 分配的连续区间长度为：

$$L_b \in \left\{ \left\lfloor \frac{N_{\text{work}}}{G_{\text{walk}}} \right\rfloor, \left\lceil \frac{N_{\text{work}}}{G_{\text{walk}}} \right\rceil \right\} = \{55, 56\}$$

由于单个 `(row, kv_head)` 包含 $P_{sel} = 33$ 个连续 Tile，长度为 $55\text{–}56$ 的连续区间在任意 `(row, kv_head)` 边界上最多被 2 个 CTA 切分，因此每个 `(row, kv_head)` 的部分槽位上限恒为：

$$S_{\text{walk}} = 1 + \left\lfloor \frac{\lfloor N_{\text{work}} / G_{\text{walk}} \rfloor}{P_{sel}} \right\rfloor = 2$$

### 2.2 TileLang 命名屏障碰撞与修复前后对比

```
[修复前 Buggy 路径: ThreadPartialSyncRewriter 每次调用重置 barrier_id_map_]
Pass 1 (storage_scope = "shared"):
  tx in [0, 256)   -> bar.sync 1, 256
  tx in [128, 384) -> bar.sync 2, 256
  tx in [256, 384) -> bar.sync 3, 128   <-- Producer Warpgroup (128 threads)

Pass 2 (storage_scope = "shared.dyn"):
  barrier_id_map_ 为空，重新编号 ->
  tx in [0, 128)   -> bar.sync 3, 128   <-- Consumer Warpgroup (128 threads) 发生硬件屏障 ID 碰撞！
  后果：64 个 Producer 线程 + 64 个 Consumer 线程即可凑齐 128 到达数，提前释放对方未写完的 SMEM 阶段！

[修复后 Fixed 路径: PostOrderVisit 预扫描已有 tvm_storage_sync(scope, id, count)]
Pass 2 (storage_scope = "shared.dyn"):
  used_barrier_ids_ = {1, 2, 3}
  while (used_barrier_ids_.count(barrier_id)) ++barrier_id; ->
  tx in [0, 128)   -> bar.sync 4, 128   <-- 硬件屏障完全隔离，H100 sparse_mla_fwd_pipelined 精度恢复
```

---

## 3. 量化性能矩阵与可复现验证 (`own-offline` vs `author-benchmark`)

运行本仓库原创标准库仿真脚本验证全部 7 项不变量：

```powershell
python experiments\flashinfer_sm103_qsa_walk_mhc4096_sm120_ce_ring_kimi_k3_tilelang_sim\run.py --check
```

| 子系统 / 内核 | 硬件平台 | 关键参数 / 负载 | 基准延迟 / 状态 | 优化后延迟 / 状态 | 加速比 / 结果 | 证据分类 |
|---|---|---|---|---|---|---|
| `cake_qsa_nvfp4` Persistent-Walk + `tcgen05` Merge | GB300 (`sm_103a`, 152 SMs) | `rows=1..4` / `rows=128` (HND) | `11.71–15.92 us` / `72.11 us` (`72.37 us` Graph) | `10.53–13.95 us` / `67.79 us` (`67.74 us` Graph) | **`1.07x–1.15x`** / **`1.06x–1.07x`** | `author-benchmark` + `own-offline` |
| `cake_qsa_nvfp4` `mma.sync` + 188-SM Split Override | RTX PRO 6000 (`sm_120a`, 188 SMs) | `rows=4` NHD / `rows=128` HND | `28.59 us` / `320.93 us` (`321.59 us` Graph) | `16.13 us` / `169.62 us` (`169.10 us` Graph) | **`1.77x`** / **`1.89x` (`1.90x` Graph)** | `author-benchmark` + `own-offline` |
| `cake_deepgemm_mega_mhc` Hidden-4096 (4-Route, 20-iter) | B200 (`sm_100a`) / GB300 (`sm_103a`) | `T=1024` / `T=4096` vs vLLM | B200 `39.095 / 118.465 us`; GB300 `35.510 / 109.945 us` | B200 `27.424 / 86.913 us`; GB300 `25.608 / 79.744 us` | B200 **`1.426x / 1.363x`**; GB300 **`1.387x / 1.379x`** | `author-benchmark` + `own-offline` |
| `cake_minimax_h3` Out-Proj Half-Width Tail (`256x128`) | GB300 (`sm_103a`) / GB200 (`sm_100a`) | `P=8, M=4816` (`608` CTAs -> partial wave `16`) | GB300 `82.2 us` / GB200 `98.2 us` | GB300 `79.0 us` / GB200 `93.4 us` | GB300 **`1.041x`** / GB200 **`1.052x`** | `author-benchmark` + `own-offline` |
| `IpcVariant.COPY_ENGINE_RING_PIPED` (Slow-Hop Aware) | 8× SM120 PCIe Gen5 | `48 MiB/rank` (2 pieces) / `96 MiB/rank` (3 pieces) | `2.1832 ms` / `4.4753 ms` | `2.0792 ms` / `4.0287 ms` | **`1.050x`** / **`1.111x`** | `author-benchmark` + `own-offline` |
| `cake_kimi_k3_latent_moe` `evict_first` + 2-CTA TP1 Tail | B200 / B300 / GB200 / GB300 | Decode $T \le 128$ / Prefill $T=4096$ (GB200) | Decode baseline / `244.7 us` | `evict_first` / `218.2 us` (`256x512` wide tile on $\ge 4096/8192$) | Decode **`1.079x–1.201x`** / Prefill **`1.121x`** | `author-benchmark` + `own-offline` |
| TileLang `ThreadPartialSyncRewriter` 去重 | Hopper H100 (`sm_90a`) | `sparse_mla_fwd_pipelined` (`shared` + `shared.dyn`) | Producer `[256,384)` 与 Consumer `[0,128)` 撞车 `bar.sync 3, 128` | `used_barrier_ids_` 去重分配 `bar.sync 4, 128` | **消除跨 Warpgroup 相位错乱与数值损坏** | `implementation` + `own-offline` |

---

## 4. 面向云解决方案架构师的设计练习与可证伪问题

1. **计算练习（SM103 vs SM100 Persistent-Walk 边界推演）**：为什么 `flashinfer@131f85942e` 仅在 `sm_103a`（GB300, 152 SMs）上启用 `decode_tcgen05_persistent`，而在 `sm_100a`（GB200, 148 SMs）上保持关闭？
   - *解答提示*：在 `sm_100a` 上，persistent-walk 内核中动态解算 `(row, kv_head, tile)` 坐标的整数除法与跨 `(row, kv_head)` 边界切换开销在 `rows=64`（2 波）时实测慢 `9–10%`；而在 `sm_103a` 上由于 TMEM/SMEM 吞吐与 152 SM 尾波比例不同，仅在 `rows=128`（256 CTAs vs 152 SMs，第二波空置 48 个 SM）处产生净收益（`+6%`）。
2. **可证伪假设（SM120 Copy-Engine Ring Piped 分片下限）**：若在 8×SM120 PCIe Gen5 拓扑上将 `PIPED_MIN_CHUNK_BYTES` 从 `4 MiB` 强行下调至 `512 KiB`（总张量 `4 MiB`），流水线是否仍能提速？
   - *证伪判据*：不能。将 `512 KiB` 分片拆成 2–3 个 Piece 后，每个 Piece 仅 `170–256 KiB`，双流 CUDA Event 同步与多次 `cudaMemcpyPeerAsync` 启动开销会超过 DMA 与 SM 归约的重叠收益，导致延迟劣于 `COPY_ENGINE_RING` 或 `PULL_PUSH_2STAGE`。
