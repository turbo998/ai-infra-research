# FlashInfer Cake Blackwell (`SM100`/`SM103`) 融合 TP GEMM + MNNVL Lamport All-Reduce、`tcgen05.wait::ld` TMEM 排空与 2-CTA Producer-Tail 同步契约

> **English Version**: [flashinfer-cake-sm100-gemm-lamport-allreduce-tmem-drain.en.md](flashinfer-cake-sm100-gemm-lamport-allreduce-tmem-drain.en.md) | **关联实验**: [`experiments/flashinfer_cake_gemm_lamport_allreduce/README.md`](../../experiments/flashinfer_cake_gemm_lamport_allreduce/README.md)

### 一句话结论

FlashInfer 在 Blackwell (`SM100`/`SM103`) 上合入的 **Cake 融合张量并行 GEMM + MNNVL Lamport All-Reduce（`cake_gemm_allreduce`）** 将行并行线性层 $\text{out} = \sum_{r=0}^{W-1} X_r W_r^T$ 压缩为单次内核发射（`192` 线程 / 6 Warps，64 列 $\times$ `token_tile` `tcgen05` 累加器 Tile）：在 $M \le 16$ 采用 **One-Shot NVLS 组播发布**、在 $16 < M \le 128$ 采用 **Two-Shot Scatter / Owner-Reduce / Multicast Broadcast**，通过将打包 BF16 中的 `-0.0`（`0x8000`）清洗为 `+0.0`（`0x0000`）彻底消除与 32-bit Lamport 脏哨兵 `0x80000000u` 的数值碰撞，并以固定 Rank 顺序（$r = 0 \dots W-1$）执行 `add.rn.f32.bf16` 累加与单次 `__float22bfloat162_rn` 舍入实现**跨 Rank 逐比特一致（Bitwise-Identical）输出**，在解码波段 $M \le 64$ 较 `mm_bf16 + allreduce_fusion` 提速 **1.1x–1.3x**（$M = 128$ 持平，$M \ge 256$ 因权重按 Tile 重复流式读取由两步路径反超约 **1.3x**）；同时修复了两类 PTX `tcgen05` 内存一致性缺陷——在 39 个持久化 Kimi-K3 AttnRes 内核的 `tcgen05.dealloc` 前补齐 `tcgen05.wait::ld.sync.aligned`（B300 `sm_103a` 零性能损耗 `1.0000x`），以及在 9 个 2-CTA（`cta_group::2`）FP8$\times$FP4 混合 GEMM 内核的 Load Warp 退出前增设 `ring_depth`（`5..12` 级）Producer-Tail `mbarrier_wait` 排空，杜绝异步跨 CTA Multicast Arrive 落入已退出 CTA 共享内存的未定义行为。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `flashinfer/comm/cake_gemm_allreduce.py` & `flashinfer/jit/cake_gemm_allreduce.py` | `flashinfer-ai/flashinfer` commit `e563bf9867ba6cf5ef0f7becff219343c6ac519e` / blobs `d41ff11eac`, `b6bef816ac` | 定义 `cake_gemm_allreduce` 公开 API、12 条路由表 `ROUTES`（`world_size` $\in \{2,4,8\} \times$ `token_tile` $\in \{16,32,64,128\}$）、`select_route` 与三缓冲 Lamport 工作区字节计算 `required_workspace_bytes`。 |
| `csrc/cake_gemm_allreduce/cake_gemm_allreduce_binding.cu` & `cake_gemm_allreduce_kernels.cu` | `flashinfer-ai/flashinfer` commit `e563bf9867ba6cf5ef0f7becff219343c6ac519e` / blobs `380a309fef`, `9b2ef6d494` | 编码 3D K-Major `SWIZZLE_128B` TMA 描述符 `EncodeKMajorMap`，实现 12 个 `sm_100a`/`sm_103a` 设备内核（6 级 TMA 流水线、32 列 TMEM 累加、`-0.0` BF16 清洗、Lamport `0x80000000u` 轮询归约、后台脏区清理与 PDL 支持）。 |
| `csrc/experimental/cake_kimi_k3_attn_res/*_kernel.cu` | `flashinfer-ai/flashinfer` commit `3bf6d3202239a107989451257c6f594cb5d2f008` | 在 39 个 288 线程持久化 Kimi-K3 AttnRes 内核的清理屏障 `barrier.sync 9, 288` 与 `tcgen05.dealloc` 之前插入 `tcgen05.wait::ld.sync.aligned`，闭合 TMEM 读取的反依赖（Anti-Dependency）竞态。 |
| `csrc/experimental/deepgemm_mixed_gemm/*_kernel.cu` | `flashinfer-ai/flashinfer` commit `b815220c91936cd7e11a0b91fddb3307a4317a26` | 在 9 个 2-CTA（`cta_group::2`）FP8$\times$FP4 混合 GEMM 内核的 TMA Load Warp 主循环后追加 `ring_depth`（`5..12`）次 `mbarrier_wait(mma_done_addr + load_stage * 8)` Producer-Tail 排空。 |

### 核心实现路径、数据布局与同步契约

1. **12 路由表、3D K-Major TMA 描述符与解码波段边界（`cake_gemm_allreduce.py:24-85`, `cake_gemm_allreduce_binding.cu:40-165`）**：
   - 每个 Rank 持有行并行分片 $X_r \in \text{BF16}[M, K_r]$（$K_r \bmod 128 = 0$）与 $W_r \in \text{BF16}[N, K_r]$（$N \bmod 64 = 0$）。
   - `select_route(M)` 对 $M \le 16$ 选择 `(token_tile=16, twoshot=False)`（One-Shot），对 $16 < M \le 32$、$32 < M \le 64$、$M > 64$ 分别选择 `token_tile` 为 `32`、`64`、`128` 且 `twoshot=True`（Two-Shot）。
   - 主机发射器通过 `EncodeKMajorMap` 为 $W$ 和 $X$ 构造 3D K-Major `CUtensorMap`：`global_dim = {64, rows, K / 64}`，`global_strides = {K * 2, 128}` 字节，`box_dim = {64, box_rows, 2}`（其中 $W$ 的 `box_rows = 64`，$X$ 的 `box_rows = token_tile`），采用 `CU_TENSOR_MAP_SWIZZLE_128B` 与越界零填充。
   - 网格维度为 `gridDim = (N / 64, ceil(M / token_tile), 1)`，每 CTA 固定 `192` 线程（Warps `0..3` 为 Epilogue/All-Reduce 消费者，Warp `4` 为 `tcgen05` MMA 发射，Warp `5` 为 6 级 TMA 异步加载）。当 $M \le 128$ 时，`gridDim.y = 1`，每个列分片 CTA 仅从 HBM 流式读取一次权重分片 $W_r$；当 $M \ge 256$ 时，`gridDim.y = ceil(M / 128) >= 2`，权重分片被重复流式读取 $\lceil M / 128 \rceil$ 次，因此 $M \ge 256$ 应路由回 `mm_bf16 + allreduce_fusion`。
2. **Lamport 脏哨兵 `0x80000000u`、BF16 `-0.0` 清洗与跨 Rank 逐比特确定性归约（`cake_gemm_allreduce_kernels.cu:250-450`）**：
   - Epilogue Warps `0..3` 通过 `tcgen05.ld.sync.aligned.16x256b.x2.b32`（`n16`）或 `.x4.b32`（`n32`）从 32 列 TMEM 读出 FP32 累加结果，经 `__float2bfloat16` 写入带 Padding 步长 `66`（避免 Bank Conflict）的共享内存转置区 `xpose_h[tok * 66 + col]`，随后在 `barrier.sync 8, 128` 后按连续 8 个 BF16（`int4` / 16B）向量化写出。
   - **为什么必须清洗 BF16 `-0.0`（`0x8000`）**：MNNVL Lamport 工作区使用 32-bit 整数 `0x80000000u` 作为未写入（Dirty）哨兵。在打包的两个 BF16（`uint32_t`）中，`0x80000000u` 恰好对应低 16 位为 `+0.0`（`0x0000`）且高 16 位为 `-0.0`（`0x8000`）的合法 BF16 数值对！内核在发布前通过 `_packed16_sanitize` 将任意半字中的 `0x00008000u` 强制替换为 `0u`（`+0.0`），由于 IEEE-754 下 $-0.0 = +0.0$，此变换在保持算术完全等价的同时，确保任何已发布的 32-bit 字绝不等于 `0x80000000u`。
   - **逐比特一致（Bitwise-Identical）归约**：每个接收方或 Owner Rank 通过 `ld.volatile.global.v4.b32` 轮询本地 Unicast 缓冲区直到 4 个 32-bit 字均不为 `0x80000000u`，随后严格按照 $r = 0, 1, \dots, W-1$ 的固定顺序执行内联 PTX `add.rn.f32.bf16` 将 BF16 直接累加至 FP32 寄存器，最终仅执行一次 `__float22bfloat162_rn` 舍入。同时，所有 CTA 协同清理上一轮遗留的 Dirty 缓冲区，由 `(blockIdx.x == 0, blockIdx.y == 0, tid_1 == 0)` 等待全局原子计数器 `buffer_flags[8] == gridDim.x * gridDim.y` 后推进 `(buffer_idx + 1) % 3` 三缓冲状态机。
3. **Blackwell `tcgen05` 两大硬件同步契约修复（`#6740` & `#6819`）**：
   - **反依赖排空（`3bf6d32022`）**：PTX ISA `tcgen05` 内存一致性模型规定，对 `tcgen05.ld` 目标寄存器的真数据依赖仅排序寄存器值本身，**不保证** `tcgen05.ld` 与后续 `tcgen05.dealloc` 之间的内存反依赖（Anti-Dependency）顺序。在 288 线程持久化 Kimi-K3 AttnRes 内核中，若不加 `tcgen05.wait::ld.sync.aligned`，当所有线程通过 `barrier.sync 9, 288` 后，Consumer Warp 0 执行 `tcgen05.dealloc` 时其他 Consumer Warp 的 TMEM 读取可能尚未排空。在 `barrier.sync 9, 288` 前插入单条 `tcgen05.wait::ld.sync.aligned` 编译为单条 Scoreboard-Wait `NOP`，在 B300 上实测 83 个变体的速度比为 `0.9984..1.0024`（均值 `1.0000x`）。
   - **2-CTA 集群 Producer-Tail 排空（`b815220c91`）**：在 `cta_group::2` 模式下，Leader CTA 的 MMA Warp 使用 `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster`（掩码 `0b11`）同时向两个 CTA 的 `mma_done` 屏障发送阶段释放信号。由于 PTX 不对两次 `tcgen05.commit` 提供跨屏障顺序保证（§9.7.18.6），Peer CTA 的 Load Warp 若在完成最后一次 TMA 填充后直接退出循环，会导致最后 `ring_depth`（`5..12`）个阶段的 Multicast Arrive 信号发往已退出 CTA 的共享内存（违反 PTX ISA §2.2.2）。修复方案在 Load Warp 主循环后追加 `ring_depth` 次 `mbarrier_wait(mma_done_addr + load_stage * 8, _phase_mma_done)`，完整排空在途信号。

### 关键公式、单位与工程假设

1. **三缓冲 Lamport 工作区每缓冲区字节数（`required_workspace_bytes`）**：
   $$\text{Bytes}_{\text{buffer}}(M, N, W) = \begin{cases} 2 \cdot M \cdot N \cdot W & \text{if } M \le 16 \text{ (One-Shot)} \\ 4 \cdot \lceil M / W \rceil \cdot W \cdot N & \text{if } M > 16 \text{ (Two-Shot)} \end{cases}$$
   总 Lamport 缓冲区容量为 $3 \times \text{Bytes}_{\text{buffer}}(M, N, W)$。
2. **动态共享内存（Dynamic SMEM）分配公式**：
   $$\text{SMEM}_{\text{total}}(\text{tile}) = 1024 + 6 \times 16384 + 6 \times (256 \cdot \text{tile}) + \text{XPOSE}(\text{tile})$$
   对应 `tile` $\in \{16, 32, 64, 128\}$ 的精确字节数分别为 `126,080 B`、`152,704 B`、`206,080 B` 与 `214,528 B`（均在 Blackwell SM 228 KiB 共享内存上限内）。
3. **2-CTA 流水线退出前未等待的 Multicast Commit 数量**：
   $$N_{\text{unawaited\_without\_tail}} = \min(N_{\text{tiles}}, D_{\text{ring}}), \quad N_{\text{unawaited\_with\_tail}} = 0$$

### 合成实验与验证结果（`experiments/flashinfer_cake_gemm_lamport_allreduce`）

运行 `python experiments/flashinfer_cake_gemm_lamport_allreduce/run.py --check` 的确定性验证结果如下：

| 指标 / 场景 ($N=7168, K_r=2304, W=8$) | $M=16$ (One-Shot) | $M=32$ (Two-Shot) | $M=64$ (Two-Shot) | $M=128$ (Two-Shot) | $M=256$ (Two-Shot 越界) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `(token_tile, twoshot)` | `(16, False)` | `(32, True)` | `(64, True)` | `(128, True)` | `(128, True)` |
| 网格维度 `gridDim` / 总 CTA 数 | `[112, 1, 1]` (`112`) | `[112, 1, 1]` (`112`) | `[112, 1, 1]` (`112`) | `[112, 1, 1]` (`112`) | `[112, 2, 1]` (`224`) |
| 动态 SMEM 字节数 | `126,080 B` | `152,704 B` | `206,080 B` | `214,528 B` | `214,528 B` |
| 单缓冲区 / 三缓冲区 Lamport 字节 | `1,835,008 B` / `5.51 MB` | `3,670,016 B` / `11.01 MB` | `7,340,032 B` / `22.02 MB` | `14,680,064 B` / `44.04 MB` | `29,360,128 B` / `88.08 MB` |
| 权重流式读取倍数 / 读取量 | `1x` (`31.50 MiB`) | `1x` (`31.50 MiB`) | `1x` (`31.50 MiB`) | `1x` (`31.50 MiB`) | `2x` (`63.00 MiB`) |
| 融合 vs 两步估算加速比 / 推荐路径 | **`1.289x`** (`cake_gemm_allreduce`) | **`1.242x`** (`cake_gemm_allreduce`) | **`1.168x`** (`cake_gemm_allreduce`) | **`1.057x`** (`cake_gemm_allreduce`) | **`0.785x`** (`mm_bf16+allreduce_fusion`) |

- **哨兵防碰撞与跨 Rank 逐比特一致性验证**：当 Rank 0 产生部分和 `(0.0, -0.0)` 时，未清洗打包字 `0x80000000` 与脏哨兵碰撞（`unsanitized_dirty_sentinel_collision = True`），经 `_packed16_sanitize` 清洗后碰撞率为 `0`；在高动态范围部分和 `(65536.0, 0.00390625, -65536.0)` 下，Ring 顺序在 8 个 Rank 间产生 `2` 种不同输出，而 Cake 固定顺序 FP32 归约在全部 8 个 Rank 上输出完全一致的 `0x80800080`（`cake_bitwise_identical = True`）。

### 架构师的设计练习与可证伪问题

1. **计算练习**：当 $W=8, N=7168, M=20$ 时，为什么 Two-Shot 路由的 `required_workspace_bytes` 为 $2 \times \lceil 20 / 8 \rceil \times 8 \times 7168 \times 2 = 6,881,280\text{ B}$ 而非 $2 \times 20 \times 7168 \times 2$？
   - *解答*：Two-Shot 第一阶段按 `dest_rank = tok % W`、`dest_token = tok / W` 将 Token 分散写入各 Owner Rank，每个 Owner Rank 最多接收 $\lceil M / W \rceil = \lceil 20 / 8 \rceil = 3$ 个 Token，且需要同时容纳 Stage 0（$W$ 个 Rank 的输入槽位）与 Stage 1（广播槽位），因此总字节数为 $2 \times 3 \times 8 \times 7168 \times 2 = 6,881,280\text{ B}$。
2. **可证伪假设**：若在 `cta_group::2` 内核中仅依靠 Epilogue 等待最终 `acc_full` 的 `tcgen05.commit` 而不执行 Load Warp 的 Producer-Tail 排空，当 $N_{\text{tiles}} < D_{\text{ring}}$ 时是否会发生死锁？
   - *验证方式*：对照 `tail_state_machine_checks`，由于未填充的 $D_{\text{ring}} - N_{\text{tiles}}$ 个阶段保留了初始化时的空闲相位信用（Phase 0），Producer-Tail 对未填充阶段的等待会立即通过（不死锁），但若完全省略 Producer-Tail，已填充的 $N_{\text{tiles}}$ 个阶段的异步 Multicast Arrive 仍可能在 Peer CTA 退出后到达。
