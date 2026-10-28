# FlashInfer Cake Softmax 均衡分块与 GB300 持久化路由、Cake Kimi-K3 融合路由器、NVLink 单边 All-to-All Lamport 传输、SM12x `b12x` W4A16 E8M0 指数重偏置与 `-1` 路由过滤，及 TileLang/Triton 内核安全分析 (Phase 127 · Priority 136)

> **Language / 语言**: **中文 (当前)** | [English (`flashinfer-cake-softmax-kimi-k3-lamport-b12x-e8m0-tilelang-triton-phase127.en.md`)](flashinfer-cake-softmax-kimi-k3-lamport-b12x-e8m0-tilelang-triton-phase127.en.md)

## 1. 核心架构发现与工程背景

在 Phase 127 Track A 中，我们针对 `flashinfer-ai/flashinfer`、`tile-ai/tilelang` 与 `triton-lang/triton` 的 **11 个最新上游提交** 进行了源码级与确定性数值审计，覆盖 Blackwell (`sm_100a` B200 / `sm_103a` GB300)、Hopper (`sm_90a`) 与 GeForce/RTX PRO Blackwell (`sm_120`) 的关键性能与精度不变量：

1. **Cake Blackwell Softmax 均衡分块与 GB300 持久化路由 (`flashinfer@80a8b2a8`)**：
   - 在 `csrc/cake_blackwell_softmax.cu` 中，GB300 (`sm_103a`) 上 `6-tile 8-CTA` 分块在 `20/9 <= rows_per_persistent_cluster < 2.5`（即 GB300 上 80–89 行）时切换至持久化内核（52 个形状几何平均提速 **`1.088x`**，最高 **`1.132x`**，`87x98304` 从 `13.41 -> 11.84 us`）；同时 GB300 的 `4-tile 16-CTA` 行缓存与 `16-CTA` 共享 Tile/持久化程序对齐 B200 的均衡每 CTA 切片规则 `ceil(vocab / CTAs)`（向上取整至 32 元素），使 GB300 95 个测量形状几何平均提速 **`1.066x`**，B200 39 个形状几何平均提速 **`1.022x`**（最高 **`1.038x`**）。
2. **Cake Kimi-K3 融合路由器 (`896` 专家 Top-16) 调度优化 (`flashinfer@2c6a3ea6`)**：
   - `flashinfer.fused_moe.kimi_k3_fused_router` 在单次启动中完成 `896` 专家 Top-16 选择与 `moe_align_block_size` 路由计划构建。新增单 Token `block_m=16` 的单 CTA `LC1_NC` 程序、`29–32` Token 的 `LCG_LW`、`257–512` Token 的 `Q4S_E16`（每 Lane 发射 16 Token）以及宽批次 `Q4SPW_LR`（16 字节候选秩与 Ballot 平面扫描）。在 `1 token, block_m=16` 下，GB300 延迟从 `3.232 -> 2.656 us`（**`1.22x`**），B200 从 `3.552 -> 2.943 us`（**`1.21x`**），28 个测量单元整体几何平均提速 **`1.03x`**。
3. **NVLink 单边 All-to-All Lamport 传输协议 (`flashinfer@1f306300`)**：
   - 为 `NVLinkOneSidedAlltoAll` 新增面向 Decode 小步长的第 3 条传输路径（Lamport 传输）：每个推送的 **16 字节向量** 自带到达语义，落地元行预装发送方绝不写入的哨兵字（碰撞哨兵的极少数有效载荷字在保持数值等价的前提下改写，如 BF16 `0xFFFF` NaN 改写为 `0x7FFF7FFF`、NVFP4 `-0` 改写为 `+0`），接收方直接轮询数据本身，**彻底消除完成标志等待与系统级 Fence (`fence.acq_rel.sys`)**。
   - 默认阈值 `lamport_max_tokens` 保底为 `159` tokens（最高 `512` 且受每 Rank `64 MiB` 速度边界约束）。在 GB300 上，DeepSeek-V3 (`7168, top-8, 256E`) `EP=8, T=1` 获得 **`2.77x`（BF16: `43.6 -> 15.7 us`）** 与 **`2.69x`（NVFP4: `42.3 -> 15.7 us`）** 提速，`EP=64, T=1` 获得 **`2.54x`（BF16: `50.0 -> 19.7 us`）**；在 `DEP16` 端到端解码中每步降低 **`2.0–2.6 ms`** 延迟且输出逐位一致。
4. **SM12x `b12x` W4A16 BF16 E8M0 缩放指数重偏置 (`flashinfer@30cd27ea`) 与未路由 (`-1`) Top-K 求和守卫 (`flashinfer@f083cef0`)**：
   - **E8M0 偏置修复 (`30cd27ea`)**：旧版 `packed_dequant_e8m0x4_to_bfloat2x2` 将 E8M0 字节 `e` 物化为指数 `e + 7`（MMA 权重为 `true_weight * 2^-119`），仅在 `121 <= e <= 247` 时为正规 BF16；而真实 MXFP4 权重（如 MiMo-V2.6-Pro 第 1–16 层 `e in [111, 117]`）全部落入次正规区间（`e < 114` 时 FP4 `0.5` 直接冲零）。修复后将偏置改为 `71`（全局补偿 `2^55`，上限 Clamp 从 `247 -> 183`），使 **`57 <= e <= 183` 全部保持正规 BF16**，距次正规边界推远 **`2^64`**。
   - **未路由 `-1` 跳过 (`f083cef0`)**：在 RTX PRO 6000 Blackwell / RTX 5080 (SM120) 专家 Sidecar 场景下，非本卡负责的路由槽位传入 `-1`。`W4A16TopKSumKernel` 无映射分支原先累加了 FC2 未写入的 `-1` 行（导致全 `NaN` 或余弦相似度跌至 **`0.29`**），修复后跳过 `raw_expert < 0`，余弦相似度恢复至 **`0.99999`**（8/8 E2E 测试全部通过）。
5. **Frost MX FFMA 跨目标加载 (`64177fe5`)、Cake Ulysses 直接 Fence (`d304863b`)、Cake MoE AllReduce Union 首 Token 重叠 (`1e3f47f2`)、Hopper FMHAv2 计数器归零 (`32a8e5a7`) 及 TileLang (`fca340e5`) / Triton (`51a27535`) 守卫**：
   - Frost MX 低延迟 MoE 在 `top_k=6, E=4` 选择 FFMA 时复用 `_load_cross_compiled` 消除 SM90 默认编译目标加载 SM100 输入的首次调用崩溃；Cake Ulysses 直接路由在释放完成标志后立即发出 `fence.acq_rel.sys`（B300 提速 **`1.021x`**）；Cake MoE AllReduce Union 在 `1536–2048` tokens 下将首个 Token 的 Reduce-Scatter 提前至最后一次发布迭代内重叠存储（B300 提速 **`1.0332x`**）；Hopper FMHAv2 通过 `atomicInc(counter, N + G - 1)` 与 FP8 `atomicMax(counter, N)` 在内核尾票据直接归零动态调度计数器；TileLang 拒绝 `K % micro_size_k != 0` 的稀疏 MMA 残尾；Triton `lowerLocalStore` 为未覆盖的共享内存 CTA 副本补齐寄存器基址广播。

## 2. 确定性实验结果 (`experiments/flashinfer_cake_softmax_kimi_k3_lamport_b12x_e8m0_tilelang_triton_phase127`)

| 子模块 / 算子路径 | 基线表现 / 缺陷模式 | Phase 127 优化后表现 | 关键量化增益 |
|---|---|---|---|
| Cake Softmax (`GB300`, `87x98304`) | `8-CTA shared-tile` (`13.41 us`) | `8-CTA persistent` (`11.84 us`) | **`1.132x` 提速**（95 形状 geomean `1.066x`） |
| Cake Kimi-K3 路由器 (`1 token, bm=16`) | GB300 `3.232 us` / B200 `3.552 us` | `LC1_NC` GB300 `2.656 us` / B200 `2.943 us` | **`1.22x` (GB300) / `1.21x` (B200)** |
| NVLink 单边 All-to-All (`EP=8, T=1`) | Fence 路径 BF16 `43.6 us` / NVFP4 `42.3 us` | Lamport 16B 哨兵直轮询 `15.7 us` / `15.7 us` | **`2.77x` (BF16) / `2.69x` (NVFP4)** |
| `b12x` W4A16 BF16 E8M0 偏置 (`30cd27ea`) | Bias `7`（正规域 `121..247`，MiMo `111..117` 次正规） | Bias `71`（正规域 `57..183`，全局补偿 `2^55`） | **正规裕量扩大 `2^64`，0 次正规下溢** |
| `b12x` W4A16 `-1` 路由求和 (`f083cef0`) | 累加未初始化 `fc2_out`（`NaN` 或 `cos=0.29`） | 过滤 `raw_expert < 0` 且强制 `int32` 路由 ID | **`cos = 0.99999`（8/8 E2E 通过）** |
