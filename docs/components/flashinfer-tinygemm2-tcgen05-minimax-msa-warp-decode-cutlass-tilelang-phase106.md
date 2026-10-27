# FlashInfer `cake_tinygemm2_tcgen05` Swap-AB 瓦片、SM120 代理栅栏、SM90 MSA 与 `cake_warp_decode` Finalize 扩散及 CUTLASS/TileLang 分析 (Phase 106 · Priority 115)

> **English Version**: [flashinfer-tinygemm2-tcgen05-minimax-msa-warp-decode-cutlass-tilelang-phase106.en.md](flashinfer-tinygemm2-tcgen05-minimax-msa-warp-decode-cutlass-tilelang-phase106.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/flashinfer_tinygemm2_tcgen05_minimax_msa_warp_decode_phase106/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`flashinfer-ai/flashinfer@503b948ac73d7321622c750b5570ddd9ed6e3a5e` (`csrc/cake_tinygemm2_tcgen05/*`)**：
   - **Swap-AB 累加器瓦片与 32 字节对齐写回**：`cake_tinygemm2_tcgen05` 在每链累加瓦片（64 batch 行 / CTA）中交换 MMA 操作数（激活作为 `M=64` 的 `A` 操作数，权重作为 `N=32` 的 `B` 操作数），使每个线程束通道（lane）将其 16 个 BF16 输出合并为单条 **32 字节向量存储（`16 -> 1` 条存储指令，`16.0x` 指令压缩）**，仅在 `out.data_ptr() % 32 == 0` 时由主机侧路由；同时在 epilogue 最后一次读取后立即释放 Tensor Memory（TMEM）并重叠屏障初始化，端到端微批次 GEMM 延迟从 `28.0 us` 降至 **`19.6 us`（`1.4286x` 加速）**。
2. **`flashinfer-ai/flashinfer@8a087fcc87b1e7baed17c67603fe3051c5066d8a` & `47c17dd7aef8d9084e645c3f2e6ea3d32682b084` (`csrc/cake_minimax_h3_sm120_*`)**：
   - **SM120 环形阶段读取与 TMA 重填之间的 `fence.proxy.async.shared::cta`**：在 MiniMax-H3 SM120 MLP（FP8 W8A8 与 NVFP4 W4A4）的 FC1/FC2 核中，每次 `ab_empty` 屏障等待后插入 `fence.proxy.async.shared::cta`，严格排序共享内存读取与下一轮 TMA 异步重填（在 `10,000` 轮迭代中将未加栅栏时的 `18` 次重填写后读/读后写竞争降为 **`0`**）；同时优化 SM120 NVFP4 pre-attention GEMM 输出瓦片光栅顺序，在 RTX 5090 上实现 **`1.0494x`** 提速。
3. **`flashinfer-ai/flashinfer@b3059f79e1fd77541f6cdddb78672086eca405c1` (`flashinfer/msa_ops/_sm90_dispatch.py`, `cute_dsl/*_sm90.py`)**：
   - **Hopper SM90 MiniMax-M3 稀疏注意力（MSA）CuTe DSL 算子集**：新增 `proxy_score`、`topk=16`、`sparse_prefill` 与 `sparse_decode` 的 SM90 原生实现（要求分页交错 `K/V` 缓冲与 FP8 E4M3 精度），消除 H100 上回退至 Triton 的开销（解码延迟从 `112.0 us` 降至 **`44.8 us`，`2.50x` 加速**）。
4. **`flashinfer-ai/flashinfer@ed8b6b956dbcba4729a1378585e49902c9308761` & `2725b4ec0e378700c415e3354bdc90307f8bca4f` (`csrc/fused_moe/warp_decode/*`)**：
   - **11 档动态共享内存预留扩散 Finalize 网格与流序链首启动**：由于 `512` 线程 FC2 CTA 无法与 Finalize CTA 共驻同一 SM，`cake_warp_decode` 为 Finalize CTA 预留 `118,784 B` 至 `19,456 B` 共 **11 档动态 SMEM**，将每 SM 驻留上限控制在 `1..11` CTAs，从而把 64 个 Finalize CTAs 从拥挤在 `8` 个 SM 扩散到 **`64` 个 SM（`8.0x` SM 扩散度，Finalize 尾延迟从 `18.0 us` 降至 `6.0 us`，`3.0x` 提速）**；同时对先读 `topk_ids` 再等待的链首核取消盲目 PDL 序列化，改为流序启动（消除 `12 -> 0` 次陈旧路由读取竞争）。
5. **`NVIDIA/cutlass@ffa119a125`、`7107b05535` 与 `tile-ai/tilelang@03211b85cc`、`eaa9468d60`**：
   - CuTeDSL `Vector.reduce()` 支持 `reassoc` fastmath 树状归约（8 元素向量归约深度从 `8` 步降至 **`3` 步，`2.6667x` 深度压缩**）；`export_to_c` 为 `cute.sym_int64()` 动态维度生成 `int64_t` 形状槽位（消除 `int32_t` 截断引发的第 2 槽位起 **100% 结构体偏移错读**）；TileLang 自动调优器将有序 `compile_flags` 纳入持久化缓存哈希键（编译标志冲突率降至 **`0`**），并在 `thread_storage_sync.cc` 中补齐跨线程同迭代共享内存 WAW 冲突检测（**`100%` WAW 冲突插障覆盖**）。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：当 `cake_warp_decode` 拥有 `G = 64` 个 Finalize CTAs，若不预留动态 SMEM 则每个刚释放 FC2 的 SM 会塞入 `8` 个 Finalize CTAs，导致仅 `8` 个 SM 串行承担归约带宽；通过预留 `118,784 B` 动态 SMEM 将每 SM 限制为 `1` 个 CTA 后，计算激活的 SM 数量与聚合 L1/SMEM 写带宽提升倍数。
- **可证伪假设**：若在 `out.data_ptr() % 32 != 0` 的未对齐输出张量上强行路由 `cake_tinygemm2_tcgen05` 的 Swap-AB `M=64` 瓦片，32 字节向量存储指令将触发未对齐地址异常（主干通过主机侧 `out.data_ptr() % 32 == 0` 门控回退至安全核来证伪越界风险）。
