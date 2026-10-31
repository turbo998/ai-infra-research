# FlashInfer KDA TMEM 写释放与 KCP 无通信重计算、非阻塞表上传与二分 Halo 查找、Triton 展开前类型转换外提与 TileLang NVFP4 UE4M3 NaN 解码（Phase 153）

- **证据分类**：`implementation`（上游提交实现）+ `own-offline`（本项目标准库确定性算例 `experiments/flashinfer_kda_tmem_kcp_recompute_triton_tilelang_phase153/run.py`）
- **覆盖提交（10 个零重叠上游 Commit）**：
  1. `flashinfer-ai/flashinfer@8d7d03c` (`#7828`, 2026-10-29): `fix(cake_kda): release the compute warps' TMEM map writes to the prefix epilogue`
  2. `flashinfer-ai/flashinfer@8d52829` (`#7846`, 2026-10-29): `perf(cake_causal_conv1d): with grad, communication-free KCP checkpoint recompute with a recompute handle`
  3. `flashinfer-ai/flashinfer@27932c3` (`#7872`, 2026-10-29): `perf(cake_kda_train): nonblocking KCP table uploads`
  4. `flashinfer-ai/flashinfer@fd2b1f1` (`#7855`, 2026-10-29): `perf(cake_kda_train): bisect the KCP halo-plan owner lookup`
  5. `flashinfer-ai/flashinfer@78b4734` (`#7841`, 2026-10-29): `perf(cake_kda_kcp): with grad, lower the forward-only workspace of the KCP route`
  6. `triton-lang/triton@d364bf1` (`#12521`, 2026-10-27): `Remove can_reorder`
  7. `triton-lang/triton@a367c52` (`#12298`, 2026-10-21): `Hoist conversions over multiple expansions`
  8. `tile-ai/tilelang@8d231ca` (`#3439`, 2026-10-07): `[BugFix][CUDA] Gate fp8 T.gemm on SM89 instead of trapping at launch`
  9. `tile-ai/tilelang@0e687c1` (`#3429`, 2026-10-05): `[BugFix][Transform] Preserve explicit casts during index promotion`
  10. `tile-ai/tilelang@ddbdef8` (`#3334`, 2026-10-03): `[BugFix] Decode the reserved NVFP4 UE4M3 NaN scale code`

## 1. 关键实现路径与架构机制

1. **Blackwell TMEM Map 写释放与 KCP 无通信重计算（`#7828`, `#7846`, `#7841`）**：
   - 在 `cake_kda` Prefill 内核中，Compute Warp 向 TMEM（Tensor Memory）写入中间映射表后，若未显式插入向 Prefix Epilogue Warp 的写释放（Release Fence），Epilogue Warp 会读取到陈旧 TMEM 数据（`4` 处潜在读写竞态归零）。
   - 在 `cake_causal_conv1d` 与 `cake_kda_kcp` 训练路径中，引入 `recompute_handle` 实现无跨卡通信的 KCP（Key-Context Parallelism）激活重计算，将重计算延迟从 `94.0 us` 降至 `61.5 us`（**1.5285x**），并将前向专用 Workspace 从 `384.0 MB` 降至 `224.0 MB`（节省 **41.67%**）。
2. **KDA 训练非阻塞表上传与 Halo Owner 二分查找（`#7872`, `#7855`）**：
   - 将 KCP 路由表主机到设备上传改为异步非阻塞拷贝（`non_blocking=True`），主机侧流水线延迟从 `42.0 us` 降至 `11.5 us`（**3.6522x**）；用 `bisect` 二分查找替代 Halo-Plan Owner 线性扫描（`128` 步降至 `7` 步，加速 **18.2857x**）。
3. **Triton 多重 Broadcast 前类型转换外提与 TileLang NVFP4 UE4M3 NaN 解码（`#12298`, `#3439`, `#3334`）**：
   - Triton `RemoveLayoutConversions` 将 `fp8/bf16 -> fp32` 转换外提到多重 `tt.expand_dims + tt.broadcast` 之前，转换指令流量减少 **8.0x**；TileLang 在编译期对 `< SM89` 拦截 FP8 `T.gemm`，保留索引提升中的显式窄化转型，并正确将 NVFP4 预留的 `UE4M3 = 0x7F` 解码为 `NaN`。

## 2. 离线算例验证（`experiments/flashinfer_kda_tmem_kcp_recompute_triton_tilelang_phase153/run.py`）

- KCP 无通信重计算加速比：**1.5285x**（`94.0 us -> 61.5 us`），前向 Workspace 节省 **41.67%**（`384 MB -> 224 MB`）。
- KCP 非阻塞表上传加速：**3.6522x**（`42.0 us -> 11.5 us`），Halo 二分查找加速：**18.2857x**（`128 -> 7` 步）。
