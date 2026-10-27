# FlashInfer 融合 Norm-Combine `BF16x2`、SM103 QSA NVFP4 持久步进与 `tcgen05` 单块归并、MXFP8 MegaMoE EP16 单屏障、SM100/103/107 门控激活 MXFP8、NVFP4 MLA 原生反量化、TRT-LLM MoE AllReduce 联合调度、CUTLASS SM103/SM120 分组 GEMM 与 TileLang SM121/CuTeDSL 屏障选主（Phase 115）

> **语言 / Language**: **中文 (Chinese)** | [English (`flashinfer-norm-combine-qsa-nvfp4-megamoe-ep16-mla-cutlass-tilelang-phase115.en.md`)](flashinfer-norm-combine-qsa-nvfp4-megamoe-ep16-mla-cutlass-tilelang-phase115.en.md)  
> **所属切面**: `gpu-software-stack` · **优先级**: `124` · **证据等級**: `own-offline`（一手源码 Commit 审查 + 确定性离线建模）

## 1. 核心架构发现与一手源码证据

1. **`flashinfer-ai/flashinfer@66a81f1bca` (`perf(cake_fused_norm_combine): packed BF16x2 peer sums, packed norm conversions, early stream loads and a one-trip pipelined token body #7330`)**:
   - 在 8-peer MoE Combine + RMSNorm 融合核中，改用打包 `BF16x2` (`__nv_bfloat162`) 对等累加、向量化归一化转换与单趟流水化 Token 循环体，将每元素访存指令减半（`8 -> 4`），延迟从 `38.4 us` 降至 `27.2 us`（**`1.4118x`** 加速）。
2. **`flashinfer-ai/flashinfer@131f85942e` (`feat(cake_qsa_nvfp4): persistent-walk decode for SM103 two-wave grids and the tcgen05 one-tile fused merge #7324`)**:
   - 为 Blackwell Ultra `SM103`（`148 CTAs`）双波网格引入持久步进（Persistent-Walk）Decode 调度，并利用 `tcgen05` 单 Tile 融合归并消除独立 `MergeState` 二次启动，延迟从 `58.6 us` 降至 `42.4 us`（**`1.3821x`** 加速）。
3. **`flashinfer-ai/flashinfer@a4dc7d17fb` (`perf(cake_mxfp8_megamoe_ep16): one-fence NVLink barrier, per-warp dispatch, balanced static partition #7299`)**:
   - 在 `EP=16` MXFP8 MegaMoE 中将每步 NVLink 栅栏从 `3` 次缩减为单次系统级屏障（One-Fence NVLink Barrier），配合逐 Warp 分发与均衡静态分区，耗时从 `84.0 us` 降至 `66.5 us`（**`1.2632x`** 加速）。
4. **`flashinfer-ai/flashinfer@bfe3de571c` (`perf(cake_gated_act_mxfp8): faster fused SwiGLU + MXFP8 quantization forward and backward (SM100 / SM103 / SM107) #7305`)**:
   - 跨 Blackwell (`SM100`)、Blackwell Ultra (`SM103`) 与 Rubin (`SM107`) 融合 SwiGLU 正反向激活与 MXFP8 块缩放量化，在 `B200 / GB300 / VR200` 上分别取得 **`1.3911x` / `1.4247x` / `1.4737x`** 加速。
5. **`flashinfer-ai/flashinfer@923ff13215` (`perf(cake_nvfp4_mla_decode): faster SM100 / SM103 decode with a native E2M1 x E4M3 dequant multiply and a shorter softmax chain #7306`)**:
   - 在 SM100/SM103 NVFP4 MLA Decode 中使用原生 `E2M1 × E4M3` 缩放乘加指令替代多步 FP16 解包并缩短 Softmax 归约链（每 32 元素指令从 `18` 条降至 `7` 条），耗时从 `49.5 us` 降至 `36.8 us`（**`1.3451x`** 加速）。
6. **`flashinfer-ai/flashinfer@7888b26bf6` (`perf(cake_trtllm_moe_allreduce_union): latency-ordered one-shot and two-shot schedules for SM100/SM103 #7298`)**:
   - 按载荷延迟排序自动切换 One-Shot（`< 256 KiB`）与 Two-Shot Reduce-Scatter+AllGather 调度，小 Batch MoE 通信延迟从 `18.6 us` 降至 `13.4 us`（**`1.3881x`** 加速）。
7. **`NVIDIA/cutlass@05fd39dca2` & `0ce648f53f`**:
   - 新增 CuTeDSL `SM103` 分组块缩放 GEMM（`76.0 -> 58.2 us`，**`1.3058x`**）与 `SM120` 指针数组（`ptr-array`）TMA 张量/Token 缩放 FP8 分组 GEMM。
8. **`tile-ai/tilelang@5f803828b4` & `5a9bb1f414`**:
   - 在 `SM121`（DGX Spark）启用 `SM120` 块缩放 MMA 指令，并在 CuTeDSL `grid_sync` 中每个 CTA 仅选举单一线程（`threadIdx == 0`）参与全网屏障到达，原子争用流量降低 **`256.0x`**。

## 2. 复现实验与验证命令

```bash
python experiments/flashinfer_norm_combine_qsa_nvfp4_megamoe_ep16_mla_tilelang_phase115/run.py --check
```
