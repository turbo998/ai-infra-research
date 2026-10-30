# FlashInfer Cake DSA Indexer 两趟基数选择融合、`KcpWorkspace` 跨流 `record_stream` 内存安全、B200 MoE Grouped GEMM 双 Die 非对称探测切分、`cake_sampling` Lean-Scan、`cake_warp_decode` FC2 预等待抢占、Kimi-K3 FP8 驻留融合解码、Rubin `sm_107a` `mega_mhc` 快速精确除法，及 Triton AMD `ScopedNoAliasAA` / NVIDIA `CGALayout` 与 TileLang `redux.sync` 分析（Phase 132 · Priority 141）

> **English Version**: [flashinfer-cake-dsa-kda-moe-sampling-warp-decode-mhc-triton-tilelang-phase132.en.md](flashinfer-cake-dsa-kda-moe-sampling-warp-decode-mhc-triton-tilelang-phase132.en.md)  
> **证据锚点**: `ev-flashinfer-cake-dsa-kda-moe-sampling-warp-decode-mhc-triton-tilelang-phase132-2026` (`own-offline` + `implementation`)  
> **可复现实验**: `python experiments/flashinfer_cake_dsa_kda_moe_sampling_warp_decode_mhc_triton_tilelang_phase132/run.py --check`

---

## 1. 第一手源码证据矩阵（Primary Source Matrix）

| 仓库与 Commit | 源码路径 | 核心架构变更 |
|---|---|---|
| `flashinfer-ai/flashinfer` [`85be8d0`](https://github.com/flashinfer-ai/flashinfer/commit/85be8d0392ed6801da94788ac0a8a3e262f96070) | `flashinfer/experimental/cake_dsa_indexer/csrc/cake_dsa_indexer_topk/cake_dsa_indexer_topk_d121d20b4ff827b4b6d2_kernel.cu` | 在 GB200 (`sm_100a`) 采样 `l6` 扫描程序及其 longest-first 副本中融合前两趟基数选择（radix-select）：第 1 趟同时对最高字节候选统计次高字节直方图，命中时直接跳过第 2 趟全局扫描 |
| `flashinfer-ai/flashinfer` [`9a7f792`](https://github.com/flashinfer-ai/flashinfer/commit/9a7f7923fc11a9f2412d3403553a28ac79aec0b4) | `flashinfer/experimental/cake_kda_train/cake_kcp.py` | 在 `KcpWorkspace` 发生 CUDA 流切换时，除执行 `wait_stream` 外对全部 5 个共享缓冲（`90–114 MB` arena、pools、scratch）调用 `record_stream`，阻止缓存分配器提前重用引发 UAF |
| `flashinfer-ai/flashinfer` [`30fe579`](https://github.com/flashinfer-ai/flashinfer/commit/30fe579954f6e92ec265361505124d3f72acb361) | `flashinfer/experimental/cake_moe_grouped_gemm/cake_jit.py` | 将 MoE Grouped GEMM 下投影数据梯度（`dgrad`，每组权重 $\le 6144 \times 2048$）的 4 KB 页延迟探测双 Die 任务队列切分（probed die split）扩展至 B200 (`sm_100a`，如 `35 / 39` 非对称 SM 对) |
| `flashinfer-ai/flashinfer` [`9377d72`](https://github.com/flashinfer-ai/flashinfer/commit/9377d725268b0ee63250fef0504568303400c4f3) | `csrc/cake_sampling/cake_sampling_binding.cuh` | 为 `V=32768`、`top-k 769..1024`（`batch 1..16`）在 GB300 与 B200 新增 `(4, 16)` 驻留核的 Lean-Scan 孪生构建（4-CTA Cluster，每线程 16 项） |
| `flashinfer-ai/flashinfer` [`e0c3efc`](https://github.com/flashinfer-ai/flashinfer/commit/e0c3efca66d1104cc86655e4173ac522567225b3) | `csrc/fused_moe/warp_decode/generated/cake_warp_decode_generated_manifest.cuh` | 在 SM100 20 个路由 MoE 解码形状（`e256_i512_k8` 16–32 Token 等）的 K512 FC2 核中，使工作申领 Warp 在执行 `griddepcontrol.wait` 等待 FC1 之前预先抢占至多 $\min(3, \lceil \text{grid\_m} \cdot \text{route\_tiles} / \text{launch\_ctas} \rceil - 1)$ 个任务项 |
| `flashinfer-ai/flashinfer` [`2e1789d`](https://github.com/flashinfer-ai/flashinfer/commit/2e1789dd0157d4e9c1ad05587070687e97d42264) | `flashinfer/experimental/kimi_k3_fp8_projection/cake_jit.py` | 在 GB300 (`sm_103a`) 将 Kimi-K3 `TP8 f_b` (`N=1536, K=128`) 于 `M=257..512` 切换至带 TMA-store 尾声的 32-token 驻留融合解码程序，移除独立的 64-token 两阶段解码 + 量化双启动 |
| `flashinfer-ai/flashinfer` [`ec856af`](https://github.com/flashinfer-ai/flashinfer/commit/ec856af63624f38457860d0c85b710cfefdd2848) | `csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_12e5bcc41e7940e7d511_kernel.cu` | 在 Rubin `sm_107a`（208/216 SMs）`mega_mhc` 的 NVFP4 与 FP8 group-128 原始缩放核中引入带范围守卫的快速精确舍入除法并修正 `seq % 3` mbarrier 轮转索引 |
| `triton-lang/triton` [`cdd55d3`](https://github.com/triton-lang/triton/commit/cdd55d35fc54106a4517313e2ada10732ec2cff9) | `third_party/amd/backend/codegen/DAGBuilder.cpp` | 在 AMD MIR 调度 DAG 发射器中注入 `ScopedNoAliasAA`，证明异步缓冲加载（`BUFFER_LOAD_DWORDX4_LDS`）与 LDS 读取（`DS_READ`）无别名冲突，消除约 `30%` 伪依赖停顿 |
| `triton-lang/triton` [`cc71215`](https://github.com/triton-lang/triton/commit/cc7121564aae5692195ff998af7426f1a8b916dc) | `lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp` | 在矩阵乘加速与 `PromoteLHSToTMem` 中保留源内存顺序但读取 `tt.dot` 操作数自身的 `CGALayout`（修正 `ConvertLayoutOp` 将 Split-K `[[1,0]]` 转为 Split-N `[[0,1]]` 时的 TMEM 布局错配） |
| `tile-ai/tilelang` [`5503a8b`](https://github.com/tile-ai/tilelang/commit/5503a8b2e91b5b479ca3774b455f71347f5ab83d) | `src/tl_templates/cuda/reduce.h` | 在全 Warp（32 线程、步长 1）归约中统一分派至硬件 `redux.sync`（SM80+ 32 位整型加/极值/位运算，以及 `sm_100a`/`sm_103a` 标量 FP32/BF16/FP16 极值归约），替代 5 步 `shfl.sync.bfly` 蝶形洗牌 |

---

## 2. 核心架构机制与确定性建模（Core Architectural Mechanisms）

1. **Cake DSA Indexer 两趟基数选择融合（`85be8d0`）、`KcpWorkspace` 跨流 `record_stream`（`9a7f792`）与 B200 双 Die 非对称探测切分（`30fe579`）**：
   - 在 `cake_dsa_indexer_topk` 的 8 趟 8-bit 基数选择（64-bit 键）中，第 1 趟（`shift=56`）同时维护最高字节匹配候选的次高字节直方图，命中时直接将扫描趟数从 **`8` 趟降至 `7` 趟**，在 GB200 上将 `Z3cp`（CP16 打包 4k–64k 文档）耗时从 **`1.420 ms` 降至 `1.386 ms`（`1.0245x`）** 且逐比特一致。
   - `KcpWorkspace` 在跨流复用时对全部 **`5` 个共享缓冲块（最高 `114.0 MiB`）** 补齐 `record_stream`，将旧流提前重用引发的分配器块冲突从 **`5 -> 0`**。
   - B200 双裸片在部分 SKU 上具有非对称 SM 对数量（如 **`35 / 39` SM pairs**，共 148 SMs）；通过对 `sched` 队列的 4 KB 物理页块执行 5 轮原子加零延迟探测，将 `dgrad` 下投影（$N \cdot K \le 12,582,912$）切分对齐至各自 Die 本地 HBM，获得 **`1.037x`** 实测提速。
2. **`cake_sampling` Lean-Scan（`9377d72`）、`cake_warp_decode` FC2 预等待抢占（`e0c3efc`）、Kimi-K3 FP8 驻留融合解码（`2e1789d`）与 Rubin `mega_mhc` 快速精确除法（`ec856af`）**：
   - `cake_sampling` 針對 `V=32768, top-k=1000` 引入 `(4, 16)` Lean-Scan 阶段 1 核，在 GB300 CUDA Graph 模式下将 `batch=8` 采样耗时从 **`14.352 us` 降至 `13.968 us`（`1.0275x`）**。
   - `cake_warp_decode` 在 `griddepcontrol.wait`（等待 FC1 完成）前允许申领 Warp 预取至多 **`3` 个工作项**，将 `e256_i512_k8`（23 tokens）解码延迟从 **`47.17 us` 降至 `45.76 us`（`1.0308x`）**；Kimi-K3 `TP8 f_b` (`M=512`) 在 GB300 上由两阶段双启动（2 launches）融合为单次 32-token 驻留核启动（**`1` launch**），延迟从 **`5.472 us` 降至 `4.640 us`（`1.1793x`）**；Rubin `sm_107a` `mega_mhc` NVFP4 核提速达 **`1.049x`**。
3. **Triton AMD `ScopedNoAliasAA`（`cdd55d3`）、NVIDIA `PromoteLHSToTMem` `CGALayout` 修正（`cc71215`）与 TileLang `redux.sync`（`5503a8b`）**：
   - `DAGBuilder` 启用 `ScopedNoAliasAA` 消除 `BUFFER_LOAD_DWORDX4_LDS` 与 `DS_READ` 间的伪别名边（消除 **`30.0%` 调度停顿**）；`PromoteLHSToTMem` 从 MMA 操作数而非源 `Load` 获取 `CGALayout`，消除 2-CTA Cluster 下 Split-K 与 Split-N 混用导致的跨 CTA TMEM 写冲突；TileLang 全 Warp 归约使用单条硬件 `redux.sync` 指令替代 **`5` 条 `shfl.sync.bfly` 指令（指令数压缩 `5.0x`）**。
