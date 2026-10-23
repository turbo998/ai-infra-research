# FlashInfer Cake DSA 远端采样阈值、SM103a 稠密投影 GEMM 与 FP32 路由反向、KDA 文档分组批处理及 TileLang 向量化/归约降级防护

> **English Edition**: [flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang.en.md](flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang.en.md)  
> **Phase 63 · Priority 72 · Track A (`question-202`)** · 核验日：`2026-10-24`（UTC+08:00）

---

## 1. 核心架构问题与背景

在大规模混合精度长上下文训练与 MoE 投影流水线中，Blackwell（`sm_100a` GB200、`sm_103a` GB300、`sm_107a`）算子栈与 TileLang 编译器暴露出四类影响性能与正确性的关键工程瓶颈：

1. **DSA Indexer 在 Attention-Sink + Local-Band 分布下的重扫（Re-stream）退化**：`flashinfer.dsa_indexer.dsa_indexer_topk` 在 `sm_100a` / `sm_107a` 上对多 segment 行采用**逆序 tile 扫描（descending tile-order scan）**。当真实训练或推理轨迹同时包含起始位置的 attention sink（$j < 64$，加性偏置 $+5$）与查询附近的局部窗口（$q - j < 512$，加性偏置 $+3$）时，最早被采样的局部窗口高分 key 会人为抬高首轮阈值 $\tau_0$。当扫描随后抵达行首的 attention sink 区域时，大量高分 sink key 集中涌入 candidate pool 并引发 `pool_ovf`，迫使算子回退执行整行重扫（whole-unit re-stream）。
2. **稠密投影 GEMM 的 `sm_103a`（GB300）缺失、小 $N$ 波量化（Wave Quantization）与 FP32 路由器反向 `NotImplementedError`**：`cake_dense_projection_gemm` 此前仅编译 `104` 个实例（`47 sm_100a / 57 sm_107a`），不仅完全缺失 GB300 `sm_103a` 专属核，而且当 $N \in \{32, 48, 64, 128\}$ 时默认采用 `BLOCK_N=256` 产生严重空载 tile，同时 FP32 MoE 路由器的 `dgrad` / `wgrad` 直接抛出 `NotImplementedError`。
3. **KDA 长序列训练（$S=261,393, H=64$）反向检查点 Arena 显存爆炸**：`kda_train` 反向激活检查点（activation checkpointing）Arena 默认按每 `(token, head)` 分配 `4,488 B`，在 $S=261,393, H=64$ 下单层 Arena 高达 **71.7 GiB**（`71,703 MiB`），直接耗尽单卡可用显存余量。
4. **TileLang 编译器 `tir::Analyzer` 局部绑定泄漏与 Batched 1D 归约尾块丢失**：`LowerTileOp` 在 `VectorizeRewriter` 前未克隆 `Analyzer`，在 `LowerParallelLoop` 的 `PartitionLoop` 中直接复用外层 `Analyzer`，导致循环区间绑定污染后续 `parallel` 循环的向量化判定；同时 `FinalizeReducerLowerer` 对 `layout_batch_size > effective_batch` 的批量 1D 归约仅发射首块（offset `0`），静默丢弃后续归约块。

---

## 2. 上游源码实现路径与关键机制

### 2.1 FlashInfer Cake DSA Indexer：远端采样首轮阈值与非 FTZ PTX 封装（`1ff9f67725` / `#7125`）

在 `flashinfer/dsa_indexer/csrc/dsa_indexer_topk.cu` 中，`#7125` 将 `sm_100a` 与 `sm_107a` 的逆序扫描核（`__CUDA_ARCH__ != 1030`）重构为 **Far-Sample First Threshold** 策略：
- **仅对远端采样建立初始阈值 $\tau_0$**：在首个采样 pass 中，跳过最后几个局部窗口 tile，仅在远端背景分布（far-sample keys）上做快速基数排序提取首轮过滤阈值 $\tau_0$。局部窗口 tile 与行首 attention sink 均留待主扫描（main scan）阶段与远端高分 key 共同竞争候选池，从而将单次扫描的候选池容量从紧耦合的局部峰值解耦，彻底消除 `pool_ovf` 触发的二次全量 HBM 读取。
- **共享内存压缩与非 FTZ PTX 原语**：窄行/拆分核的 `SMEM_TOTAL` 从 `207,104 B` 缩减 **12,160 B** 至 **`194,944 B`**；在 `dsa_indexer_common.cuh` 中新增 `add_f32x2_noftz` 与 `fma_f32x2_rn_noftz`（封装 `add.rn.f32x2` / `fma.rn.f32x2`，显式省略 `.ftz` 修饰符），确保 FP32 对数/指数累加在次正规数（subnormal）区间与 IEEE-754 严格一致。

| 基准行 | 形状描述（$T, T_{kv}$, segments, top-$k$） | GB200 旧版 (ms) | GB200 `#7125` (ms) | GB200 加速比 | GB300 (`sm_103a` 正序对照) |
|---|---|---:|---:|---:|---|
| `Z1` | 262k doc, CP16 zigzag rank 0 ($16384, 270336, 2, 2048$) | `22.588` | **`12.692`** | **1.78x** (`1.7797x`) | `11.954 -> 11.825 ms` (`1.01x`) |
| `Z1k4096` | `Z1` with $\text{top-}k = 4096$ | `23.654` | **`13.721`** | **1.72x** (`1.7239x`) | `13.489 -> 13.346 ms` (`1.01x`) |
| `Z1k256` | `Z1` with $\text{top-}k = 256$ | `22.138` | **`12.561`** | **1.76x** (`1.7624x`) | `11.797 -> 11.676 ms` (`1.01x`) |
| `S6sink` | $1 \times 262\text{k}$ causal doc, sink stats ($262144, 262144, 1, 2048$) | `366.792` | **`211.554`** | **1.73x** (`1.7338x`) | `207.289 -> 208.151 ms` (`1.00x`) |
| `P1sink` | Recorded P1 geometry with sink stats ($16231, 268757, 8, 2048$) | `5.479` | **`3.450`** | **1.59x** (`1.5881x`) | `2.656 -> 2.671 ms` (`0.99x`) |
| `Z2cp` | $8 \times 32\text{k}$ docs, CP16 zigzag rank 0 ($16384, 270336, 16, 2048$) | `3.317` | **`2.252`** | **1.47x** (`1.4729x`) | `2.164 -> 2.138 ms` (`1.01x`) |
| `S4sink` | $1 \times 64\text{k}$ causal doc, sink stats ($65536, 65536, 1, 2048$) | `19.751` | **`14.671`** | **1.35x** (`1.3463x`) | `14.129 -> 14.184 ms` (`1.00x`) |
| `Z2` | $8 \times 32\text{k}$ docs, full causal ($262144, 262144, 8, 2048$) | `43.909` | **`34.133`** | **1.29x** (`1.2864x`) | `30.892 -> 30.858 ms` (`1.00x`) |

> **关键对偶结论**：GB300（`sm_103a`）原本采用正序（ascending）扫描，天然先扫描行首 attention sink 建立稳定高阈值，因此在 `Z1` 上一直维持 `11.825 ms`；`#7125` 让 GB200（`sm_100a`）逆序扫描从 `22.588 ms` 降至 `12.692 ms`（**1.78x**），消除了跨代架构间的性能断层。

### 2.2 `cake_dense_projection_gemm`：`sm_103a` 扩展、小 $N$ 规则与 FP32 路由器反向（`2706a076ac` / `#7108`）

1. **三架构编译矩阵扩容**：编译实例数从 `104`（`47 sm_100a / 57 sm_107a`）增至 **`247`**（`80 sm_100a / 80 sm_103a / 87 sm_107a`）。同时在 `#7124`（`c01aa7779b`）中，grouped GEMM 通过编译宏 `-DWGRAD_OUT=16` / `-DWGRAD_OUT=32` 将 BF16 与 FP32 `wgrad` 折叠至同一份 C++ 源码（仅保留 `5` 个物理 `.cu` 文件）；在 `#7122`（`6b38488084`）中移除 `select_program()` 永不可达的 `c1_t8`、`c2_t8`、`c4_t8` 三个 Blackwell Softmax 核。
2. **小 $N$ 与波量化规则（`EXACT_WAVE_BLOCK_N = 224`）**：针对 $N \in \{32, 48, 64, 128\}$ 新增窄块程序，并引入 `EXACT_WAVE_BLOCK_N = 224` 对齐 SM 波次（wave occupancy），在 GB200 上将 `4096->512` FP32 `wgrad`（$T=16231$）从 `91.0 us` 降至 **`48.5 us`**（**1.878x**），在 GB300 上将 `5120->512` FP32 `dgrad` 从 cuBLAS BF16 的 `261.1 us` 降至 **`73.6 us`**（**3.548x**）。
3. **FP32 MoE 路由器三路拆分（Split-BF16x3）前反向闭环**：采用 `_ROUTER_BLOCK_K = 32, _ROUTER_BLOCK_N = 128`，将 `ROUTER_SPLITS` 配置为 `{"kk": 4, "kn": 1, "nn_t": 6}`（将转置路由器 `wgrad` 的 `nn_t` 拆分项从 `11` 项削减至 **`6` 项**），并实现融合 `_router_split_bf16_pair_t` 核。在 GB200 上 `router 5120->256 dgrad` 达 **1.943x**（`462.4 -> 238.0 us`）；在 GB300 上 `router 6144->256 fwd` 相比 `cuBLAS FP32 (BF16x9)` 达 **2.173x**（`488.5 -> 224.8 us`）。

### 2.3 Cake KDA 训练：文档分组批处理与分段路由保护（`fd5e515c6f` / `#7102`）

`#7102` 在 `kda_train_arena` 与 `kda_train` 中引入 `batch_tokens=B` 与 `max_docs_per_batch=M`：
- **贪心保序分组（`doc_batches`）**：沿 `cu_seqlens` 将变长文档切分为若干连续文档组 $(d_0, d_1)$，每组独立执行前向并在反向时复用同一个按最高组 token 数分配的 checkpoint Arena：
  $$\text{Arena}_{\text{grouped}} = \max_{g} \left(\sum_{d \in g} L_d\right) \times H \times \text{BytesPerTokenHead}_{\text{lean}}$$
  配合 `lean=True, tail_passes=4`（将每 `(token, head)` 字节从 `4,488 B` 压缩至 `2,820 B`），在不改变跨文档边界数学语义（各文档首状态均为 `h0`、末状态写入 `ht`）的前提下大幅削峰。
- **小链数分段路由保护**：当某文档组满足 `num_docs * H <= 4` 时，显式拒绝 `segmented` 比例分配路由并回退至 `grouped` 路由，规避极少链数下的尾块停顿。

| 基准配置（GB300, $S=261,393, H=64$） | 分组与策略 | 默认 Arena (MiB) | 优化后 Arena (MiB) | 显存缩减倍数 | B3 反向延迟变化 |
|---|---|---:|---:|---:|---:|
| `P14_h64_s261393`（4 文档：`23496, 88510, 114974, 34413`） | `lean=True, tail_passes=4`（不分组） | `71,703` | `45,054` | **1.59x** | `79.817 -> 79.746 ms` (`-0.1%`) |
| `P14_h64_s261393` | `lean=True, tail_passes=4`，**2 文档组** | `71,703` | **`25,755`** | **2.78x** | `79.817 -> 84.809 ms` (`+6.3%`) |
| `P14_h64_s261393` | `lean=True, tail_passes=4`，**3 文档组** | `71,703` | **`19,806`** | **3.62x** | `79.817 -> 86.083 ms` (`+7.9%`) |
| `EQ4_H64`（均衡 $4 \times 65,348$ tokens） | `lean=True, tail_passes=4`，**2 文档组** | `71,703` | **`22,538`** | **3.18x** | `80.884 -> 81.286 ms` (`+0.5%`) |
| `EQ4_H64`（均衡 $4 \times 65,348$ tokens） | `lean=True, tail_passes=4`，**单文档/组（4 组）** | `71,703` | **`11,280`** | **6.36x** | `80.884 -> 86.840 ms` (`+7.4%`) |

### 2.4 TileLang 编译器：Analyzer 克隆隔离与批量 1D 归约偏移循环（`fe8c34618f`, `741bf8b722`, `6db1fc771a`）

1. **`VectorizeRewriter` Analyzer 克隆（`fe8c34618f` / `#3612`）与 `PartitionLoop` 作用域隔离（`741bf8b722` / `#3226`）**：在 `src/transform/lower_tile_op.cc` 中，通过 `auto analyzer_clone = analyzer->Clone(); return VectorizeRewriter(analyzer_clone.get())(stmt);` 以及在 `LowerParallelLoop` 内部构造局部 `arith::Analyzer local_analyzer`，彻底阻断前序循环的 `Bind(var, Range)` 污染后续 `T.Parallel` 循环的模运算证明，修复多维/顺序并行循环无法向量化为 `float4`/`half2` 的退化。
2. **批量 1D 归约完整偏移发射（`6db1fc771a` / `#3070`）**：`FinalizeReducerLowerer` 此前仅在偏移 `0` 发射一次 `effective_batch` 归约；修复后外层包裹 `for (size_t offset = 0; offset < layout_batch_size; offset += effective_batch)`，确保 2D `(M, N)` 缓冲上的行归约覆盖全部 `layout_batch_size / effective_batch` 个批次。

---

## 3. 本项目确定性实验验证与架构师设计建议

本项目在 `experiments/flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang/run.py` 中对上述四项机制建立了零外部依赖的确定性验证器（单元测试：`tests/test_phase63_experiments.py`）：
- **架构选型建议**：
  1. 在 GB200（`sm_100a`）部署 DeepSeek Sparse Attention（DSA）长文本训练/推理时，必须升级至包含 `#7125` 远端采样首轮阈值的内核，以消除 attention sink 诱发的 1.78x 逆序扫描退化。
  2. 在 256k 级打包序列上运行 KDA 架构训练时，优先配置 `lean=True, tail_passes=4` 并将 `batch_tokens` 设为单批最长文档长度（或 2 组切分），仅以 `0.5%–6.3%` 的重计算开销换取 **2.78x–3.62x**（最高 **6.36x**）的 Arena 显存节省。
