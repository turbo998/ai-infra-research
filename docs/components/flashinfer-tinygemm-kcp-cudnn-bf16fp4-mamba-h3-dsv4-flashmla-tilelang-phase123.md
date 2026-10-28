# FlashInfer CAKE `tinygemm2` `PACKED_M64_N8` / KCP Pool Sharing / Rubin `SM107` AOT / cuDNN 4-Slot Prefill Cache / Blackwell `bf16_fp4` / Mamba SSU / MiniMax-H3 `SM120` / DSV4, FlashMLA `SM100` WAR Barrier 与 TileLang 降级守卫深度解析（Phase 123）

> **English Version**: [flashinfer-tinygemm-kcp-cudnn-bf16fp4-mamba-h3-dsv4-flashmla-tilelang-phase123.en.md](flashinfer-tinygemm-kcp-cudnn-bf16fp4-mamba-h3-dsv4-flashmla-tilelang-phase123.en.md)  
> **关联问题 ID**：`q-gpu-flashinfer-tinygemm-kcp-cudnn-bf16fp4-mamba-h3-dsv4-flashmla-tilelang-phase123`（`priority: 132`）  
> **所属组件**：`gpu-software-stack`  
> **证据等级**：`implementation`（一手 GitHub 提交源码与 diff 审阅）+ `author-benchmark`（上游提交披露的 R200/GB300/B200/RTX PRO 6000 性能数据，明确与本仓库合成实验区分）+ `own-offline`（零依赖 Python 3.12 标准库解析器与几何验证）  
> **配套可复现实验**：[`experiments/flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123/run.py`](../../experiments/flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123/run.py)

---

## 1. 背景与架构师问题定义

在 Blackwell（`SM100`/`SM103`/`SM120`）与 Rubin（`SM107`，212 SMs）架构的 GPU 软件栈演进中，算子调度器、AOT 编译管线与 DSL 编译器面临四类关键架构挑战：
1. **Rubin `sm_107a` 窄 Batch GEMM 的 Cluster 同步开销与 AOT 缺失**：在 212-SM 的 Rubin R200 上，8-row 无偏置 GEMM 若沿用 `M128_N8_2CTA`（2-CTA cluster）只能发出 64 个 CTA 且需承担跨 CTA barrier 开销；同时 `flashinfer/aot.py` 未将 `SM100f` CUTLASS FP8/MoE 模块注册到 `has_sm107`，导致 vLLM DeepSeek-R1 FP8 在 Rubin 上触发 **789 s** 运行时 JIT 编译并突破 1,200 s 启动上限。
2. **上下文并行（KCP）通信缓冲池冗余与 cuDNN Prefill 几何切换开销**：`cake_kda_kcp` 为前向/反向各自独立分配发送池并急切分配 Ring 缓冲；而 cuDNN Prefill 图缓存仅保留单槽，导致多模态/动态 Chunk 交替几何每步重走完整 graph preparation。
3. **Blackwell/SM120 混合精度与状态空间算子精调**：`cake_blackwell_bf16_fp4` 2-CTA pair-tile 的权重/缩放加载时机、`cake_selective_state_update` 在 B200 上的 64-head/8-group z-gated 路由，以及 RTX PRO 6000（`SM120`）NVFP4 变长注意力的 Warp Specialization 分派阈值。
4. **FlashMLA `SM100` 跨半 Warpgroup 归约的 WAR 竞态与 TileLang 编译器类型守卫**：`head64` 稀疏解码在 `rowwise_max_buf[idx ^ 64]` 读取与 `rowwise_max_buf[idx] = li` 写入之间缺少 `NamedBarrier`；TileLang 将标量读取模系数塞入 16-bit `DataType::lanes()` 导致大步幅溢出。

---

## 2. 一手源码深度剖析

### 2.1 FlashInfer CAKE `tinygemm2` `PACKED_M64_N8`（`#7735`）与 `cake_kda_kcp` 缓冲池复用（`#7714`）
- **`PACKED_M64_N8` 单 CTA `tcgen05` 变体（commit `1d622fdb`）**：
  - 在 `TinyGemm2ProgramId` 中新增第 9 个程序枚举 `PACKED_M64_N8 = 8`，每个单 CTA 处理 `(16, 8)`（`output_features × batch_rows`），动态共享内存需求为 `101,376 B`，要求 `input_features_multiple = 64` 且仅在 `sm_107a`（Rubin R200）无 bias 的 8-row 窗口启用：
    $$\text{use\_packed\_m64\_n8} \iff (\text{arch} = \texttt{sm\_107a}) \land (M = 8) \land \neg\text{has\_bias} \land (K \equiv 0 \pmod{64}) \land \bigl((N=1536, K\in[2048,3072]) \lor (N=2048, K\in[2048,2560]) \lor (N=4096, K\in[3072,6144])\bigr)$$
  - 对于 `(8, 4096, 4096)`，`M128_N8_2CTA` 仅启动 $2 \times (4096/128) = 64$ 个 CTA（32 个 2-CTA cluster），而 `PACKED_M64_N8` 启动 $4096/16 = 256$ 个单 CTA block 铺满 R200 的 212 个 SM 且无跨 CTA cluster 同步开销。作者实测（`author-benchmark`）：R200 上 `8x4096x4096` 从 `6.24 us` 降至 `5.98 us`（**1.0435x**），53 个重路由形状取得 **1.01x–1.44x** 加速。
- **`cake_kda_kcp` 发送池别名与惰性 Ring 缓冲（commit `e4c7e4a6`）**：
  - 定义 `SHARED_POOLS = {"b1_send": "c1_send", "b2_send": "c2_send"}` 与 `LAZY_POOLS = ("ring_out", "ring_in")`。由于前向 `c1_send`/`c2_send` 与反向 `b1_send`/`b2_send` 生命周期严格互斥，反向直接复用前向底层 tensor；`ring_out`/`ring_in` 仅在 `exchange="ring"` 时按需分配。每个 `(device, dtype)` 池条目在 `all_to_all` 模式下从 10 个底层缓冲降至 **6 个**（节省 4 个缓冲，在 $B=2, H=32, D=128, V=128$ BF16 下节省 **8.0 MiB**），`ring` 模式下从 10 个降至 **8 个**。

### 2.2 Rubin `SM107` AOT 预编译注册（`#6795`）与 cuDNN 4 槽 Prefill 准备态缓存（`#7720`）
- **Rubin `SM107` AOT 注册（commit `4d0f564e`）**：
  - 在 `flashinfer/aot.py` 中，当 `has_sm107` 为真时，使用兼容 Rubin 的 `sm100f_arch` 注册 `gen_cutlass_fused_moe_sm100_module` 与 `gen_gemm_sm100_module_cutlass_fp8`。消除了 Rubin 运行时 **789 s** 的 NVCC JIT 编译耗时，使 vLLM DeepSeek-R1 FP8 GSM8K 启动+评测总耗时从 `1,254 s`（超时 `> 1,200 s`）降至 `465 s`（**2.6968x** 启动提速）。
- **cuDNN 4 槽 Prefill 图准备态复用（commit `42494623`）**：
  - 将单槽 `_cudnn_prepared_state` 扩展为容量 `_MAX_CUDNN_PREPARED_CACHE_ENTRIES = 4` 的 `_cudnn_prepared_cache` 字典，并在命中时通过 `_rebind_prepared_prefill_state()` 重新绑定当前步的 `qo_indptr_host`、`kv_indptr_host` 与页表元数据。作者实测（`author-benchmark`）：B200 上交替 Prefill 几何切换的首层 Host 耗时在 Profile B1 从 `121.69 us` 降至 `74.89 us`（节省 **46.80 us**，**1.6249x**），Profile B3 从 `117.31 us` 降至 `69.64 us`（节省 **47.67 us**，**1.6845x**）。

### 2.3 Blackwell `bf16_fp4`（`#7732`）、Mamba2 SSU（`#7724`）、MiniMax-H3 `SM120`（`#7730`）与 DSV4（`#7729`）
- **Blackwell `bf16_fp4` 2-CTA 预 Barrier 加载（commit `e9908762`）**：在 CuTe 布局 `M=128` 2-CTA pair-tile（`768x2112x2048`）内，将权重与缩放因子加载提前至 `barrier.cluster.arrive/wait` 之前，GB300 从 `9.264 -> 9.149 us`（**1.0126x**），B200 从 `10.101 -> 9.992 us`（**1.0109x**）。
- **Mamba2 `selective_state_update` `sm_100a` 路由（commit `6bac9852`）**：在 148-SM B200 上将 batch-64、`64h/8g`、`d=128, dstate=128` z-gated 单 token decode 路由至 `mtp_short_ring_heads64_groups8_t1_stp_z`，延迟从 `40.440 -> 38.368 us`（**1.0540x**；带 softplus 从 `40.480 -> 38.472 us`，**1.0522x**）。
- **MiniMax-H3 `SM120` NVFP4 Warp Specialization（commit `4b948ed8`）**：在 188-SM RTX PRO 6000 上新增 `ATTENTION_WS_NODELTA_LEAD` 与 `ATTENTION_WS_NODELTA_TAIL`（`384` 线程 = 8 个 QK/softmax warp + 4 个 PV warp，`80,896 B` 动态 SMEM，程序总数 `13 -> 15`），当未开启 delta 补偿且工作量满足 $\sum L_i^2 \in [2^{23}, 24576^2]$ 且加权平均长度 $\le 40960$ 时启用：`single_4096` 从 `603.01 -> 579.25 us`（**1.041x**），`seg3_ragged` 从 `688.10 -> 671.70 us`（**1.024x**）。
- **`cake_dsv4` 稀疏 MLA 调优（commit `1531ff91`）**：197 个评测形状中 GB300 上 89 个加速、B200 上 36 个加速，**0 个回退**。

### 2.4 FlashMLA `SM100` `head64` WAR Barrier（`#241`）与 TileLang 编译器守卫（`#3453`, `#3484`, `#3369`, `#3367`）
- **FlashMLA `SM100` 稀疏解码 WAR 竞态修复（commit `395cb3af`）**：
  - 在 `csrc/sm100/prefill/sparse/fwd_for_small_topk/head64/phase1.cuh` 中，跨半 warpgroup 归约先读取 `plan.rowwise_max_buf[idx_in_warpgroup ^ 64]`，随后将 `rowwise_max_buf` 复用作分母和 `li` 的暂存缓冲（`plan.rowwise_max_buf[idx_in_warpgroup] = li`）。若缺少同步，warp 0/1（lanes 0..63）可提前将 `li`（如 `128.0`）覆写进 `rowwise_max_buf[0..63]`，导致 warp 2/3（lanes 64..127）读到被污染的 `max`（误差达 `120.0`）。插入 `NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync)` 后彻底消除该 WAR 竞态。
- **TileLang 编译器守卫**：
  - `storage_rewrite.cc`（`#3453`, commit `0573808d`）：新增 `Optional<PrimExpr> scalar_read_coeff` 字段保存模索引系数，不再将其乘入 16-bit `DataType::lanes()`（避免 `coeff=65536, lanes=4` 时 `262,144 > 32767` 溢出报错）。
  - `lower_shared_tmem.cc`（`#3484`, commit `10866657`）：仅当外部调用的首参数在已验证的 `CUtensorMap` 声明集合中时才记入 `desc_name_map`。
  - `atomicadd_vectorize.cc`（`#3369`, commit `fab9a39f`）：限制 CUDA `AtomicAddx4` 仅对 `float32` 指针生效，拒绝 `fp16`/`bf16` 误向量化。
  - `gemm_sp.cc`（`#3367`, commit `fca340e5`）：强制校验稀疏 MMA 的 `K % micro_size_k == 0`。

---

## 3. 合成离线实验与验证结果（`own-offline`）

运行 `python experiments\flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123\run.py --check` 输出如下关键指标：

| 模块与验证项 | 变更前基线 | 变更后结果 | 收益 / 架构影响 |
| :--- | :--- | :--- | :--- |
| `tinygemm2` `(8, 4096, 4096)` (`sm_107a`) | `M128_N8_2CTA` (`64` CTAs, `6.24 us`) | `PACKED_M64_N8` (`256` CTAs, `5.98 us`) | **1.0435x** 提速，消除 2-CTA cluster barrier |
| `cake_kda_kcp` 每 `(device, dtype)` 缓冲数 | `10` 个急切分配缓冲 | `all_to_all`: `6` 个 / `ring`: `8` 个 | 节省 **4 个**（`8.0 MiB`）或 **2 个**（`4.0 MiB`）缓冲 |
| Rubin `SM107` DeepSeek-R1 FP8 启动耗时 | `1,254.0 s`（含 `789 s` JIT，超时） | `465.0 s`（AOT 预编译 `SM100f`） | **2.6968x** 启动提速，消除 1,200 s 评测超时 |
| cuDNN Prefill 4 槽缓存切换（B200 B1 / B3） | `121.69 us` / `117.31 us`（单槽 0 命中） | `74.89 us` / `69.64 us`（4 槽 5 次命中） | 首层 Host 耗时立减 **46.80 us / 47.67 us**（**1.68x**） |
| FlashMLA `SM100` `head64` 跨半归约误差 | `120.0`（`li=128.0` 提前覆写 `max=8.0`） | `0.0`（`NamedBarrier(128, wg0_sync)`） | 消除跨 warp WAR 数值破损 |
| TileLang `scalar_read_coeff`（`65536 × 4`） | 溢出 16-bit `DataType::lanes()` (`>32767`) | 独立 `PrimExpr` 字段保留 `coeff=65536` | 支持超大步幅 Buffer 的向量化标量读 |

---

## 4. 架构师的设计练习与可证伪问题

1. **计算题**：在 Rubin R200（212 SMs）上执行无偏置 `(M=8, N=4096, K=4096)` BF16 GEMM 时，为什么单 CTA `(16, 8)` 瓦片（256 CTAs，`101,376 B` SMEM/CTA）优于 `M128_N8_2CTA`（64 CTAs）？若 $N=8192$，单 CTA 块数变为 512（超过 2 waves），此时 2-CTA cluster 的权重共享收益是否会反超单 CTA？
2. **可证伪假设**：在 B200 上当交替出现的 Prefill 几何种类超过 4 种（如 6 种不同 `max_seq_len` 桶）时，`_MAX_CUDNN_PREPARED_CACHE_ENTRIES = 4` 将再次发生容量抖动；将容量提升至 8 的 Host 内存代价与命中率收益比是多少？
