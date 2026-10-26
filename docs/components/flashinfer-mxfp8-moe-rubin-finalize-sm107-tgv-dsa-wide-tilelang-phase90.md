# FlashInfer MXFP8 CuTe-DSL Fused MoE、Rubin 非融合确定性 Finalize、SM107 深流水 TGV、Cake DSA Wide-First Indexer、Kimi-K3 TP12 Tail、5-CTA MoE Finalize AllReduce、Blackwell Cached-Cluster Softmax 与 TileLang SM8x/Reduce 解析（Phase 90）

> **English Version**: [flashinfer-mxfp8-moe-rubin-finalize-sm107-tgv-dsa-wide-tilelang-phase90.en.md](flashinfer-mxfp8-moe-rubin-finalize-sm107-tgv-dsa-wide-tilelang-phase90.en.md)  
> **证据等级**：`implementation`（上游固定 commit 源码审查）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/flashinfer_mxfp8_moe_rubin_finalize_sm107_tgv_dsa_wide_tilelang_phase90/run.py`）。未运行真实 GPU/RDMA 硬件 benchmark。

## 1. 一手源码证据与关键实现路径

1. **CuTe-DSL Fused MoE `MXFP8 x MXFP8`（Nemotron 适配）**（`flashinfer-ai/flashinfer` PR `#6793`, commit `b92f112955`）：
   - 在 SM100/SM103 `cute_dsl_fused_moe` 中新增 `quant_mode="mxfp8"`（E4M3 权重与激活 + 每 32 元素 1 字节 `E8M0` 块缩放因子），复用 `W4A8` 内核模板并将 B 操作数切换为 FP8，同时接入 `QuantConfig(MXFP8, MXFP8)` 统一 MoE API。
   - **字节开销公式**：每元素有效字节数 $B_{\text{eff}} = 1 + \frac{1}{32} = 1.03125\text{ B}$，相比 BF16（$2.0\text{ B}$）节省带宽比 $\frac{2.0}{1.03125} \approx 1.9394\times$，在 $K=7168$ 时每行包含 $7168 / 32 = 224$ 个 E8M0 缩放块。
2. **Rubin (SM107) CuTe-DSL Fused MoE 非融合确定性 Finalize**（`flashinfer-ai/flashinfer` PR `#6950`, commit `5240384d8f`）：
   - 此前 SM107 `blockscaled_contiguous_grouped_gemm_finalize_fusion.py` 仅实现融合 `atomicAdd` scatter-add，导致 `use_fused_finalize=False`（SGLang 默认路径及 `--enable-deterministic-inference` 强制路径）抛出 `NotImplementedError`。该提交补齐 SM107 连续写出 `[M, top_k, K]` 中间缓冲并按固定顺序归约的确定性路径，消除原子浮点加法的非结合律误差。
3. **SM107 TGV GEMM 深 A/B 流水 Tactics (`29..46`)**（`flashinfer-ai/flashinfer` PR `#7049`, commit `f2ebfdc8c4`）：
   - SM107 共享内存容量高于 SM100/SM103（模型中 $320\text{ KiB}$ vs $228\text{ KiB}$），此前 TGV `cute_ext` 内核始终按 SM100 容量分配流水。新增 18 个 SM107 专属 tactics（ID `29..46`），复用 18 种 tile 形状并将 A/B stage 深度从 4 提升至 6（$1.5\times$ 流水深度）。
4. **Cake DSA Indexer `sm_103a` Wide-First 调度 (`>= 512` 均Tile)**（`flashinfer-ai/flashinfer` PR `#7406`, commit `35cba307a0`）：
   - 在 GB300 (`sm_103a`) 上，当每个工作单元平均 key tiles $\ge 512$ 时，主机调度计划优先选择 8 queries/unit 的展开式 wide scan 程序，而非 3-warp 窄扫描路径。
5. **Cake Kimi-K3 TP12 Tail 四段式 Token 路由与持久化 K3**（`flashinfer-ai/flashinfer` PR `#7402`, commit `8707500798`）：
   - 针对 GB200/GB300 NVL72 域内 12-rank BF16 LatentMoE tail，按 token 数 $M$ 划分四档：$M \le 8$ 单次生成启动；$M \in [9, 16]$ 采用 one-shot ESS K1 + K3-ESS；$M \in [17, 960]$ 采用 two-shot ESS K1 + K3-ESS；$M > 960$ 采用 two-shot persistent K3。
6. **Cake MoE Finalize + One-Shot Lamport AllReduce + Ordered RMSNorm (`5 CTAs/SM`)**（`flashinfer-ai/flashinfer` PR `#7392`, commit `ee376e9737`）：
   - 在 $H=7168$ 下融合 Top-K 专家加权归约、共享专家累加、One-Shot Lamport AllReduce、残差加与有序 RMS 交换，共 $32$ 个特化程序，以 `5 CTAs/SM` 驻留切分 small-token 与 large-token 路径。
7. **Cake Blackwell Cached-Cluster Softmax 驻留上限与 PDL Twins**（`flashinfer-ai/flashinfer` PR `#7404`, commit `01a04018fc`）：
   - 为 Blackwell FP32 row-major `[rows, vocab]` 采样 softmax 的 14 个 Cake 程序（含 7 个 row-caching cluster 与 2 个 shared-tile 程序）引入驻留上限 cluster spread 与 `sm_103a` PDL（Programmatic Dependent Launch）双生内核。
8. **TileLang SM8x 稀疏元数据校验、DeepSelect Top-K 与 FP4/FP8 Reduce 常量钳位**（`tile-ai/tilelang` PRs `#3335` `80364b3d97`, `#3304` `f8053068c7`, `#3189` `1a0830a50d`）：
   - `make_cutlass_metadata_layout_sm8x` 显式拒绝 `float8_e4m3fn` 等非白名单类型，防止静默回退生成错误 interweave 布局；
   - 新增无输入采样的确定性 `DeepSelect` Top-K；
   - 修复 `fp8_e4m3`/`float4` 在 `reduce_max/min` 初始化时生成非法 `inff` 字面量的问题（钳位至最大有限值 $448.0$ / $6.0$），并在 `CheckReductionDType` 中拒绝 $<8\text{ bits}$ 的子字节累加器。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：当 $M=512, \text{top\_k}=8, K=7168$ 时，Rubin SM107 非融合 finalize 的 BF16 中间张量大小是多少？
  - 答：$512 \times 8 \times 7168 \times 2 = 58,720,256\text{ Bytes} = 56\text{ MiB}$。
- **可证伪假设**：在 SM103a DSA Indexer 上，当每个工作单元平均 key tiles $< 512$ 时，强制使用 8-query wide scan 会因寄存器压力与尾部填充浪费而劣化延迟。
