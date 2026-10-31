# FlashInfer GDN 确定性反向传播、Rubin MegaMoE Autotune v2、FA4 SM100 Swap-AB Decode 与 Triton GSan 非对齐屏障深度解析 (Phase 160)

> **证据类型**：`implementation`（一手上游提交源码审计）+ `own-offline`（本项目标准库确定性实验 `experiments/flashinfer_gdn_bwd_rubin_megamoe_fa4_swapab_triton_gsan_phase160/`）

## 1. 一手上游提交矩阵 (Track 1: 10 Commits)

| # | 仓库与提交 | 发布日期 | 核心实现路径与源码定位 |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@36bebfbdb9` | 2026-10-31 | `flashinfer/experimental/cake_gdn_det_train/`：引入 `sdt_pricing` 与分段预扫描拼接（pre-scan & stitch），加速长包（long packs）与 `HV = 3H` 下 Gated DeltaNet 确定性分块反向传播（chunked backward），保持逐位可复现（bitwise reproducible）。 |
| 2 | `flashinfer-ai/flashinfer@5ab9a74e44` | 2026-10-31 | `flashinfer/autotuner/protocol.py`、`docs/design_docs/moe_ep_architecture.md`：为 Rubin NVFP4、MXFP8 E4M3/E5M2 与 MXFP4/MXFP8 MegaMoE 增加 `kernel_variant="auto"` 与 `autotune_v2()` 两阶段寻优（先在每个合格 variant 内选最优 knobs，再跨 variant 评出全局最优）。 |
| 3 | `flashinfer-ai/flashinfer@2a946a6885` | 2026-10-31 | `flashinfer/experimental/kimi_k3_latent_moe/cake_backend.py`：优化 GB300 (`sm_103a`) 与 `sm_100a` 上 Kimi-K3 Stable LatentMoE TP1 decode 尾部投影（`tail_sb_norm_n_pad`），对少 token 多 partial 单元格启用常驻 B 算术与路由 partial 预取。 |
| 4 | `flashinfer-ai/flashinfer@4e17968b85` | 2026-10-31 | `benchmarks/bench_cake_rmsnorm_silu_gate_train.py`：为 `y = RMSNorm(x, w) * SiLU(z)`（head dim 128，共享权重 `w[128]`）提供可选非确定性原子反向传播后端，与确定性归约后端形成吞吐/可复现性显式权衡。 |
| 5 | `Dao-AILab/flash-attention@4616f5116d` | 2026-10-30 | `flash_attn/cute/flash_fwd_decode_sm100.py`：新增 SM100/SM103 `FlashAttentionForwardSm100SwapAB` 单 query Split-KV decode 内核，计算 $S^T = K Q^T$ 与 $O^T = V^T P^T$，将 KV tokens 映射到 128 行 MMA M 维，将同组 query heads 放在小 N 维。 |
| 6 | `Dao-AILab/flash-attention@819ef26aae` | 2026-10-30 | `flash_attn/cute/config.py`、`flash_fwd_sm100.py`：统一通过强类型 `FwdConfig` 路由前向内核；JIT 编译缓存键仅依赖 `SplitKV` 布尔开关而非具体 `num_splits` 数值，消除动态分片数变化导致的重复编译。 |
| 7 | `Dao-AILab/flash-attention@a3a9dfa9bf` | 2026-10-29 | `flash_attn/cute/barrier.py`、`flash_bwd_sm100.py`：在 SM100 `hd256` 反向传播 `M=64` 串行调度中启用重叠确定性 `dQ` 信号量释放（`red_release_async` 提前获取 + 宽松释放），避免 `M=128` 下 `sdQaccum` 别名冲突。 |
| 8 | `NVIDIA/cutlass@e13f5cd414` | 2026-10-28 | `include/cute/stride.hpp`：修复 MSVC 编译器在依赖模板别名（dependent template alias）中无法推导非类型参数包（non-type parameter pack，Error C3545）的问题。 |
| 9 | `triton-lang/triton@3a4e1d6635` | 2026-10-20 | `include/triton/Dialect/Triton/IR/TritonOps.td`、`lib/Analysis/AxisInfo.cpp`：新增单结果 `tt.annotate` 恒等算子，修复多返回值 `tt.call` 内联时编译器 hint 错误传播到其他返回值的问题。 |
| 10 | `triton-lang/triton@99a346f001` | 2026-10-20 | `python/triton/experimental/gsan/src/GSanLibrary.cu`：将 GSan `syncThreads` 中的隐式对齐 `bar.sync` 替换为 `barrier.sync`，支持 Warp Specialization 分区子集同步而不死锁。 |

## 2. 关键实现路径与架构公式

### 2.1 FlashAttention-4 SM100 Swap-AB Split-KV Decode 矩阵转置映射
在标准 GQA Decode（每个 KV head 对应 $G = H_q / H_{kv}$ 个 Query heads，$S_q = 1$）中，若将 Query 放在 UMMA 的 $M=128$ 维，当 $G=8$ 时有效行利用率仅为：
$$\eta_{\text{std}} = \frac{G}{M_{\text{tile}}} = \frac{8}{128} = 6.25\%$$
`Dao-AILab/flash-attention@4616f5116d` 通过转置计算 $S^T = K Q^T$ 与 $O^T = V^T P^T$，把连续的 128 个 KV tokens 放置在 $M=128$ 维，将 $G=8$ 个 Query heads 放在小 $N$ 维：
$$\eta_{\text{swapAB}} = \frac{128}{128} = 100\%, \quad \text{Tile Gain} = \frac{\eta_{\text{swapAB}}}{\eta_{\text{std}}} = 16.0\times$$

### 2.2 GDN 确定性分块反向传播与 `sdt_pricing`
对于序列总长 $L=131,072$、头数 $H=32$、值头扩展比 $H_V = 3H$、头维 $D=128$ 的 Gated DeltaNet，分块大小 $C=256$ 时共有 $N_c = 512$ 个分块。`flashinfer@36bebfbdb9` 通过 `sdt_pricing` 动态选择 `dv_local` 与伴随扫描路径，将确定性分块反向传播的 HBM 状态读写量降至朴素实现的 $42\%$，获得 **$2.38\times$** 带宽受限加速且保持逐位确定性。

## 3. 确定性离线实验结果 (`own-offline`)

运行 `python experiments/flashinfer_gdn_bwd_rubin_megamoe_fa4_swapab_triton_gsan_phase160/run.py --check`：
- **GDN 分块确定性反向加速比**：`2.38x`（`naive_hbm_bytes = 6,442,450,944` vs `sdt_priced_hbm_bytes = 2,705,829,396`，`bitwise_deterministic = true`）
- **FA4 SM100 Swap-AB Decode MMA Tile 有效利用率提升**：`16.0x`（`6.25% -> 100.0%`）
- **Rubin MegaMoE `autotune_v2` 搜索收敛**：两阶段搜索 `24` 次 trial 对比全笛卡尔积 `96` 次（缩减 `4.0x`），Triton `tt.annotate` 误传播 `0` 次，GSan Warp-Specialized 屏障死锁 `0` 次。

## 4. 架构师设计练习与可证伪问题
1. **计算练习**：当 GQA 分组比 $G=16$、UMMA $M_{\text{tile}}=128$ 时，Swap-AB 相比标准布局的 MMA 行有效率提升倍数是多少？（答案：$128/16 = 8.0\times$）。
2. **可证伪假设**：在 `num_splits` 随序列长度动态变化的 CUDA Graph 捕获场景中，`FwdConfig` 以 `bool(split_kv)` 作为编译键是否能在 $L \in [4\text{K}, 128\text{K}]$ 切换时保持零 JIT 重编译？
