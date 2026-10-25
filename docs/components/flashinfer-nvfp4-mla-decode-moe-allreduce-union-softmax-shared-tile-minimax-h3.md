# FlashInfer SM100/SM103 NVFP4 MLA 解码原生 E2M1×E4M3 反量化、延迟排序 One-Shot/Two-Shot MoE All-Reduce Union、Shared-Tile 簇级 Softmax 与 MiniMax-H3 256×448 MXFP8 FC1

## 1. 工程问题定义与核心收益

在大规模 Blackwell (`sm_100a` B200 / `sm_103a` GB300) 与消費级/工作站级 Blackwell (`sm_120a` RTX 5090 / RTX PRO 6000) 推理部署中，Decode 尾延迟与长序列吞吐受制于四个微架构瓶颈：

1. **DeepSeek-V4 NVFP4 MLA 解码反量化与在线 Softmax 关键路径过长**（`flashinfer-ai/flashinfer@923ff1321520`，`#7306`）：MLA 解码使用单行 512 宽的 NVFP4 潜在向量同时充当 K 与 V（`E2M1` 数值 + `UE4M3` 16 元素块缩放，64-token 页大小）。旧版内核将 `E2M1` 与 `UE4M3` 分别转换为 FP32 后再相乘，拉长了软件流水线寄存器生命周期与在线 Softmax 更新链。
2. **MoE Finalize + All-Reduce + Residual + RMSNorm 在中大 Token 批下的单轮（One-Shot）NVLink 扇入放大**（`flashinfer-ai/flashinfer@7888b26bf6b5`，`#7298`）：在 `H=7168`、`world_size ∈ {2, 4, 8}` 下，当 `T ≥ 64` 且 `WS ∈ {4, 8}` 时，One-Shot 策略要求每个 Rank 拉取 `(WS - 1)` 份完整行，导致跨 GPU NVLink 读取流量随 `WS` 线性膨胀。
3. **超宽词表（128K–256K）Cached-Cluster Softmax 的寄存器压力限制 SM 驻留率**（`flashinfer-ai/flashinfer@ce1758b8b38f`，`#7308`）：8-tile 簇内核若将全部 8 个 Tile（每线程 64 个 FP32）保存在寄存器中，每个 SM 仅能驻留 2 个 CTA；而 Persistent 变体虽能降低启动开销，但在多波次（Multi-Wave）中等行数下存在动态共享内存（67.7–68.7 KiB）限制。
4. **MiniMax-H3 扩散模型 MLP (`H=5376, I=14336`) 的 FC1 输出 Tile 宽度与 SM120 FP8 变长注意力 PV 交接延迟**（`flashinfer-ai/flashinfer@fd0e4fdd96b0` `#7304`、`b8ed186df31b` `#7334`、`53ecbde224f4` `#7333`）。

---

## 2. 第一手源码实现路径与数据布局审计

### 2.1 NVFP4 MLA Decode：`E2M1 × E4M3` 原生 `mul.rn.bf16x2` 反量化与 `cake-build/1` 契约 (`#7306`)

在 `flashinfer/experimental/nvfp4_mla_decode/cake_backend.py`、`cake_jit.py` 与 `csrc/cake_nvfp4_mla_decode/cake_nvfp4_mla_decode_4cd8124d834b7ab3a055_kernel.cu` 中：

- **数据布局与每 Token 字节数**：每个 MLA 潜在行宽 $D = 512$，包含 $512 \times 0.5\text{ B} = 256\text{ B}$ 的打包 `E2M1` 字节和 $512 / 16 = 32\text{ B}$ 的 `UE4M3` 块缩放因子，总计 **$288\text{ B/token}$**，按 `PAGE_SIZE = 64` 分页。
- **硬件原生反量化路径**：新内核 `4cd8124d834b7ab3a055` 将 `E2M1` 四元组与对应的 `UE4M3` 缩放因子同时解包至 `bf16x2` 寄存器对，直接执行原生 `mul.rn.bf16x2` 乘积累加，并缩短了在线 Softmax `ex2.approx.ftz.f32` 与运行最大值重缩放的依赖链。
- **AOT/JIT 统一清单（`cake-build/1`）**：引入 `cake_build.py` 与 `build.json`，支持无 FlashInfer 依赖的预编译 `aot/<name>.so`（校验 `build_cuda_version()` 与 `aot/build_info.json` 的 `cudart_version` 一致性）及 JIT 回退。
- **作者实测（Author-Benchmark，严禁混作本仓库硬件实测）**：
  - **GB300 (`sm_103a`)**：全 8 组基准测试加速 **`1.046x–1.124x`**（如 `B=32, q_len=1, kv_len=65536` 从 `598.32 us` 降至 **`532.19 us`**，即 **`1.124x`**，等效 KV 有效带宽 **`3.5481 TB/s`**；`B=32, kv_len=131072` 从 `1189.57 us` 降至 **`1058.39 us`**，**`1.124x`**）。
  - **B200 (`sm_100a`)**：加速 **`1.029x–1.074x`**（`B=32, q_len=1, kv_len=8192` 从 `105.30 us` 降至 **`98.02 us`**，**`1.074x`**）。

### 2.2 延迟排序 One-Shot / Two-Shot MoE All-Reduce Union (`#7298`)

在 `flashinfer/jit/cake_trtllm_moe_allreduce_union.py` 中：

- **统一算子边界**：单内核融合 MoE 专家加权归约（`active_expert_tokens * expert_scales`）、跨 GPU All-Reduce、残差相加（`residual_out`）与 RMSNorm（`norm_out`），固定 CTA 维度 `224` 线程、`__cluster_dims__(4, 1, 1)`（每 Cluster 896 线程）。
- **One-Shot 与 Two-Shot 流量公式**：设 Token 数为 $T$、隐藏维度 $H=7168$、元素字节数 $b=2$：
  $$\text{Bytes}_{\text{1-shot}} = (W - 1) \cdot T \cdot H \cdot b, \qquad \text{Bytes}_{\text{2-shot}} = 2 \cdot \frac{W - 1}{W} \cdot T \cdot H \cdot b$$
  理论跨卡远程读取比值为：
  $$\frac{\text{Bytes}_{\text{1-shot}}}{\text{Bytes}_{\text{2-shot}}} = \frac{W}{2}$$
  当 $W=4$ 时 Two-Shot 节省 **$1.3333\times$**（即 $2.0\times$ 远程读比值）；当 $W=8$ 时节省 **$4.0\times$** 远程 NVLink 读取流量！
- **内核路由表**：
  - 小 Token 批（`WS=2` 或 `WS=8, T≤16`）：采用延迟排序 One-Shot 变体 `lat_b4`（`min_blocks=4`, `max_clusters=142`）与 `lat_late_b5`（`min_blocks=5`, `max_clusters=175`），优先按到达延迟顺序发射本地归约与远程推拉。
  - 中大 Token 批（`WS∈{4,8}, T≥64`）：切换至 Two-Shot 变体 `two_shot_b4` 与 `two_shot_b5`（阶段 1 归约 $1/W$ 切片，阶段 2 广播已完成切片并在寄存器内直接计算 RMSNorm）。
- **作者实测（Author-Benchmark）**：
  - `WS=8, fp16, T=2048, E=8, PDL=off`：B200 从 `357.76 us` 降至 **`135.26 us`**（**`2.645x`**）；B300 从 `357.92 us` 降至 **`134.14 us`**（**`2.668x`**）。
  - `WS=8, bf16, T=256, E=8, PDL=off`：B200 从 `60.35 us` 降至 **`26.08 us`**（**`2.314x`**）；B300 从 `56.10 us` 降至 **`26.50 us`**（**`2.117x`**）。

### 2.3 Shared-Tile Blackwell Cached-Cluster Softmax (`#7308`) 与 MiniMax-H3 / SM120 优化 (`#7304`, `#7334`, `#7333`)

- **Shared-Tile Cluster Softmax (`csrc/cake_blackwell_softmax.cu`)**：
  - 新增 `SoftmaxRoute::kCachedClusterSharedTiles = 7` 及两个内核 `c8_t4_s4`（动态共享内存 `34,432 B`）与 `c16_t4_s4`（动态共享内存 `34,944 B`）。
  - 每个 CTA 将前 4 个 Tile 保留在寄存器中，并将额外最多 4 个 Tile 通过 `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` 批量拷贝至共享内存，使 8-tile 块的 SM 驻留率从 `2 CTAs/SM` 提升至 **`4 CTAs/SM`**（相比 Persistent 变体的 `67,712–68,736 B` SMEM 减半）。
  - 在 GB300 上 `32 × 256000` 从 `15.20 us` 降至 **`12.96 us`**（**`1.173x`**）；B200 上 `16 × 256000` 从 `10.69 us` 降至 **`9.89 us`**（**`1.081x`**）。
- **MiniMax-H3 `256 × 448` MXFP8 FC1 (`flashinfer/diffusion_ops/minimax_h3_mlp.py`)**：
  - 新增 `minimax_h3_mlp_mxfp8_w448`，将 2-CTA Pair 的输出 Tile 从 `256 × 256`（128 个 SwiGLU 输出列，`112` 个 N-tile）扩大到 **`256 × 448`**（224 个 SwiGLU 输出列，仅 **`64` 个 N-tile**），并在 TMEM 累加器半区交错执行 SwiGLU，在 B200 上提速 **`1.023x–1.060x`**，在 B300 上提速 **`1.033x–1.068x`**（`rows=4096`: `208.16 -> 194.88 us`, **`1.068x`**）。
- **SM120 FP8 Varlen Attention (`#7334`) 与 Out-Proj (`#7333`)**：
  - `#7334` 在持久循环末尾提前为下一个工作单元（`nslot = slot + num_bids`）预取 Q/K0/V0 TMA 张量并提前释放 PV 阶段交接，RTX 5090 加速 **`1.055x`**（`34.91 -> 33.09 us`），RTX PRO 6000 加速 **`1.041x`**（`27.68 -> 26.59 us`）。

---

## 3. 可复现 CPU/Offline 算例与验证

本仓库通过纯 Python 3.12 标准库脚本对上述数据布局、NVLink 流量比率与作者基准数据进行确定性建模验证：

```powershell
python experiments\flashinfer_nvfp4_mla_decode_moe_allreduce_union_softmax_shared_tile_minimax_h3\run.py --check
```

| 子系统 / 指标 | 参数配置 | 优化前基线 | 优化后结果 | 加速比 / 关键比值 | 证据分级 |
|---|---|---:|---:|---:|---|
| NVFP4 MLA Decode (`#7306`) | GB300, `B=32, q=1, kv=64K` (`288 B/tok`) | `598.32 us` | `532.19 us` | **`1.1243x`** (`3.55 TB/s`) | `author-benchmark` + `own-offline` |
| NVFP4 MLA Decode (`#7306`) | B200, `B=32, q=1, kv=8K` | `105.30 us` | `98.02 us` | **`1.0743x`** | `author-benchmark` + `own-offline` |
| MoE AllReduce Union (`#7298`) | B300, `WS=8, fp16, T=2048, E=8` (`two_shot_b5`) | `357.92 us` | `134.14 us` | **`2.6683x`** (`4.0x` 远程读节省) | `author-benchmark` + `own-offline` |
| Shared-Tile Softmax (`#7308`) | GB300, `32 × 256000`, `c16_t4_s4` (`34,944 B` SMEM) | `15.20 us` | `12.96 us` | **`1.1728x`** (`4 CTAs/SM`) | `author-benchmark` + `own-offline` |
| MiniMax-H3 MXFP8 FC1 (`#7304`) | B300, `rows=4096, 256×448 tile` (`64` N-tiles) | `208.16 us` | `194.88 us` | **`1.0681x`** | `author-benchmark` + `own-offline` |

---

## 4. 架构师设计练习与可证伪问题

1. **架构计算题**：当 `world_size = 8`、`H = 7168`、`dtype = bf16`、`T = 512` 时，计算 One-Shot 与 Two-Shot 模式下单 GPU 的跨 NVLink 远程读取字节数。
   - **解答**：单 Rank 行字节数为 $512 \times 7168 \times 2 = 7,340,032\text{ B} = 7.0\text{ MiB}$。One-Shot 远程读取 $(8-1) \times 7.0\text{ MiB} = \mathbf{49.0\text{ MiB}}$；Two-Shot 远程读取 $2 \times \frac{7}{8} \times 7.0\text{ MiB} = \mathbf{12.25\text{ MiB}}$，精确降低 **$4.0\times$**。
2. **可证伪假设**：在 `vocab_size ≤ 65536` 且单 CTA 分块不超过 4 个 Tile（`max_tiles == 4`）时，`kCachedClusterSharedTiles` (`c8_t4_s4` / `c16_t4_s4`) 是否仍会被选中？
   - **证伪验证**：根据 `csrc/cake_blackwell_softmax.cu` 中 `cached_shared_tiles_kernel()` 的 `if (variant.max_tiles != kCachedMaxTiles) return nullptr;`（其中 `kCachedMaxTiles = 8`），4-tile 变体已可直接在寄存器内实现 `4 CTAs/SM` 驻留，因此不会走 Shared-Tile 路径。
