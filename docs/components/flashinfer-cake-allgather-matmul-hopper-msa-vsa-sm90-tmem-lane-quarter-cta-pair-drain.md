# FlashInfer Cake All-Gather Matmul (`SM100a`/`SM103a`)、Cake MSA/VSA Hopper (`SM90a`) 与 Blackwell `tcgen05` TMEM Lane-Quarter 及双 CTA Stage-Release 排空架构深挖

> **研究阶段**: Phase 46 (`priority: 55`) · **问题编号**: `q-phase46-flashinfer-cake-allgather-hopper-msa-vsa-tmem-race` · **确定性复现脚本**: [`experiments/flashinfer_cake_allgather_hopper_msa_vsa_tmem_race/run.py`](../../experiments/flashinfer_cake_allgather_hopper_msa_vsa_tmem_race/run.py)

---

## 1. 核心架构问题与一手提交总览

在 NVIDIA Hopper (`sm_90a`) 与 Blackwell (`sm_100a` B200 / `sm_103a` B300) 上，FlashInfer 的 Cake 内核栈同时面临三类底层软硬件协同瓶颈：

1. **融合通信-矩阵乘（Cake All-Gather Matmul）路由与 SM-Push 发射粒度不足**：在 B200 (`sm_100a`) `world_size=2` 中大问题规模（`1024x2048, K=8192`）下，默认单 CTA (`_cg1`) 拷贝引擎路由未利用双 CTA (`cta_group::2`, `_cg2`) 协同加载；而在 `world_size=4` (`512x2560, K=8192`, Llama-3.1-70B TP4 QKV) 与 B300 (`sm_103a`) `world_size=8` (`1024x1280, K=8192`, Llama-3.1-70B TP8 QKV) 的 SM-Push 路由中，每个远端 Peer 仅发射 `8` 个 Push CTA（总计 `24` 或 `56` 个 CTA），导致 NVLink 写入带宽在 GEMM 消费前无法打满。
2. **多阶段稀疏注意力（Cake MSA）与可变块稀疏注意力（Cake VSA）在 Hopper/Blackwell 上的调度失衡**：Hopper (`sm_90a`) 长期依赖通用 CuTe DSL (`fi_sm90`)，缺乏针对 sparse-decode / sparse-prefill / proxy-decode / proxy-prefill / top-k 的专用预编译路由表；Blackwell (`sm_100a`/`sm_103a`) 的 Union Prefill 按查询块顺序调度，GQA 查询头共享 KV 块导致的重尾块落在最晚批次，且 Top-K 归并仅用 1 个 Warp 串行扫描。
3. **Blackwell PTX ISA `tcgen05` 张量内存（TMEM）寻址与双 CTA 多播屏障退出竞态**：
   - `cake_dsa_train` 反向内核中，128 线程 Warpgroup 的 4 个 Warp 调用 `tcgen05.ld` 时仅传入列偏移 `0..31`，未将 `tmem_lane_base = (tid & 96) << 16` 写入 Bits `16..22`，导致 4 个 Warp 全部重复读取 TMEM Lane `0..31`；
   - `cake_alpha_moe` NVFP4 Epilogue 将线程级 `physical_feature = consumer_warp * 32 + lane` 左移 16 位传入 `tcgen05.ld`，违反了 PTX ISA 对 `taddr` 必须 Warp-Uniform 的硬件契约；
   - `cake_minimax_h3`（7 级流水）与 `cake_dense_projection_gemm`（49 个双 CTA 内核）在 `cta_group::2` 下由 Leader CTA (`cta_rank == 0`) 发出 `tcgen05.commit...multicast::cluster`（Mask `0b11`）信号至两个 CTA 的 `mma_done[s]`，但主循环仅等待 `step >= STAGES`，导致最后 `min(STAGES, K_steps)` 级屏障未排空时 Follower CTA (`cta_rank == 1`) 已提前退出并复写共享内存。

| 仓库与提交 | 日期 | 核心架构变更 | 实测性能 / 正确性收益 |
|---|---|---|---|
| `flashinfer@d9695940c5` (`#6865`) | 2026-10-21 | `cake_all_gather_matmul` SM100a `ws2` `1024x2048` 切 `_cg2`；`ws4` `512x2560` SM-Push 每 Peer 发射 `16` CTA（共 `48` CTA） | B200 `ws2` BF16/FP16 K-major 加速 **`1.065x`**（`101.9 -> 95.4 us`）；`ws4` 加速 **`1.086x`**（`127.5 -> 113.7 us`） |
| `flashinfer@9c68e44fd2` (`#6761`) | 2026-10-20 | `cake_all_gather_matmul` SM103a `ws8` `1024x1280` (`K=8192`) SM-Push 每 Peer 发射 `16` CTA（`7 × 16 = 112` CTA） | B300 TP8 QKV 最慢 Rank 中位延迟 **`324.8 -> 317.1 us`**（Node 1 `1.012x`，Node 2 `1.034x`） |
| `flashinfer@8a93c9b833` (`#6252`) | 2026-10-21 | `cake_msa` 新增 Hopper (`sm_90a`) **89 个**专用内核路由表（覆盖 4 大 MSA 算子与 Top-K） | H100 Top-K 较 `fi_sm90` 最高加速 **`3.237x`**（`13.775 -> 4.256 us`），Proxy Decode 加速 **`2.108x`** |
| `flashinfer@7f6f119d85` (`#6725`) | 2026-10-20 | `cake_msa` Blackwell Heavy-First GQA 折叠持久 Union Prefill + 8-Warp Rank-Count 流式 Top-K 归并 | GB300 / B200 BF16 Paged `first_topk` Prefill 加速 **`3.300x / 2.727x`**；Flat Union Prefill 加速 **`1.825x / 1.557x`** |
| `flashinfer@ddf080d8a5` (`#6877`) | 2026-10-21 | `cake_vsa_sm90` 独立包（**17 个** Hopper 变长块稀疏注意力内核 + `k1` 4-TMA-Box `V^T` 加载 + Ragged Queue） | H100 `ext33` 几何平均 **`1.0515x`**，`h1-m512-n8192-k48` 加速 **`1.2524x`**（`13.344 -> 10.639 us`） |
| `flashinfer@c8be875c7c` (`#6778`) | 2026-10-20 | `cake_dsa_train` 4 个反向内核 80 处 `tcgen05.ld` 补齐 `tmem_lane_base = (tid & 96) << 16` | 修复 Warpgroup 4 个 Warp 重复读取 Lane `0..31` 导致 3/4 累加器丢失的静默错误 |
| `flashinfer@e4e7510f51` (`#6767`) | 2026-10-20 | `cake_alpha_moe` NVFP4 Epilogue 66 处 `tcgen05.ld` 改为 Warp-Uniform `(consumer_warp * 32) << 16` | 满足 PTX ISA 9.7.16.10.5.2 Warp-Uniform `taddr` 契约，消除 Lane 1..31 越界读取 |
| `flashinfer@653cce0a6e` (`#6820`) & `2b3cd4f2f9` (`#6763`) | 2026-10-20 | `cake_minimax_h3` 与 `cake_dense_projection_gemm`（共 **50 个**双 CTA 内核）追加 `producer_tail` 排空 `mma_done` | 彻底消除 Follower CTA 提前退出引发的 `cta_group::2` 多播屏障踩踏与偶发数值损坏 |

---

## 2. Cake All-Gather Matmul：`SM100a` `_cg2` 路由与 `16`-CTA/Peer Push 发射几何

### 2.1 `SM100a` `ws2` `1024x2048` (`K=8192`) 双 CTA (`_cg2`) 切换

在 `flashinfer@d9695940c5` (`#6865`) 中，`cake_all_gather_matmul` 针对 B200 (`sm_100a`) `world_size=2`、输出形状 `1024 x 2048` (`K=8192`) 的拷贝引擎（Copy-Engine）路径，将原本路由到单 CTA (`_cg1`) 的三个分支切换至双 CTA (`_cg2`) 孪生内核：

| 数据类型与布局 | 升级前内核 (`_cg1`) | 升级后内核 (`_cg2`) | Node 1 中位延迟 (`us`) | Node 2 中位延迟 (`us`) |
|---|---|---|---|---|
| `BF16` N-major (`nt`) | `agm_ws2_m2048_n2048_k8192_bf16_nt_sm100a` | `agm_ws2_m2048_n2048_k8192_bf16_nt_sm100a_cg2` | `103.5 -> 98.4` (**`1.035x`**) | `102.4 -> 101.6` (**`1.027x`**) |
| `BF16` K-major (`nn`) | `agm_ws2_m2048_n2048_k8192_bf16_nn_sm100a` | `agm_ws2_m2048_n2048_k8192_bf16_nn_sm100a_cg2` | `101.9 -> 95.4` (**`1.065x`**) | `104.2 -> 99.6` (**`1.051x`**) |
| `FP16` K-major (`nn`) | `agm_ws2_m2048_n2048_k8192_fp16_nn_sm100a` | `agm_ws2_m2048_n2048_k8192_fp16_nn_sm100a_cg2` | `101.8 -> 95.3` (**`1.065x`**) | `104.4 -> 100.2` (**`1.052x`**) |

### 2.2 `SM100a` `ws4` 与 `SM103a` `ws8` 的 `16`-CTA/Peer SM-Push 发射模型

在 SM-Push 路径中，专用 Push 内核通过 NVLink 将本地分片直接推送到 `(world_size - 1)` 个远端 Peer 的对称缓冲，随后在本地流上执行 GEMM：

- **B200 (`sm_100a`) `ws4` `512 x 2560, K=8192`（Llama-3.1-70B TP4 QKV 投影）**：每个 Peer 需传输 $512 \times 8192 \times 2 = 8\text{ MiB}$。将 `kPushCtasPerPeer` 从 `8` 提升至 `16` 后，总 Push CTA 数从 $3 \times 8 = 24$ 增至 $3 \times 16 = 48$，单 CTA 负责字节数从 `1 MiB` 降至 `512 KiB`，BF16 N-major 延迟从 `127.5 us` 降至 `113.7 us`（**`1.086x`**），FP16 K-major 从 `128.8 us` 降至 `114.2 us`（**`1.082x`**）。
- **B300 (`sm_103a`) `ws8` `1024 x 1280, K=8192`（Llama-3.1-70B TP8 QKV 投影，`#6761`）**：每个 Peer 需传输 $1024 \times 8192 \times 2 = 16\text{ MiB}$。当 `M == 1024` 时对 `sm_103a` 启用 `16` CTA/Peer，总 Push CTA 数从 $7 \times 8 = 56$ 翻倍至 $7 \times 16 = 112$，最慢 Rank 中位延迟从 `324.8 us` 降至 `317.1 us`（Node 1 **`1.012x`**）及 `314.5 -> 311.2 us`（Node 2 **`1.034x`**）。

---

## 3. Cake MSA (`SM90a`/`SM100a`/`SM103a`) 与 Cake VSA (`SM90a`) 稀疏注意力内核

### 3.1 Hopper (`SM90a`) 89-Program Cake MSA 与 17-Program Cake VSA

- **`cake_msa` Hopper 后端（`#6252`）**：为 `cake_msa_sparse_attention`、`cake_msa_sparse_prefill`、`cake_msa_proxy_attention`、`cake_msa_proxy_prefill` 四大算子引入 **89 个** Hopper (`sm_90a`) 专用程序（`22` sparse-decode、`6` sparse-prefill、`24` proxy-decode、`5` proxy-prefill、`32` top-k）。在 H100 80GB 上，相比 `fi_sm90` CuTe DSL 基准，`topk_nvp_forced_h4_tiles512_tq256` 从 `13.775 us` 降至 `4.256 us`（**`3.237x`**），`topk_plain_h4_tiles512_tq256` 从 `14.079 us` 降至 `4.832 us`（**`2.914x`**），`proxy_decode_fp8_b4_s256_h32_k4` 从 `16.929 us` 降至 `8.031 us`（**`2.108x`**）。
- **`cake_vsa_sm90` 独立包（`#6877`）**：构建 **17 个** Hopper 变长块稀疏注意力程序（`2` 个持久队列内核 `attention` / `attention_queue` + `15` 个小规模专用内核），将 `k1` 的 $V^T$ 加载拆分为 4 个 $64 \times 16$ TMA Box 以消除循环寄存器溢出，并为长序列/单头（`total_q_blocks >= 8` 且 `num_heads <= 2` 或 `max_kv >= 32768`）启用预排重尾 Ragged Work Queue，在 H100 上使 `h1-m512-n8192-k48` 提速 **`1.2524x`**（`13.344 -> 10.639 us`），`h8-m128-n32768-k64` 提速 **`1.2126x`**（`35.136 -> 28.952 us`）。

### 3.2 Blackwell Heavy-First GQA 折叠持久 Union Prefill 与 8-Warp Top-K (`#6725`)

在 `flashinfer@7f6f119d85` (`#6725`) 中，Blackwell (`sm_100a`/`sm_103a`) `cake_msa` 针对 GQA 组内共享 KV Tile 的特性进行了三项关键重构：
1. **Heavy-First 持久工作队列**：按每个 Super-Tile 的 Union KV 块数降序排列工作项，通过 `grid.x = min(total_work, num_sms)` 持久 CTA 优先消化最重查询行，在复现模型中将 4-SM 尾部最大 KV 块负载从 `192` 降至 `100`（尾部不均衡比 **`1.92x`**）；
2. **GQA 折叠 Q-Tile**：当 `num_q_rows * gqa_ratio <= 64` 时，将同一 KV 组内的所有 Query Head 折叠进单个 `64`-Row MMA Tile，使共享 KV 页仅通过 TMA 加载一次；
3. **8-Warp Rank-Count 流式 Top-K 归并**：将原本仅 Warp 0 工作的 32 路归并改为 8 个 Warp 协作对 `8 * K <= 256` 候选集执行并行 Rank-Count 选择（GB300/B200 Top-K 归并提速 **`2.133x / 2.129x`**，BF16 Paged `first_topk` Prefill 整体提速 **`3.300x / 2.727x`**）。

---

## 4. Blackwell PTX `tcgen05` TMEM Lane-Quarter 寻址与双 CTA 多播屏障排空

```
  Blackwell 128-Lane TMEM Address Layout (PTX ISA 9.7.16.10.5.2):
  bits [31:23]       bits [22:16] (7-bit lane 0..127)        bits [15:0] (column)
  +------------+------------------------------------------+----------------------+
  |  reserved  |  tmem_lane_base = (tid & 96) << 16       |  k_col / feature_col |
  |            |  Warp 0: 0..31 | Warp 1: 32..63          |  (MUST be uniform    |
  |            |  Warp 2: 64..95| Warp 3: 96..127         |   across whole warp) |
  +------------+------------------------------------------+----------------------+
```

1. **`cake_dsa_train` Warpgroup TMEM Quarter 漏偏移修复（`#6778`）**：在 `tcgen05.ld.sync.aligned.32x32b` 中，每个 Warp 固定访问从其基地址起的 32 个 TMEM Lane。修复前 4 个反向内核（共 `4 × 20 = 80` 处 `tcgen05.ld`）直接传入 `tmem_col`，导致 Warpgroup 内 4 个 Warp 全部只读取 Lane `0..31`（仅覆盖 `32/128` Lane）；修复后注入 `uint32_t const tmem_lane_base = (tid & 96u) << 16`，使 Warp `0..3` 精确映射至 `0..31`、`32..63`、`64..95`、`96..127`（完整覆盖 **128** 个 Lane）。
2. **`cake_alpha_moe` NVFP4 Epilogue Warp-Uniform 基址修复（`#6767`）**：修复前代码计算 `physical_feature = consumer_warp * 32 + lane` 并将 `(physical_feature << 16)` 拼入 `taddr`，导致同一 Warp 内 32 个线程传入 32 个不同 `taddr`（违反 PTX Warp-Uniform 规范并在高 Lane 触发越界读取）。修复后改为 `warp_lane_base = (consumer_warp * 32) << 16`（跨 66 处 Epilogue 站点），保证每个 Warp 内基址严格唯一。
3. **双 CTA (`cta_group::2`) `multicast::cluster` `mma_done` 尾部屏障排空（`#6820`, `#6763`）**：在 `cake_minimax_h3`（`STAGES = 7`）与 `cake_dense_projection_gemm`（49 个 `cta_group::2` 内核，`STAGES = 4`）中，Leader CTA 通过 `tcgen05.commit.cta_group::2...multicast::cluster`（Mask `0b11`）向 Leader 与 Follower CTA 的 `mma_done[s]` 同时投递完成信号。新增的 `producer_tail` 在主循环结束后对最后 `min(STAGES, K_steps)` 个流水级逐一执行 `mbarrier_wait(mma_done + s, mma_phase[s])`，确保 Follower CTA 绝不在 Leader CTA 最后一次多播 Commit 落地前退出。
