# FlashInfer Rubin MoE 终归约共享缓存批量等待屏障、MoE EP 分离非本地路由剔除、SM103a 对端外层分块顺序与持续化 Kimi-K3 `tcgen05.wait::ld` 排空

> **English Version**: [flashinfer-rubin-moe-finalize-barrier-ep-split-sm103a-ag-matmul.en.md](flashinfer-rubin-moe-finalize-barrier-ep-split-sm103a-ag-matmul.en.md)

- **关联队列题 ID**：`flashinfer-rubin-moe-finalize-barrier-ep-split-sm103a-ag-matmul`（`GPU 软件栈`, `priority: 39`, `advanced`）
- **证据等级**：`implementation`（上游一手源码提交）+ `own-offline`（Python 3.12 标准库确定性归约屏障、EP 路由过滤与 SM103a 分块顺序仿真）
- **核验日期**：2026-10-21（北京时间 UTC+08:00）
- **一手上游来源**：
  - `flashinfer-ai/flashinfer` commit [`91b201a670ea5d86df8e0e4d601a510f606eac01`](https://github.com/flashinfer-ai/flashinfer/commit/91b201a670ea5d86df8e0e4d601a510f606eac01)（PR `#6307`, 2026-10-20）：`flashinfer/fused_moe/cute_dsl/rubin/blockscaled_contiguous_grouped_gemm_finalize_fusion.py`（blob `93bcb13737`）
  - `flashinfer-ai/flashinfer` commit [`69f082fe759f23e3e37d457da9075c11c4e15eee`](https://github.com/flashinfer-ai/flashinfer/commit/69f082fe759f23e3e37d457da9075c11c4e15eee)（PR `#6840`, 2026-10-20）：`flashinfer/moe_ep/backends/split/kernel/fused_moe/backend.py`（blob `86abac7266`）、`bridge.py`（blob `f0a697e2a7`）、`tests/moe_ep/test_compute_bridge.py`（blob `20f7925929`）、`tests/moe_ep/test_split_fused_moe_kernel_vs_reference.py`（blob `f5d6757fad`）
  - `flashinfer-ai/flashinfer` commit [`41b39c4af9ea7e3cad075effb97b571a57d51b78`](https://github.com/flashinfer-ai/flashinfer/commit/41b39c4af9ea7e3cad075effb97b571a57d51b78)（PR `#6841`, 2026-10-20）：`flashinfer/jit/cake_all_gather_matmul.py`（blob `53f099b216`）、`csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_k_major_sm_103a.cu`（blob `70649196ff`）、`csrc/cake_all_gather_matmul/cake_all_gather_matmul_ea2faa64e8c1ea61c8b8_kernel.cu`（blob `8ba43de81c`）
  - `flashinfer-ai/flashinfer` commit [`3bf6d3202239a107989451257c6f594cb5d2f008`](https://github.com/flashinfer-ai/flashinfer/commit/3bf6d3202239a107989451257c6f594cb5d2f008)（PR `#6740`, 2026-10-20）：`flashinfer/experimental/cake_kimi_k3_attn_res/csrc/cake_kimi_k3_attn_res/cake_kimi_k3_attn_res_0d77890adbb0f7175149_kernel.cu`（blob `90fcb9011e`）等 22 个持续化（Persistent）内核

---

## 1. 核心架构问题：Rubin/Blackwell 异步 TMA 归约、EP 分离零权重伪跳过与 TMEM 释放竞态

在 Blackwell Ultra（`sm_103a`）与 Vera Rubin 世代的 MoE / 张量并行融合内核中，异步硬件引擎（TMA Bulk Reduction、Tensor Memory `tcgen05.ld`）和统一专家算子（Unified Fused-MoE Runner）暴露出四类关键正确性与性能缺陷：

1. **Rubin MoE Finalize 共享内存 `sC` 提前覆写（`#6307`）**：
   在 `blockscaled_contiguous_grouped_gemm_finalize_fusion.py` 中，Epilogue 线程先将归约结果写入共享内存 `sC[sC_row, None, 0]`，再发起异步 TMA 批量归约写回全局内存，随后立即调用 `self.epilog_sync_barrier.arrive_and_wait()` 进入下一 Tile。由于线程级屏障并不等待 TMA 硬件引擎读完 `sC`，下一 Tile 的 Epilogue 线程会在 TMA 尚在读取时覆写 `sC` 行！
2. **MoE EP Split 后端中“零权重并不跳过 GEMM”（`#6840`）**：
   在 `RANK_MAJOR` 专家并行（EP）分派下，非本卡专家的选路被标记为 `-1`。旧版 `build_activation_pack_rank_major` 将非本地选路映射为合法的本地专家 ID `local_expert_offset` 并将权重置为 `0.0`。然而，统一 MoE Runner 仅按专家 ID 是否落在 `[local_expert_offset, local_expert_offset + num_local_experts)` 决定是否调度 GEMM，**权重为 `0` 的本地专家 ID 依然会完整执行无用的专家 GEMM**！
3. **SM103a TP=4 Gate/Up AllGather+Matmul 分块遍历顺序（`#6841`）**：
   在 `M <= 128`（单 M-Tile）、`N = 14336`（`56` 个 `block_n=256` 面板）、`ws=4` 的 TP4 Gate/Up 融合内核中，若采用对端内层（Peer-Inner）遍历，CTA 每处理一个 B 面板都需切换对端并检查对端就绪标志；切换为对端外层（Peer-Outer, `po`, `kMainFormPo=4`）可在单个对端就绪后连续扫完其全部 `56` 个 B 面板。
4. **持续化 Kimi-K3 Attention-Residual 内核中 `tcgen05.ld` 未排空即释放 TMEM（`#6740`）**：
   在 SM100 架构上，**寄存器读取（Register Read）并不会标记 `tcgen05.ld` 异步 Tensor Memory 加载指令完成**；只有显式执行 `tcgen05.wait::ld` 才能排空 `tcgen05.ld` 流水线。若在 `barrier.sync 9, 288;` 后由 `consumer_warp == 0` 直接释放 TMEM（`tcgen05.dealloc`），其余消费者 Warp 尚未完成的 `tcgen05.ld` 会触发硬件级 TMEM 释放冲突。

---

## 2. 一手源码机制走读与公式推导

### 2.1 Rubin MoE Finalize：`cp_async_bulk_commit_group` + `cp_async_bulk_wait_group(0, read=True)`（`91b201a670`）

在 `flashinfer/fused_moe/cute_dsl/rubin/blockscaled_contiguous_grouped_gemm_finalize_fusion.py`（blob `93bcb13737`）中，Epilogue 结尾在进入 `epilog_sync_barrier` 前插入显式 TMA 读完成等待：

```python
# The next tile reuses sC. Finish the async reductions' reads
# before any epilogue thread overwrites its shared rows.
cute.arch.cp_async_bulk_commit_group()
cute.arch.cp_async_bulk_wait_group(0, read=True)
self.epilog_sync_barrier.arrive_and_wait()
```

注意 `read=True` 参数：它仅需等待 TMA 引擎完成对源共享内存缓冲 `sC` 的**读取阶段**（无需等待全局内存原子累加落盘完成），从而在保证 `sC` 零覆写竞态的同时最小化屏障停顿。

### 2.2 MoE EP Split 远端哨兵专家 ID 映射（`69f082fe75`）

在 `flashinfer/moe_ep/backends/split/kernel/fused_moe/bridge.py`（blob `f0a697e2a7`）中，`build_activation_pack_rank_major` 接收全局专家数 `num_experts`：

$$\text{masked\_expert} = \begin{cases} (\text{local\_expert\_offset} + \text{num\_local\_experts}) \bmod \text{num\_experts}, & \text{if } \text{num\_experts} \neq \text{None} \land \text{num\_local\_experts} < \text{num\_experts} \\ \text{local\_expert\_offset}, & \text{otherwise} \end{cases}$$

当本卡仅持有部分专家（$\text{num\_local\_experts} < \text{num\_experts}$）时，$\text{masked\_expert}$ 必然落入本卡区间 $[\text{local\_expert\_offset}, \text{local\_expert\_offset} + \text{num\_local\_experts})$ 之外的合法远端专家 ID。统一 MoE Runner 在按本卡专家范围过滤时直接剔除该路由，连同 CUDA Graph 重放中动态变为全 `-1`（`torch.full_like(idx, -1)`）的批次也能零开销跳过。

### 2.3 SM103a `po`（`kMainFormPo=4`）分块双胞胎与 `tcgen05.wait::ld`（`41b39c4af9`, `3bf6d32022`）

- 在 `csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_k_major_sm_103a.cu`（blob `70649196ff`）与 `flashinfer/jit/cake_all_gather_matmul.py`（blob `53f099b216`）中，`MAIN_FORMS = ("cg1", "cg2", "cg2s1", "cg2s2", "po", "pi")`（对应 `main_form_code` `0..5`）。针对 `sm_103a` 的 `TILE_ORDER_POINTS["sm_103a"] = [[4, "bfloat16", "k_major", 128, 14336, "po"]]`（覆盖 `1..128` 行的单 M-Tile 场景），分派器启用单 CTA 对端外层内核 `kernel_cake_all_gather_matmul_ea2faa64e8c1ea61c8b8`（`THREADS=192`, `SMEM_TOTAL=197632` B）。
- 在 `cake_kimi_k3_attn_res_*_kernel.cu`（`#6740`）中，所有 22 个持续化内核在 `barrier.sync 9, 288;` 及 `consumer_warp == 0` 释放 TMEM 之前插入：
  ```cuda
  // Only tcgen05.wait::ld completes a tcgen05.ld (a register read does not); drain before the dealloc.
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  asm volatile("barrier.sync 9, 288;" ::: "memory");
  ```

---

## 3. 标准库确定性离线实验结果（`experiments/flashinfer_rubin_moe_barrier_ep_split_ag_matmul/run.py --check`）

| 评测维度 | 旧版基线行为 | 修复/优化后行为（`#6307` / `#6840` / `#6841` / `#6740`） | 量化收益 / 结论 |
|---|---|---|---|
| **Rubin MoE Finalize `sC` 复用（8 Tiles × 128 Rows）** | `epilog_sync_barrier` 仅同步线程，**7 / 8 Tiles（87.5%）** 发生 `sC` WAR 覆写 | 插入 `cp_async_bulk_commit_group()` + `cp_async_bulk_wait_group(0, read=True)`，**0 / 8（0.0%）** 覆写 | **消除 100% 共享内存 `sC` 覆写竞态** |
| **MoE EP Split 桥接器 5 组参数化矩阵** | 掩码路由指向 `local_expert_offset`，仅置零权重 | `(0,16->4), (8,16->12), (12,16->0), (0,4->0), (8,None->8)` **5 / 5 全通过** | **精确映射远端合法哨兵专家 ID** |
| **512-Token × Top-8 三轮 CUDA Graph 重放专家评估次数** | `4,096 + 4,096 + 4,096 = 12,288` 次本地专家评估 | `2,725 + 0 + 4,096 = 6,821` 次本地专家评估（全远端批次 `0` 次） | **剔除 5,467 次无效 GEMM 评估（`-44.49%`）** |
| **SM103a TP4 `M=128, N=14336` 对端切换次数** | 对端内层（Peer-Inner）遍历发生 **224 次**对端切换 | 对端外层（`po`, Form Code `4`）遍历仅 **4 次**对端切换 | **对端就绪切换削减 `-98.21%`** |
| **Kimi-K3 持续化内核 TMEM 释放风险 Warp 数** | 仅靠寄存器依赖，**9 / 9 个**消费者 Warp 存在未完成 `tcgen05.ld` | `barrier.sync 9, 288` 前执行 `tcgen05.wait::ld.sync.aligned`，**0 / 9** | **彻底根除持续化内核 TMEM 释放冒险** |

---

## 4. 解决方案架构师检视清单与可证伪问题

1. **架构师核查项**：
   - 在 CuTeDSL / CUTLASS 4.x 编写复用共享内存 Epilogue 缓冲（`sC`）的 TMA 归约算子时，必须检查 `epilog_sync_barrier.arrive_and_wait()` 之前是否配对了 `cp_async_bulk_commit_group()` 与 `cp_async_bulk_wait_group(0, read=True)`。
   - 在分离式 MoE EP（如 DeepEP / TRT-LLM / FlashInfer Split Backend）中，切勿假设将 `topk_weights` 置 `0` 即可跳过专家 GEMM；必须向桥接层传入全局 `num_experts` 并将非本地槽位重映射至远端专家 ID。
2. **后续可证伪问题**：
   - 若在 `num_local_experts == num_experts`（单卡持有全量专家，无远端分片）时出现 `-1` 掩码槽位（例如 Top-K 动态截断填充），如何在不引入越界专家 ID 的前提下让统一 MoE Runner 跳过 `-1` 槽位？

---

## 5. 关联证据 ID

- `claim-flashinfer-rubin-moe-finalize-sc-bulk-wait`
- `claim-flashinfer-moe-ep-split-nonlocal-route-exclusion`
- `claim-flashinfer-sm103a-po-tile-tcgen05-wait-ld-offline`
