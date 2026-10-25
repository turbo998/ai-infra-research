# FlashAttention-4 SM100 2-CTA CLC 逻辑坐标契约、GQA 反向 `pack_gqa` 与 Per-CTA TMA 越界防护、veRL 单块锁页 LoRA 桶暂存及 PyTorch DeviceMesh/XPU 运行时加固

> **关联研究问题 (`question_id`)**：`fa4-sm100-clc-packgqa-tma-verl-lora-bucket-pytorch-devicemesh-xpu`（Phase 76，`priority: 85`）  
> **先修依赖 (`depends_on`)**：`mooncake-memfabric-rl-affinity-lmcache-packed-mp-megatron-dcpv2-torchtitan`  
> **关联证据 ID**：`ev-fa4-sm100-clc-packgqa-tma-verl-lora-pytorch-devicemesh-2026`  
> **本地可复现实验**：`experiments/fa4_sm100_clc_2cta_pack_gqa_ragged_tma_verl_lora_pytorch_mesh/run.py`  
> **英文版文档**：[`fa4-sm100-clc-2cta-pack-gqa-ragged-tma-verl-lora-pytorch-mesh.en.md`](fa4-sm100-clc-2cta-pack-gqa-ragged-tma-verl-lora-pytorch-mesh.en.md)

---

## 1. 核心架构问题与背景

在 Blackwell (`SM100` / GB300 / B200) 与 Hopper (`SM90` / H100) 的注意力算子、强化学习 (RLHF/RLVR) Rollout 权重同步以及分布式通信/XPU 运行时中，2026 年 10 月下旬修复并优化了六项关键底层问题：

1. **FA4 SM100 2-CTA Cluster Launch Control (CLC) 的坐标空间错位**：在 `2CTA` 模式下，同一 Cluster 的两个 CTA 必须协同处理同一个逻辑 `work_m` Tile；但此前 CLC 调度器按持久化模式传入 `use_cluster_idx=False` 与 CTA 级 `num_block`，`clc_work_to_coords` 向两个 CTA 分别返回 `block_idx * 2 + rank`，导致 `2CTA` MMA 混入了不同 Tile 的 $Q$ 行，在 GB300 (`FA_CLC=1`, varlen packed GQA `hd128`) 上产生高达 **`0.39` 的数值误差**。
2. **FA4 SM100 GQA 反向传播的 `cp.reduce.async.bulk` FP32 原子累加开销**：此前 SM100 GQA 反向对每个 Query Head 独立运行 Tile，并通过 `cp.reduce.async.bulk` 在全局内存以 FP32 原子归约 $dK, dV$ 及执行后处理 kernel；同时若盲目将 `qhead_per_kvhead` 个 Query Head 折叠进同一 $Q$ Tile，在包含长序列的变长 MQA 批次上会导致长 Tile 串行化并拉长关键路径（B200 上 `3946 us` vs 未打包 `2297 us`）。
3. **FA4 SM90/SM100 变长 K/V 5D Wrap-Around TMA 描述符的 `~128 KiB` 硬件预取越界**：`#2983/#3012` 引入的 5D wrap-around TMA 描述符缺少有界张量 footprint，TMA 硬件单元在部分越界 Box 后方会继续预取最多 **`~128 KiB`**；当 $V$ 或 $K$ 位于分配末尾且紧邻未映射虚拟地址空间时，直接触发 `cudaErrorIllegalAddress`（此前被 PyTorch 缓存分配器掩盖）。同时 H100 变长调度器的每 Batch Tile 对齐填充若未计入描述符工作区上限（如 `B=49, seqlen_q=128, H=6` 启动 `582` 个 CTA 却仅分配 `302` 个描述符槽位），会导致 CTA 覆写越界内存。
4. **veRL -> vLLM 桶式 IPC LoRA 权重同步的数万次小张量克隆与锁页分配**：在 MoE Per-Expert LoRA 同步中，`update_weights_from_ipc` 对每个收到的专家张量在设备端克隆两次，随后 vLLM `LoRAModel.from_lora_tensors` 又逐个执行 `.to("cpu", dtype).pin_memory()`，单次同步触发数万次微小设备分配、D2H 拷贝与锁页主机内存分配。
5. **PyTorch `DeviceMesh` 非升序子组拆分、XPU `_StaticXpuLauncher` 句柄泄漏与 Mooncake `msgpack-c 6.1.0` 时间戳编码**：`DeviceMesh._init_one_subgroup` 在非升序子组 rank 下破坏后端关于全局升序 rank 的契约；`Py_BuildValue("(OO)", ...)` 使 XPU kernel 胶囊引用计数变为 `2` 导致 `close()` 时永不释放 Level Zero `ze_kernel_handle_t`；Mooncake Conductor 在 `msgpack-c 6.1.0` 下将 `1.0` 等整值 `double` 编码为整数，导致 SGLang 拆包失败。

---

## 2. 第一手源码与提交审计

| 仓库与提交 | 合并日期 | 证据层级 | 核心源码路径与改动重点 |
|---|---|---|---|
| `Dao-AILab/flash-attention@98c1cee0ed59` (PR `#3034`) | `2026-10-24` | `implementation` + `author-benchmark` | `flash_attn/cute/flash_fwd_sm100.py` + `tile_scheduler.py`：统一 SM100 前向逻辑工作 Tile 契约 `num_work_m = ceil_div(seqlen_q, R * C * S)` 与 `work_q_block(work_m, stage, cta_rank)`；GB300 `FA_CLC=1` varlen packed GQA `hd128` 最大误差从 **`0.39` 降至 `0.0`（逐位一致）**。 |
| `Dao-AILab/flash-attention@bc52a40aef87` (PR `#2967`) | `2026-10-23` | `implementation` + `author-benchmark` | `flash_bwd_sm100.py`、`flash_bwd_preprocess.py`、`flash_bwd_postprocess.py`、`interface.py`：在 GQA 反向将 `qhead_per_kvhead` 个查询头折叠进 $Q$ Tile 行（$dK, dV$ 驻留 TMEM 直接写回，省去 FP32 原子归约与信号量），并加入 `_bwd_pack_gqa_heuristic` 关键路径保护；B200 (`1400 MHz`) `hd128` GQA 2k 提速 **`+13%..+26%`**、`hd256` 提速 **`+4%..+11%`**。 |
| `Dao-AILab/flash-attention@10b74d98fb26` (PR `#3071`) | `2026-10-23` | `implementation` | `flash_fwd_sm100.py`、`flash_fwd_sm90.py`、`flash_bwd_sm100.py`、`flash_bwd_sm90.py`：移除 5D wrap-around TMA 描述符，改为每 CTA 128B GMEM TMA 描述符原地重写序列末尾边界，并以 `rows // tile + 2 * batch + 1` 计算每头槽位上限（H100 `B=49, S=128, H=6` 从 `302` 槽位增至 `888` 槽位以覆盖 `582` 个启动 CTA）。 |
| `volcengine/verl@3c6d9b43b67c` (PR `#8352`) | `2026-10-23` | `implementation` + `author-benchmark` | `verl/workers/rollout/vllm_rollout/utils.py`：将每个接收到的 IPC LoRA 桶直接拷贝进 **1 块 `lora_dtype` 锁页 CPU 内存**并分发切片视图，省去 `_update_weights` 二次克隆并使 vLLM `.to("cpu").pin_memory()` 变为零拷贝 no-op；GB300 上 `4000` 张量 (`250 MiB`) **`165 -> 35 ms` (`4.7x`)**、`16000` 张量 (`500 MiB`) **`705 -> 131 ms` (`5.4x`)**。 |
| `pytorch/pytorch@020ab1c83528` (`#201949`), `8e9dd78a4c84` (`#200890`) & `kvcache-ai/Mooncake@12a7e9f10a41` (`#4906`) | `2026-10-25` | `implementation` | `device_mesh.py` 默认排序 `split_group` 子组 rank；`StaticXpuLauncher.cpp` 用 `Py_BuildValue("(NO)", ...)` 修复引用计数泄漏并用 `unique_ptr` 保证 `zeKernelDestroy -> zeModuleDestroy` 顺序；Mooncake `mooncake_conductor.cpp` 强制 `pack_double` 保持 `float64` 编码。 |

---

## 3. 关键数学推导与系统验证

### 3.1 FA4 SM100 2-CTA 逻辑工作 Tile 映射与行不相交证明
令 $R = \text{m\_block\_size} = 128$，$C = \text{cta\_group\_size} \in \{1, 2\}$，$S = \text{q\_stage} \in \{1, 2\}$。每个逻辑工作 Tile 覆盖 $R \cdot C \cdot S$ 行：
$$\text{num\_work\_m} = \left\lceil \frac{\text{seqlen\_q}}{R \cdot C \cdot S} \right\rceil, \quad \text{work\_q\_block}(\text{work\_m}, \text{stage}, \text{cta\_rank}) = (\text{work\_m} \cdot S + \text{stage}) \cdot C + \text{cta\_rank}$$
对应行跨度（stride）向量为：
$$\text{stride}(\text{row\_in\_cta}, \text{cta\_rank}, \text{stage}, \text{work\_m}) = (1, \; R, \; R \cdot C, \; R \cdot C \cdot S)$$
修复前，CLC 将 `work_m` 错误乘上 `2 + cta_rank`，使得 `work_m = 0, stage = 0, cta_rank = 1` 的 CTA 计算出 `q_block = (1 * 2 + 0) * 2 + 1 = 5`（对应第 `640..767` 行，而非正确的 `q_block = 1` 即第 `128..255` 行），导致 2-CTA 协同 MMA 读写错位。

### 3.2 FA4 SM100 GQA 反向 `pack_gqa` 启发式与 Per-CTA TMA 描述符工作区上界
1. **GQA 反向 `pack_gqa` 关键路径准则 (`#2967`)**：折叠 $g = \text{qhead\_per\_kvhead}$ 后，单个 KV Head 的 CTA 需迭代 $g \cdot \lceil \text{seqlen\_q} / \text{tile\_m} \rceil$ 步，而网格总 Tile 数缩小 $g$ 倍。仅当最长打包 CTA 的迭代数不超过每个 Cluster 槽位的平均工作量、且波次量化损失低于带宽收益时才开启 `pack_gqa`，从而避免 `MQA 16/1 hd256` 变长包在 B200 上从 `2297 us` 退化到 `3946 us`。
2. **Per-CTA TMA 描述符工作区安全上界 (`#3071`)**：由于变长调度器对每个 batch 最多填充一个 `(cluster, q_stage)` 大小的 Tile，每头最大 M-block 数为：
   $$M_{\text{blocks, bound}} = \left\lfloor \frac{N_{q,\text{rows}}}{\text{tile\_m}} \right\rfloor + 2 \cdot B + 1$$
   在 H100 (`B=49, seqlen_q=128, H=6, tile_m=128`) 上，$M_{\text{blocks, bound}} = 49 + 98 + 1 = 148$，总槽位数 $148 \times 6 = 888 \ge 582$（实际启动 CTA 数），彻底修复了旧公式分配 `302` 槽位导致的越界写。

### 3.3 veRL 单块锁页 LoRA 桶接收链分析
对于包含 $N$ 个专家张量的 IPC 桶（如 $N = 16000$ 个 $(16, 1024)$ BF16 张量，共 `500 MiB`）：
- **旧路径 (`main`)**：$2N = 32,000$ 次设备端克隆 + $N = 16,000$ 次 D2H 拷贝 + $16,000$ 次 `cudaHostAlloc` 锁页分配，GB300 耗时 **`705 ms`**；
- **新路径 (`#8352`)**：分配 **1 块** `500 MiB` 锁页主机内存 `block = torch.empty(sum(sizes), dtype=lora_dtype, pin_memory=True)`，将每个设备张量直接异步拷贝至对应切片视图 `dst.copy_(tensor)`，后续 vLLM `.to("cpu", dtype).pin_memory()` 直接返回原视图（0 次额外拷贝），耗时降至 **`131 ms`（提速 `5.4x`）**。
