# 分布式 Checkpoint 去重、分桶与 GDS/POSIX 恢复算例

本目录提供与 [训练存储、数据加载与分布式 Checkpoint 恢复路径](../../docs/components/storage-checkpoint.md) 配套的**原创标准库 CPU/offline 合成算例**（证据状态：`own-offline`）。

## 1. 验证的源码与规范不变量

基于固定版本 `pytorch/pytorch@1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340`（`v2.6.0`）`torch/distributed/checkpoint/` 与 NVIDIA GPUDirect Storage（`cuFile`）Overview Guide：

1. **跨 Rank 副本张量去重（`dedup_save_plans`）**：
   - 未去重总字节数：$R \times (S_{\text{sharded}} + \sum B_{\text{rep}})$；
   - `save_to_lowest_rank = True`（`_dedup_save_plans.py:39-40`）：所有跨 Rank 重复项全部落入 Rank 0，导致 Rank 0 写出峰值 $S_{\text{sharded}} + \sum B_{\text{rep}}$；
   - 默认 `save_to_lowest_rank = False`（`default_planner.py:224`）：按 `min(plan_indices, key=lambda idx: plan_to_size[idx])` 将重复项贪心均衡到当前计划最小的 Rank，消除 Rank 0 写出倾斜（算例中 8 Ranks 从 `17.18 GB` 降至 `15.57 GB`，倾斜比降低 `1.103x`）。
2. **单 Rank 多线程 LPT 分桶与 D2H Loader 退化边界（`filesystem.py:216-265`）**：
   - `_split_by_size_and_type` 按张量字节数降序排序并贪心放入当前最轻的线程桶；
   - `_OverlappingCpuLoader`（独立 CUDA stream + 默认 `inflight_threshhold = 10,000,000 B` 窗口流水化 D2H 与写盘）**仅在 `thread_count == 1` 且张量仍在 GPU 时启用**；当 `thread_count > 1` 且未经 `async_save` 预暂存至 CPU 时，退化为 `_SerialCpuLoader`（每个写盘线程内串行调用 `tensor.cpu()`）。
3. **`async_save`（`BlockingAsyncStager`）与 GPUDirect Storage（`cuFile`）对比**：
   - `async_save` 通过 `BlockingAsyncStager`（`staging.py:66-153`）先在主训练流/拷贝流中将 GPU 状态同步复制到 `pin_memory=True` 的 Host 内存中（阻塞步长仅 `0.506 s`），随后由后台单线程 Executor 执行 `_save_state_dict` 两阶段 `reduce_scatter("plan")` + `all_reduce("write")` 与 `.metadata.tmp -> .metadata` 原子提交；
   - GPUDirect Storage（`cuFileRead`/`cuFileWrite`）在满足 4 KiB 块对齐、非 `cudaMallocManaged`（UVM）且未超 GPU BAR1 窗口时，直接通过 PCIe/NVMe DMA 绕过 Host Page Cache（`host_memory_bounce_bytes_per_rank = 0`）；一旦触发未对齐或 UVM 条件，退化回 Compatibility Mode 主机内存反弹路径。

## 2. 复现与校验命令

```bash
python experiments/checkpoint_recovery/run.py --check
```

## 3. 限制声明

- 本算例仅验证静态源码分桶、去重、D2H 调度分支与一阶 Young-Daly 最优 Checkpoint 周期解析界，不代表真实 Lustre/WEKA/NVMe-oF 文件系统、PCIe Root Complex 或 GPU 硬件实测吞吐。
