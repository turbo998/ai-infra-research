# 分布式训练 6D 并行与 FSDP2/DeepEP 状态与通信算例

本目录提供与 [分布式训练 6D 并行与 FSDP2/ZeRO 联合状态与通信机制](../../docs/components/training-parallelism.md) 配套的**原创标准库 CPU/offline 合成算例**（证据状态：`own-offline`）。

## 1. 验证的源码与规范不变量

基于固定版本 `pytorch/torchtitan@086bf6c166ec85c1298eb5596fa9bf95f6a2d840`（`v0.3.0`）与 `pytorch/pytorch@1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340`（`v2.6.0`）：

1. **6D 正交设备网格与 `fsdp` / `efsdp` 折叠（`parallel_dims.py:186-340`）**：
   - 全局乘积不变量：`pp * dp_replicate * dp_shard * cp * tp == world_size`；
   - 稠密层分片网格：`fsdp = dp_shard * cp`（`CP` Rank 同时参与权重/梯度/优化器 FSDP2 切分）；
   - MoE 专家层分片网格：`efsdp = (dp_shard * cp * (tp // etp)) // ep`，使单卡专家层静态除数 `efsdp * ep * etp` 与稠密层完全一致。
2. **FSDP2 双缓冲预取峰值与 `Shard(dim != 0)` Copy-Out 额外分配（`_fsdp_param_group.py:271-307` & `_fsdp_collectives.py:274-316`）**：
   - 标准 `Shard(0)` 下，前向隐式双缓冲预取同时驻留当前层 `unsharded_param` 与两份 `all_gather_output`（`comm_ctx.all_gather_state`），瞬态缓冲峰值为 `3x` 单层未切分参数；
   - 当 `fsdp_placement.dim != 0`（如与 TP 组合落在 `Shard(1)`）时，`foreach_all_gather_copy_out` 额外调用 `torch.empty_like(t)` 分配一份等大临时张量并执行 `chunk + cat`，将瞬态缓冲推高至 `4x`（算例中每层额外分配 `209.75 MiB`）。
3. **CP Ring Attention 因果掩码负载倾斜与首尾配对（`_attention.py:495-516, 1144-1191`）**：
   - 连续切分下，Rank `CP-1` 与 Rank `0` 的有效因果注意力计算比达 `(2*CP - 1) : 1`（`CP=8` 时为 `15.0x`，最大与均值比为 `1.875x`）；
   - `_RoundRobinLoadBalancer`（`ROUND_ROBIN_CYCLE = 2`）将序列切为 `2*CP` 份并按 `(chunk[r], chunk[2*CP - 1 - r])` 首尾配对，使所有 CP Rank 有效 FLOPs 严格相等（倾斜比 `1.0x`）。
4. **DeepEP v2 训练态 Compact 去重布局（`deepep.py:244-395`）**：
   - 当单卡持有多个本地专家（`E_local = E / ep > 1`）时，训练态 `do_expand=False` 在跨卡 All-to-All 中对落入同一目标 Rank 多个本地专家的 Token 仅传输 1 次（在 `E=64, ep=8, topK=8` 下期望发包从 `8.0` 降至 `5.4326`，节省 `32.09%` 流量）。

## 2. 复现与校验命令

```bash
python experiments/training_parallelism/run.py --check
```

## 3. 限制声明

- 本算例仅验证静态源码分片不变量、瞬态缓冲倍数与一阶集合通信带宽边界，不包含真实多机 GPU 集群或 RoCE/NVLink 硬件实测。
