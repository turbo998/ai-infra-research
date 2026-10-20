# TorchTitan Dist-MoE MXFP8 Mega 融合 SwiGLU 推理与 RL Ulysses CP 逐 Token 切分、TransformerEngine cuBLASMp WGRAD 工作区复用及 Megatron-FSDP 冻结模块钩子延迟

> **语言 / Language**: **中文 (当前)** | [English (`torchtitan-dist-moe-swiglu-rl-cp-te-cublasmp-wgrad-megatron-mfsdp.en.md`)](torchtitan-dist-moe-swiglu-rl-cp-te-cublasmp-wgrad-megatron-mfsdp.en.md)
> **证据等级**: `implementation` + `author-benchmark`（上游源码/提交）+ `own-offline`（本项目标准库确定性复现实验）
> **上游版本锚点**:
> - `pytorch/torchtitan@ed3c713afeb21a514714ff59fec4fd4bb2d3b09d` (`2026-10-20`, PR `#5465`)
> - `pytorch/torchtitan@9371a2ffcb8655d74e6e1c099d7ca7b756259cae` (`2026-10-20`, PR `#5424`)
> - `NVIDIA/TransformerEngine@1aaede2f9edcaf70020c112c673e0b16818bc6a4` (`2026-10-20`, PR `#3629`)
> - `NVIDIA/TransformerEngine@5eeb62a2fcb020af9778e3d29b14b5dc5c5ad50a` (`2026-10-20`, PR `#3685`)
> - `NVIDIA/Megatron-LM@d37b3a8fdb37af1b3a1d75720636af4c323bd522` (`2026-10-19`, PR `#8113`)
> - `LMCache/LMCache@64b05b8af88d5fc13b375cdc06b5425cd56dea47` (`2026-10-19`, PR `#5791`)
> - `ai-dynamo/dynamo@2ae8a46299b1c7d0287d1ab8fe4b67018bafe560` (`2026-10-20`, PR `#16633`)

---

## 1. 核心架构发现与问题定位

### 1.1 TorchTitan Dist-MoE MXFP8 Mega 融合 SwiGLU 推理加速 (`ed3c713`)

在 TorchTitan RL 训练的 रोलアウト生成器（Generator）中，Dist-MoE 的 MXFP8（`32x32` 块缩放）原先复用了面向训练的 **Staged 流水线**（`W13` 与 `W2` 分为两个独立内核启动，且包含 DGRAD 缩放张量）：
- **架构机制 (`ed3c713`)**：
  1. 在 `DistMoeTransform` 中暴露 `mxfp8_mega_fused_inference=True`（要求 `expert_precision="mxfp8"` 且 `block_scaled_config.pipeline="mega"`）。
  2. 在 `_prepare_mxfp8_weight` 中，当 `inference=True` 时，仅生成 FPROP 所需的 `32x32` 缩放因子，将 `dgrad_scale` 设为空张量（`numel() == 0`，以满足 FSDP 管理字段不可为 `None` 的约束，并在 `prepared()` 中强制断言 `inference == (self.dgrad_scale.numel() == 0)`），同时对 `W13` 权重执行 Gate/Up 行交错重排（`interleaved_fc13 = inference and gate_up`），使 Mega 内核在同一个 Warp 的 `W13` Epilogue 内原位融合 SwiGLU 激活并直接衔接 `W2` GEMM。
  3. **数值与容差边界**：MXFP8 Mega 融合路径与 Staged MXFP8 路径**逐比特完全一致（Bitwise Identical）**，相对 BF16 的范数相对误差约为 `0.066`（测试容差阈值设为 `_MXFP8_REL_TOL = 0.12`，而若 Gate/Up 行未正确交错会导致误差飙升至 `~0.90`）。
- **作者实测性能 (`author-benchmark`, 单卡 B200, Qwen3-30B-A3B `EP=4` 单层 Dist-MoE)**：

| 路由后行数 $R$ | Staged BF16 ($\mu\text{s}$) | Staged MXFP8 ($\mu\text{s}$) | Staged 相对 BF16 加速 | Mega BF16 ($\mu\text{s}$) | Mega + Fused SwiGLU MXFP8 ($\mu\text{s}$) | Mega+SwiGLU 相对 BF16 加速 | Mega+SwiGLU 相对 Staged MXFP8 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `512` | 105.2 | 117.3 | 0.90x | 105.6 | **61.3** | **1.72x** | **1.91x** |
| `2,048` | 122.9 | 125.6 | 0.98x | 123.8 | **72.0** | **1.72x** | **1.74x** |
| `5,120` | 144.8 | 151.1 | 0.96x | 144.2 | **87.4** | **1.65x** | **1.73x** |
| `8,192` | 174.1 | 171.4 | 1.02x | 174.0 | **112.0** | **1.55x** | **1.53x** |
| `10,240` | 218.4 | 192.9 | 1.13x | 219.8 | **135.5** | **1.62x** | **1.42x** |
| `16,384` | 300.4 | 230.7 | 1.30x | 300.4 | **192.0** | **1.56x** | **1.20x** |
| `32,768` | 450.4 | 342.6 | 1.31x | 450.7 | **292.8** | **1.54x** | **1.17x** |
| `65,536` | 790.3 | 583.6 | 1.35x | 789.6 | **516.8** | **1.53x** | **1.13x** |

在单机 8x B200 的 **Qwen3-30B-A3B DAPO-Math RL** 端到端测试（Trainer `FSDP=4, EP=4` + Generator `DP=4, EP=4`，中位路由行数 $R \approx 9,600$）中，单步训练耗时加速从 `1.06x` 提升至 **`1.10x`**，Generator 解码步加速提升至 **`1.09x`**。

---

### 1.2 TorchTitan RL Trainer Ulysses Context Parallelism (`9371a2f`)

- **问题根因**：在开启上下文并行（`CP > 1`）时，`preprocess_inputs` 会沿打包 Token 维度切分 `labels`，但 RL 损失函数（GRPO / DAPO）消费的逐 Token 参数 `loss_kwargs`（`advantages`、`generator_logprobs`、`loss_mask`、`temperature`）却以未切分的全长张量传入损失函数，导致 `CP > 1` 无法训练。
- **修复机制 (`9371a2f`)**：
  1. 在 `torchtitan/distributed/context_parallel.py` 新增 `shard_per_token_tensors`，按当前 CP Rank 截取连续的 `[T / CP]` 分片；若配置了 `context_parallel_load_balancer` 则显式报错拦截（因为负载均衡重排会打乱 `labels` 顺序而与未重排的 `loss_kwargs` 失配，Ulysses CP 默认 `load_balancer=None`）。
  2. 由于 GRPO/DAPO 损失是掩码加权逐 Token 求和再除以全局有效 Token 数，且 FSDP 在 `(dp_shard, cp)` 上关闭了梯度除法（求和归约），各 CP Rank 局部求导之和严格等于全序列梯度。
  3. **实测验证 (`author-benchmark`, 4x GB300, Qwen3-0.6B)**：`CP=2 x TP=2` 相对 `FSDP=2 x TP=2` 的损失相对误差仅 **`2.0e-7`**（`0.3046408296`），单卡峰值活跃显存从 **`4.47 GiB` 降至 `3.92 GiB`（节省 `-12.3%`）**。

---

### 1.3 TransformerEngine cuBLASMp 0.11.0+ WGRAD 工作区零拷贝复用 (`1aaede2`) 与线性注意力全图编译 (`5eeb62a`)

1. **cuBLASMp WGRAD 复用 DGRAD All-Gather 输出 (`1aaede2`)**：
   - 在 cuBLASMp 通信+GEMM 重叠后端中，反向传播的 `dgrad`（AllGather + GEMM）已经聚合了 `grad_output`，但旧版 TE 在 `wgrad` GEMM 前又发起了一次冗余的 `gather_along_first_dim`。
   - 从 **cuBLASMp 0.11.0**（`nvte_cublasmp_version() >= 1100`）起，`nvte_all_gather_gemm(..., transb=false)` 保证将按 Rank 顺序拼好的全局输入 $B$（即 `grad_output`，形状 `(N, K)`）留在 `ctx->workspace` 的起始位置（偏移 `0`）。
   - TE 新增 `nvte_comm_gemm_workspace()` 与 `get_cublasmp_all_gather_output()`，在 `Linear`、`LayerNormLinear`、`LayerNormMLP (fc2)` 及融合 `te.ops` 反向传播中（针对非量化与 Per-Tensor FP8，且未开启 `delay_wgrad_compute` 时）直接将 `ctx->workspace` 映射为 `grad_output_total` 视图：
     - Row-parallel `Linear` 每次反向传播的 All-Gather 次数从 **`1 -> 0`**；
     - `LayerNormMLP` 每次反向传播的 All-Gather 次数从 **`2 -> 1`**（仅 `fc1` 仍需重新聚合输入）。
2. **线性注意力 `torch.compile(fullgraph=True)` (`5eeb62a`)**：
   - 将 `cudnn.linear_attention.ops` 算子解析从 `forward` 期的 `_import_linear_attention_op` 移至 `LinearAttention.__init__` 构造期缓存为实例属性 `self._op`，消除了 TorchDynamo 追踪 `importlib` 导致的图断裂，使 GDN、GDN2 与 DeltaProduct (GDP) 在 THD 变长前向、循环状态与反向传播中全面通过 `fullgraph=True` 编译。

---

### 1.4 Megatron-FSDP 冻结模块清理延迟 (`d37b3a8`)、LMCache 本地请求跳过 PD 传输 (`64b05b8`) 与 Dynamo TRT-LLM 死进程快速退出 (`2ae8a46`)

- **Megatron-FSDP (`d37b3a8`)**：当冻结模块（`_trainable_parameter_countdown.initial_value == 0`）的 `grad_input` 全为 `None` 时，PyTorch 的 `register_full_backward_hook` 会在模块内部反向传播完成**之前**提前触发，导致过早释放未分片权重（Unsharded Weights）。修复方案将其推入 `module.context._delayed_post_backward_callbacks` 延迟到反向结束统一清理，并在 `ModelChunkSchedulePlan.run` 中显式调用 `model.context.post_backward()`。
- **LMCache (`64b05b8`)**：`PDBackendAsync.batched_submit_put_task` 在 `transfer_spec is None`（本地非分离请求）时立即 `return`，避免错误执行 `mem_obj.ref_count_up()` 导致内存对象引用计数泄漏（`ref_count: 2 -> 1`）。
- **Dynamo TRT-LLM (`2ae8a46`)**：当 MPI 子进程崩溃使 `engine.get_health_check_fatal_error()` 记录了 `EngineDeadError` 时，`exit_worker_on_fatal_error` 先停止 Runtime 准入并**直接跳过 `engine.shutdown()` / `engine.cleanup()`**（避免向已死 MPI Rank 发送 IPC 导致永久死锁），立即以状态码 `1` 退出进程。

---

## 2. 本项目确定性复现实验 (`own-offline`)

- **实验脚本**: [`experiments/torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp/run.py`](../../experiments/torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp/run.py)
- **验证命令**: `python experiments/torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp/run.py --check`
- **不变量验证结果**:
  - `dist_moe_mega_swiglu_1_72x_and_empty_dgrad = true`
  - `rl_cp_per_token_shard_exact_and_mem_saved = true`
  - `te_cublasmp_0_11_eliminates_linear_wgrad_ag = true`
  - `mfsdp_lmcache_dynamo_lifecycle_safe = true`
