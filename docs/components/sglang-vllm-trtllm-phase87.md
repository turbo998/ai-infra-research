# SGLang Mamba DCP 块网格对齐与 Kimi-K3 SM107 Split-K、vLLM RL NCCL 并发握手及 TensorRT-LLM Ulysses 融合解散列（Phase 87）

> **证据类型**：First-hand Implementation Audit (`implementation` + `own-offline` 确定性 Python 3.12 标准库算例 `experiments/sglang_vllm_trtllm_phase87/run.py`)。

## 1. 核心架构发现与第一手源码路径

| 上游提交 / PR | 关键源码路径 | 架构改动与定量指标 |
|---|---|---|
| `sgl-project/sglang#41701` (`8252e3af63`, 2026-10-25) & `#46442` (`94d2b61e38`, 2026-10-25) | `python/sglang/srt/managers/{schedule_policy,scheduler}.py`, `python/sglang/srt/mem_cache/kv_cache_configurator.py` | `#41701` 修复开启 DCP（Dynamic Checkpoint Placement）与 `enable_mamba_extra_buffer` 时 Mamba 混合模型 Chunked Prefill 截断偏离 `mamba_cache_chunk_size`（如 `64`）导致后续所有检查点捐赠全部失效的缺陷：`Scheduler.init_mamba_truncation_align()` 通过 `math.lcm(self.truncation_align_size, chunk_size)` 将截断对齐与既有 DSA K-pool 对齐（如 `lcm(24, 64) = 192`）合并，并在 `PrefillAdder.floor_chunked_tokens()` 中向下对齐到网格（同时保留请求尾块 `trunc_len == remaining` 与小于网格时的防死锁前进保证）。在 1024-token 请求、单步未对齐预算 `260` 下，DCP 检查点有效捐赠次数从 **1 次恢复至 4 次（4.0x）**。`#46442` 修复统一内存融合区域报错信息中缺失 `--kv-cache-dtype-draft` 标志名的排障盲区。 |
| `sgl-project/sglang#46464` (`1b112fd62a`, 2026-10-25) & `#45733` (`933853a3f2`, 2026-10-25) | `python/sglang/kernels/ops/gemm/{cutedsl_bf16_gemm,dense_bf16_gemm_warp_splitk}.py`, `python/sglang/multimodal_gen/*` | `#46464` 针对 NVIDIA `SM107` 上 Kimi-K3 融合 MoE 前置投影（`N=6016, K=7168`，其中 `[1536:2432]` 共 **896 个专家**的路由 Logits 需保持 FP32 精度后再加 `correction_bias` 选 Top-16），引入 `cp.async` + 4 个 Compute Warps + 2 个 A-Loader Warps 的 FP32 输出 Warp Split-K BF16 GEMM（`_K3_FRONT_WARP_SPLITK = {8: (8, 11), 16: (16, 9), 32: (32, 6)}`，环形缓冲区深度最高 11 级 = `88 KB` 在途 SMEM）。`#45733` 将精确 GPU 图像预处理（Lanczos 插值与输入校验）推广至 Flux、Qwen-Image 与 Cosmos3 多模态扩散流水线。 |
| `vllm-project/vllm#63460` (`c9dbc92630`, 2026-10-25), `#63496` (`8e11a8a49e`, 2026-10-25), `#60149` (`75fc05a53a`, 2026-10-25), `#63538` (`b70de52965`, 2026-10-25) | `vllm/distributed/weight_transfer/{nccl_common,nccl_engine,sparse_nccl_engine}.py`, `vllm/config/lora.py`, `vllm/model_executor/models/dory.py` | `#63460` 抽离 `build_worker_init_payload()` 与 `run_trainer_rendezvous()`，在单线程 `ThreadPoolExecutor` 中并发发起推理 Worker 的 `init_weight_transfer_engine` RPC 与 Trainer Rank 0 的 `open_trainer_endpoint`，统一稠密（支持 `packed` 双缓冲）与稀疏 NCCL 权重同步握手并确保异常在 Worker 线程安全退出后传播；`#63496` 在 `LoRAConfig.verify_with_kernel_config()` 阶段统一拦截全部 `MEGA_MOE_BACKENDS` 与 `"aiter_mega_moe"`，替代原先仅在 Step5 模型构造函数内硬编码检查 `"deep_gemm_mega_moe"` 的滞后报错；`#60149` 新增 Dory 模型支持；`#63538` 修复 `EmbeddingGemma2` 与 `DiffusionGemma` 缺失 `_get_max_soft_tokens` 的 bug。 |
| `NVIDIA/TensorRT-LLM#17147` (`1586d85375`, 2026-10-23), `#20270` (`87342f4474`, 2026-10-24), `#20307` (`39a44fab74`, 2026-10-23) | `cpp/tensorrt_llm/kernels/ulyssesPostUnscatterKernel.cu`, `cpp/tensorrt_llm/thop/ulyssesPostUnscatterOp.cpp`, `tensorrt_llm/_torch/pyexecutor/*` | `#17147` 为 Qwen-Image Ulysses 序列并行新增融合内核 `ulyssesPackedQkvPostUnscatterKernel`，直接消费 All-to-All 原始打包接收缓冲区 `[P, B, Sp, 3, H, D]` 并以 `VEC = 8`（`uint4` 16 字节向量化读写）单次 Launch 写出连续的 `Q, K, V` `[B, P*Sp, H, D]`（Grid X 维为 `3 * P * Sp`，将 3 次独立 Unscatter Launch 合并为 **1 次**），并在 Host 侧增加 `checkInt32Dim` / `checkInt32Product` / `checkInt32SumProduct` 防止 `int32` 索引溢出。`#20270` 收缩 PyExecutor Runner 接口并显式化 Executor 契约；`#20307` 新增 Cosmos3-Edge Reasoner 模型与权重映射器。 |

## 2. 数学模型与架构师计算法则

### 2.1 Mamba + DCP 截断对齐的最小公倍数（LCM）不变量
当混合模型同时启用需要对齐粒度 $A_{\text{dsa}}$（例如 DSA K-pool 块大小 24）与 Mamba DCP 缓存块网格 $G_{\text{mamba}} = 64$ 时，调度器截断对齐长度 $A_{\text{total}}$ 必须取两者的最小公倍数：
$$A_{\text{total}} = \operatorname{lcm}(A_{\text{dsa}}, G_{\text{mamba}}) = \operatorname{lcm}(24, 64) = 192$$
对任意候选截断预算 $L_{\text{trunc}}$ 与剩余待预填充长度 $L_{\text{rem}}$，网格下取整函数定义为：
$$\operatorname{floor\_grid}(L_{\text{trunc}}, L_{\text{rem}}, G) = \begin{cases} L_{\text{trunc}}, & \text{if } G \le 1 \lor L_{\text{trunc}} \ge L_{\text{rem}} \lor L_{\text{trunc}} < G \\ \lfloor L_{\text{trunc}} / G \rfloor \cdot G, & \text{otherwise} \end{cases}$$
该法则保证了：(1) 中间块严格落在 $G$ 的整数倍上，消除跨块偏移累积导致的 DCP 0 捐赠退化；(2) 当剩余预算 $L_{\text{trunc}} < G$ 时不向下取整为 0，从而避免长 Prompt 永久饥饿死锁。

### 2.2 TensorRT-LLM Ulysses 打包 QKV 解散列的线程块与向量化映射
给定序列并行度 $P$、批大小 $B$、每卡子序列长度 $S_p$、头数 $H$、头维 $D$（满足 $D \bmod 8 = 0$）：
- 每个线程块负责一个 $(qkv\_idx \in \{0,1,2\}, p \in [0, P), s_p \in [0, S_p), b \in [0, B))$ 切片，故 $\text{gridDim} = (3 \cdot P \cdot S_p, B, 1)$；
- 每行包含 $\text{vec\_per\_row} = D / 8$ 个 `uint4`（16B = 8 个 BF16/FP16 元素），$\text{blockDim.x} = H \cdot (D / 8)$。
当 $P=2, B=1, S_p=2048, H=24, D=128$ 时，$\text{gridDim.x} = 12,288$，$\text{blockDim.x} = 384$，单内核即可完成 $75.5\text{ MB}$ 读写重排。

## 3. 可复现实验验证（`experiments/sglang_vllm_trtllm_phase87`）

运行 `python experiments/sglang_vllm_trtllm_phase87/run.py --check` 通过全部确定性断言：
- `sglang_mamba_dcp_grid.combined_lcm_align == 192`，1024-token 预填充下 DCP 有效捐赠从 `1` 提升至 `4`；
- `sglang_k3_sm107_splitk.router_num_experts == 896`，`shape_n_k == [6016, 7168]`；
- `trtllm_ulysses_packed_qkv.grid_dim_x == 12288`，Launch 次数从 `3` 降至 `1`。
