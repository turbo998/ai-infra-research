# veRL Blackwell MXFP8 Megatron 训练与 SGLang/vLLM DeepGEMM 权重热更新、Raw Log-Prob 1-ULP 失配门控、VeOmni 低精度 ReduceScatter 及 Mooncake Intel XPU PCIe P2P + Level Zero IPC 与 Dynamo KVCR

## 1. 工程问题定义与跨栈协同挑战

在 Blackwell (`SM100`) 强化学习（RLHF / GRPO）后训练与多厂商异构 KV 缓存传输系统中，训练器与推理引擎之间的数值漂移（Rollout-Actor KL Mismatch）以及跨进程设备内存传输效率是核心瓶颈：

1. **Megatron-Core MXFP8 训练的 THD 变长打包对齐缺口与 QAT 互斥边界**（`volcengine/verl@d1390eea0c94`，`#8127`）：TransformerEngine (TE) 的 MXFP8 量化器沿 Token 维度以 32 元素块计算 `wgrad` 缩放因子，要求 GEMM 双维度均被 32 整除；而旧版 THD 预处理固定按 `lcm(16, align_size)` 对齐，导致 MXFP8 激活/梯度块越界。
2. **SGLang 与 vLLM 在 MXFP8 / MXFP4 / NVFP4 在线权重热更新（Weight Refit）中的 CUDA Graph 地址保持与算子不一致**（`volcengine/verl@5265d561a3bf` `#8306`、`619c397dc991` `#8308`）：分桶（Bucketed）权重同步若在每个 Bucket 到达时直接重建后端布局或替换底层 Storage，会破坏已捕获的 CUDA Graph 指针；且训练与推理若使用不同 FP8/FP4 内核（或误将 Hopper FP32 块缩放舍入为 `E8M0`），会加剧 `k3_kl` 漂移。
3. **`exp(log_prob)` 概率差指标对 Float32 1-ULP 差异失明**（`volcengine/verl@16ec093cc4a2`，`#8025`）：当 `actor_old_log_probs = -0.10000000149` 与 `rollout_old_log_probs = -0.10000000894` 相差 1 ULP 时，求指数后的 `exp(lp)` 在 FP32 下完全相等（`rollout_probs_diff_max == 0`），导致严格 0-diff 校验漏报。
4. **Intel XPU 同节点跨进程 KV 传输被迫绕行主机内存或网络栈**（`kvcache-ai/Mooncake@db4f146ecb5a`，`#4379`）：此前 Mooncake TENT 引擎在同机 Prefill/Decode XPU Worker 之间缺少 Level Zero IPC 与 PCIe/Xe Link P2P 直拷能力，且原生 Level Zero 隐式句柄缓存存在进程重分配后的陈旧映射（Stale Mapping）风险。
5. **Dynamo KVCR 循环层（Recurrent）事件误触发几何校验隔离与逐 Token 原子计数器缓存行颠簸**（`ai-dynamo/dynamo@8bf042187a8f` `#15535`、`906e89d3a304` `#17067`）。

---

## 2. 第一手源码实现路径与实测数据审计

### 2.1 veRL Megatron MXFP8 训练配方 (`#8127`) 与 SGLang/vLLM DeepGEMM 统一热更新 (`#8306`, `#8308`)

- **THD 序列填充对齐公式 (`verl/models/mcore/util.py`)**：
  - 定义 `align_size = tp_size * cp_size * 2`（当 `cp_size > 1`）或 `tp_size`（当 `cp_size == 1`）。
  - `_compute_fp8_thd_align_size(align_size, fp8_recipe)` 调用 Megatron-Core `get_fp8_align_size(fp8_recipe)`：对于 `mxfp8` 返回基数 `32`，每条序列填充至 **`lcm(32, align_size)`**；对于常规 FP8 块量化返回基数 `16`，填充至 **`lcm(16, align_size)`**；总打包长度统一对齐至 `align_size * 128`。
  - 在 `verl/workers/engine/megatron/transformer_impl.py` 中显式拦截 `qat.enable` 与 `override_transformer_config.fp8` 同时开启（QAT 在 BF16 GEMM 内伪量化权重，而 TE FP8 直接将 GEMM 切换为 FP8）。
- **B200 实测性能与精度（Author-Benchmark，`#8127`）**：
  - **Qwen3-30B-A3B (`4×B200`, DAPO-Math-17k, 10 steps 均值)**：`MXFP8/MXFP8` 单步耗时从 `BF16/BF16` 的 `506 s` 降至 **`452 s`**（加速 **`1.1195x`**，其中 Generation 从 `134 s` 降至 **`102 s`**，缩减 **`23.88%`**；`update_actor` 从 `244 s` 降至 **`221 s`**），且 `rollout_corr/kl` 仅为 **`0.0045`**（显著低于 `FP8-TW/BF16` 的 `0.0058` 和 `FP8-BW/FP8-BW` 的 `0.0090`）。
  - **Qwen3-8B (`2×B200`, DAPO-Math-17k, 32 steps, FP8 KV Cache)**：`MXFP8/MXFP8` 的 `rollout_corr/kl` 稳定在 **`0.0088`**（与 `BF16/BF16` 的 `0.0088` 完全持平，熵保持 `0.338`），而 `FP8-TW/BF16` 的 KL 暴涨至 `0.0297` 且策略熵坍缩至 `0.062`。
- **SGLang & vLLM 分桶暂存与 DeepGEMM 标准化 (`#8306`, `#8308`)**：
  - SGLang 自定义加载器跨所有权重同步 Bucket 暂存 MXFP8 缩放因子与 MXFP4/NVFP4 专家张量，仅在最后一个 Bucket 到达后原地（In-Place）重构内核布局，不替换 CUDA Graph 已捕获的底层 Storage，并显式拒绝 `delta_flush` 及不支持的混合 `linear=MXFP8, moe=MXFP4`。
  - vLLM 通过 `set_default_vllm_quantization_backends` 为非 NVFP4 低精度 Rollout 默认请求 `linear_backend=deep_gemm` 与 `moe_backend=deep_gemm`；当检测到 Hopper 上使用 FP32 块缩放（`weight_block_size == [128, 128]` 且非 `e8m0`）时，自动设置 `VLLM_USE_DEEP_GEMM_E8M0=0` 防止块缩放被舍入为 2 的幂次。
  - **Qwen3.5-35B-A3B (`Megatron EP=8`, `Rollout TP=2`, Blackwell) `k3_kl` 实测对比**：
    - `BF16 Linear + MXFP4 MoE (DeepGEMM)`：开启匹配 QAT 后，SGLang `k3_kl` 从 `0.00571` 降至 **`0.00200`**（降低 **`2.855x`**）；vLLM `k3_kl` 从 `0.00790` 降至 **`0.00140`**（降低 **`5.643x`**）！

### 2.2 veRL Raw Log-Prob 1-ULP 门控 (`#8025`) 与 VeOmni 低精度 ReduceScatter (`#8330`)

- **Raw Log-Prob 1-ULP 检测 (`verl/utils/debug/metrics.py`)**：
  - 在 `calculate_debug_metrics` 中直接比较 `actor_old_log_probs != rollout_old_log_probs`，输出 `training/rollout_logprobs_mismatch_count`（V1 Trainer 按 `sum` 聚合）及 `rollout_logprobs_diff_max/mean`。
  - 算例验证：`lp_a = float32(-0.10000000149011612)` 与 `lp_b = float32(-0.10000000894069672)` 相差 $7.45 \times 10^{-9}$（1 ULP），但 `float32(exp(lp_a)) == float32(exp(lp_b)) == 0.904837429523468`，旧指标 `rollout_probs_diff_max == 0.0` 完全漏检，而新指标精确捕获 `mismatch_count = 1`。
- **VeOmni 低精度 ReduceScatter 负向基准记录 (`#8330`, `fb55b9288560`)**：
  - 暴露 `actor_rollout_ref.actor.veomni.low_precision_reduce_scatter_comm`（节点内 BF16 梯度传输 + FP32 累加）。在 `2×H200 NVL`、428M 稠密 Qwen3 上实测中位数单步耗时：`native-1 = 117.023 ms`、`native-2 = 118.891 ms`，而 `low-1 = 130.853 ms`、`low-2 = 119.163 ms`，且峰值预留显存从 `6.318 GiB` 增至 `6.416 GiB`。证明在小模型/小规模 NVLink 域内额外量化/反量化 Hook 开销抵消了通信节省，默认保持 `False`。

### 2.3 Mooncake TENT Intel XPU PCIe P2P + Level Zero IPC (`#4379`) 与 Dynamo 优化 (`#15535`, `#17067`)

- **Mooncake Intel XPU P2P 与 `pidfd_getfd(2)` 安全 IPC (`mooncake-transfer-engine/tent/src/platform/xpu/xpu_sycl_backend.h`, `xpu_transport.cpp`)**：
  - **同进程 D2D**：`XpuSyclBackend::init()` 仅保留 Level Zero 视图（过滤会触发 `ur_die` 的 OpenCL 重复设备），探测所有有序设备对的 `sycl_ext_oneapi_peer_access`；具备 P2P 时直接调用 `queue::memcpy`，否则以 `8 MB` 分块经 Host 内存弹跳。
  - **跨进程 Level Zero IPC**：段描述符发布 7 元组属性 `"<pid>-<base>-<size>-<handle hex>-<pidns>-<owner uuid hex>-<generation>"`。导入端校验共享 `/proc/self/ns/pid` 与本地设备对 `owner` UUID 的 P2P 可达性后，通过 Linux 系统调用 **`pidfd_open` + `pidfd_getfd(2)`** 抓取导出进程的 `dma-buf` 文件描述符并重写句柄走 fd 导入路径，配合单调递增的 `generation` 彻底解决导出端释放重分配后的地址复用陈旧映射问题！
- **Dynamo KVCR 循环层过滤 (`#15535`) 与 64-Token 批量计数器 (`#17067`)**：
  - `state_agent.rs` 将 `preprocess_residency_with_reason` 移至 `block_size != kv_block_size` 检查之前，使 Mamba/GatedDeltaNet 等非主注意力组（`NonMainAttentionKind` / `NonMainAttentionGroup`）及滞后的 `BlockRemoved`（`UnlearnedGroupIdx`）被正常过滤或放行，不再误触发 `KVCR block size is incompatible` 隔离。
  - `metrics.rs` 在 `RequestMetrics` 中引入 `pending_output_tokens: AtomicU64`（`Relaxed`），每满 **`LOCAL_FLUSH_TOKENS = 64`** 或请求结束时才向全局共享 `output_tokens` 计数器批量刷写一次，将跨核缓存行争用降低 **`64x`**。

---

## 3. 可复现 CPU/Offline 实验与验证

```powershell
python experiments\verl_mxfp8_megatron_sglang_vllm_deepgemm_refit_mooncake_xpu_p2p_dynamo_kvcr\run.py --check
```

| 子系统 / 议题 | 实验配置 | 对比基线 | 优化后结果 | 核心量化结论 | 证据分级 |
|---|---|---:|---:|---:|---|
| veRL MXFP8 RL (`#8127`) | Qwen3-30B-A3B, `4×B200` DAPO | `506 s/step` (`BF16`) | **`452 s/step`** (`MXFP8`) | 单步 **`1.1195x`**，生成 **`-23.88%`** | `author-benchmark` + `own-offline` |
| veRL DeepGEMM Refit (`#8308`) | Qwen3.5-35B-A3B, vLLM MXFP4 MoE | `k3_kl = 0.00790` (无 QAT) | **`k3_kl = 0.00140`** (含 QAT) | 训练-推理 KL 降低 **`5.6429x`** | `author-benchmark` + `own-offline` |
| veRL Raw Log-Prob (`#8025`) | Float32 `-0.10000000149` vs `-0.10000000894` | `probs_diff = 0.0` (漏检) | `raw_diff = 7.45e-9` | 捕获 **1-ULP** 失配 (`count = 1`) | `implementation` + `own-offline` |
| Mooncake XPU P2P IPC (`#4379`) | 同节点 `64 MiB` KV 块跨进程传输 | `3728.27 us` (8MB Host 弹跳) | **`1398.10 us`** (P2P IPC) | **`2.6667x`** (较网络环回 **`8.0x`**) | `implementation` + `own-offline` |
| Dynamo 计数器批刷写 (`#17067`) | 单请求生成 `4096` Tokens | `4096` 次全局原子加 | **`64` 次**全局原子加 | 共享缓存行争用降低 **`64x`** | `implementation` + `own-offline` |

---

## 4. 架构师设计练习与可证伪问题

1. **对齐计算题**：在 Megatron-Core THD 变长打包中，当 `tp_size = 2, cp_size = 2` 时，分别计算常规 `fp8_block` 与 Blackwell `mxfp8` 下的单序列对齐粒度 `per_seq_align` 与总打包对齐粒度 `total_align`。
   - **解答**：因 `cp_size = 2 > 1`，基础对齐粒度 `align_size = 2 * 2 * 2 = 8`。常规 FP8 块量化的 `per_seq_align = lcm(16, 8) = 16`；而 `mxfp8` 的 `per_seq_align = lcm(32, 8) = 32`；两者的总打包对齐粒度均为 `total_align = 8 * 128 = 1024` Tokens。
2. **可证伪问题**：为什么 Mooncake Intel XPU 跨进程 IPC 不能直接复用 Level Zero 默认的隐式句柄缓存，而必须通过 `pidfd_getfd(2)` 获取 `dma-buf` fd 并附加 `generation` 计数器？
   - **验证**：Level Zero 的不透明句柄路径（Opaque-Handle Path）在导入端维护一个按 fd 索引且永不失效的导入缓存；一旦导出进程释放并重新分配显存（复用了相同的虚拟地址、大小与描述符编号），导入端会继续映射到已失效的旧物理缓冲。通过 `pidfd_getfd(2)` 走 fd 导入路径并按单调递增的 `generation` 校验并在 `closeSegment` 时显式 `zeMemCloseIpcHandle`，可严格阻止陈旧映射。
