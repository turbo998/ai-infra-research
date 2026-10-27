# vLLM Kimi-K3 KV-Balanced FP8 MLA Decode、SGLang 无草稿目标模型辅助隐状态捕获、`persistent_topk` 屏障同步与 TRT-LLM EagerMoeWorkspace 分析（Phase 101）

- **日期 / 阶段**：2026-10-27 · Phase 101 (`priority: 110`)
- **组件归属**：`inference`
- **关联问题 ID**：`vllm-kimi-k3-kv-balanced-fp8-mla-sglang-aux-hidden-capture-phase101`
- **关联证据 ID**：`claim-vllm-kimi-k3-kv-balanced-mla-sglang-aux-hidden-phase101`
- **可复现实验套件**：[`experiments/vllm_kv_balanced_mla_sglang_aux_phase101/`](../../experiments/vllm_kv_balanced_mla_sglang_aux_phase101/run.py)

---

## 1. 架构背景与核心问题

在 Agentic 长短上下文混合解码与投机/离线蒸馏推理栈中，核心性能与正确性瓶颈集中在以下四个方面：

1. **Agentic 偏斜 KV 长度下的 Split-KV 负载失衡与主机同步开销**：在 Kimi-K3 等 MLA（Multi-head Latent Attention，$512 + 64 = 576$ FP8 值/token）模型服务 Agentic 负载时，同一 Batch 内既有 128K 长上下文工具调用，也有 1K 短轮次对话。静态固定 Split-KV 会造成长请求阻塞整个波次或短请求过度切分；若在 CPU 侧规划切分又会破坏 CUDA Graph 捕获。
2. **离线草稿训练/特征提取被迫加载完整投机草稿模型**：当用户仅需在目标模型上按 EAGLE3、DFLASH 或 DSPARK 规范捕获多层辅助隐状态（auxiliary hidden states）时，传统 `--speculative-algorithm` 会加载草稿模型权重并预留草稿 KV Cache，浪费数 GiB 显存。
3. **`persistent_topk` 基数排序跨线程块屏障的数据竞争**：在 vLLM `csrc/persistent_topk.cuh` 的 `radix_topk` 中，各线程并行清零 `next_hist[i] = 0` 后，线程 `tx == 0` 直接执行 `red_release(&state->arrival_counter, 1)`，缺失 CTA 内 `__syncthreads()` 导致其他 Warp 的清零尚未落地即触发下一阶段到达计数。
4. **MoE 动态工作区在显存压力下的配额回收与 LoRA 权重归属校验**：TensorRT-LLM `EagerMoeWorkspace` 在显存紧张时需触发缓存分配器释放与重试，且请求级 PEFT 缓存需携带 `py_lora_ckpt_source` 以对齐服务端权限策略。

---

## 2. 一手源码实证分析（12 个零重叠上游提交）

### 2.1 vLLM `#63638` (`6597ceafbc`)：Kimi-K3 Opt-In KV-Balanced FP8 MLA Decode (`KIMI_K3_KV_BALANCED_MLA`)

`vllm/models/kimi_k3/nvidia/kv_balanced_mla/{backend.py,core.py,decode.py,helpers.py}` 引入了面向 SM100/SM103/SM107 的设备端动态规划 FP8 MLA 解码后端：

- **缓存布局与寄存器/概率缩放参数**：
  - 每 token 缓存 `LATENT = 512` + `ROPE = 64` = **576 个 FP8 值**（页大小 32 或 64），查询行打包为 `ROWS = 128`（2-CTA M128 簇），每 MMA 分块 `KV_TILE = 128` tokens，单请求最大切分数 `CAPACITY = 8`，单分块最小粒度 `MIN_SPLIT_TOKENS = 1024`（`min_tiles = 8`），归并开销 `OVERHEAD_TILES = 6`。
  - 针对默认寄存器分配 `(208, 256, 40)` 在 MMA Warp 产生寄存器溢出（spill）并阻塞 Tensor Core 描述符算术的问题，重平衡为 **`REG_SPLIT = (200, 240, 64)`**（Softmax / Correction / MMA+Load Warpgroups）。
  - 概率预缩放采用 **`P_SCALE = (2.0, 6.0)`**（对数域 skip-correction 阈值 $2^2$、预缩放 $2^6$），使 $P$ 在转换至 E4M3（上限 448，最大缩放值 $2^{2+6}=256 \le 448$）时保留低至行最大值 $2^{-(9+6-2)} = 2^{-13}$ 的概率而不被冲零。
- **设备端 `[W=1..8, B]` 二分搜索工作列表规划器 (`_plan_work_list_kernel`)**：
  - 对每个候选波次数 $W \in \{1, \dots, 8\}$（预算 $W \times \text{clusters}$），通过二分搜索寻找满足总簇数 $\sum_b n_b \cdot q_{\text{tiles}} \le W \cdot \text{clusters}$ 的最小分块粒度 $c \ge 8$，计算每请求切分数 $n_b = \min(\lceil T_b / c \rceil, 8)$ 与实际单切分长度 $\text{per}_b = \lceil T_b / n_b \rceil$。
  - 选择使估计周期成本最小的最优波次 $W^*$：
    $$\text{estimate}(W) = \left\lceil \frac{\sum_b n_b \cdot q_{\text{tiles}}}{\text{clusters}} \right\rceil \times \left(\max_b \text{per}_b + \text{OVERHEAD\_TILES}\right)$$
  - **Split-Major L2 共享编码**：工作列表项编码为 `((q_tile * num_reqs + req_idx) << 8) | split`，使同一请求不同 `q_tile` 在同一 `split` 上相邻排列，通过 L2 Cache 共享该切分的 KV 页。
  - **跨层单次规划复用与 CUDA Graph 兼容**：每次前向仅由首个 MLA 层规划 `work` 列表（缓存在 `_PLANNED` 字典中），后续所有 MLA 层直接复用，全程无主机读取。

### 2.2 SGLang `#43449` (`e732f5d41e`) 与 `#46340` (`eb8c2a40ef`)、`#46254`、`#46570`、`#39774`

- **无草稿目标模型辅助隐状态捕获 (`#43449`)**：新增 `--enable-return-aux-hidden-states`。开启后，`--speculative-algorithm`（`EAGLE3`、`DFLASH` 或 `DSPARK`）仅用于指定目标模型辅助层捕获约定（保存在 `aux_hidden_state_capture`），随后 `handle_aux_hidden_state_capture` 钩子将 `speculative_algorithm` 重置为 `None`，**不加载任何草稿模型权重、不预留草稿 KV Cache、不启动投机栈**，并将 `--return-hidden-states-mode` 默认设为 `full`。
- **AMD MiniMax-M3 稀疏索引 Top-K 频率修正 (`#46340`)**：将 ROCm 下 `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ` 默认值从 `2` 修正为 `1`（禁用跨层共享过期 Top-K 索引），消除跳过层（skip layers）复用陈旧 KV 块导致的精度退化。
- **Diffusion 状态恢复测试隔离 (`#46254`)、HiCache L3 `session_ids` 桩兼容 (`#46570`) 与 NPU `--return-output-ids` 端到端验证 (`#39774`)**。

### 2.3 vLLM `#63475` (`3090266d88`)、`#63828` (`70d67f19dc`)、`#45188`、`#55699` 与 TensorRT-LLM `#20947` / `#20666`

- **`persistent_topk` 基数屏障同步 (`#63475`)**：在 `csrc/persistent_topk.cuh` 的 `next_hist[i] = 0` 循环后、线程 `tx == 0` 发射 `red_release(&state->arrival_counter, 1)` 前插入 `__syncthreads()`，彻底消除 Warp 1..7 滞后清零导致的跨 CTA 直方图数据竞争。
- **ROCm 检查点页缓存释放 (`#63828`)、构建 `MAX_JOBS` 解析鲁棒性 (`#45188`) 与 HTTP 退出时停止 SSL 证书监听协程 (`#55699`)**。
- **TensorRT-LLM `EagerMoeWorkspace` 显存压力回收重试与 LoRA 归属校验 (`#20947`, `758c7b5ba4`)**：验证在原生分配器配额限制下释放 scratch 后的重试恢复路径，并在 PEFT 请求对象中透传 `py_lora_ckpt_source`。

---

## 3. 确定性实验结果总结

运行 `python experiments/vllm_kv_balanced_mla_sglang_aux_phase101/run.py --check` 得到以下密封实验指标：

| 评估维度 | 核心指标 | 实测 / 仿真结果 |
|---|---|---:|
| Kimi-K3 KV-Balanced MLA 规划器 | 16 请求偏斜 Agentic 负载（1K–128K）最优估计周期 vs 固定 4-Split | **`134` vs `524` 分块周期 (`3.9104x` 加速)** |
| Kimi-K3 KV-Balanced MLA 容量 | $H=128, q=2$ (`q_tiles=2`) 下 Int32 最大可寻址请求数 | **`1,024` 个并发请求** |
| SGLang `--enable-return-aux-hidden-states` | EAGLE3 / DFLASH 规范下免加载草稿模型节省显存 | **`3,920.19 MiB` / `5,481.23 MiB`** |
| vLLM `persistent_topk` 屏障修复 | 插入 `__syncthreads()` 前后未同步 Warp 暴露率 | **`87.5%` (7/8 Warps) $\to$ `0.0%`** |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **Int32 部分和寻址上限推导**：在 `vllm/models/kimi_k3/nvidia/kv_balanced_mla/core.py` 中，`MAX_TILES = (1 << 31) // (ROWS * CAPACITY * LATENT * 2) = 2048`。请计算当 $H = 128$、投机验证步长 $q = 4$（如 DSpark draft verify）时，`query_tiles(128, 4)` 与单次内核可寻址的最大请求数 `max_planned_requests(128, 4)`。
2. **E4M3 概率预缩放动态范围验证**：解释为何 `P_SCALE = (2.0, 6.0)` 要求 $2^{\text{threshold} + \text{prescale}} \le 448$，若将 `prescale` 从 `6.0` 提升至 `7.0` 会在在线 Softmax 滞后最大值为 $2^2$ 时引发什么数值故障？
