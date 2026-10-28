# SGLang SM107 2-Shot AllReduce 启发式、DeepSeek-V4.1 Engram DP 自动集合通信选择、Triton 3.9 Swap-AB 缩放广播、分专家 MXFP4 Marlin MoE、vLLM Spec-Decode LoRA Logits 对齐、HiSparse 状态槽位寻址、KVCR 前缀热度对齐与 TRT-LLM B300 MoE 恢复（Phase 120）

> **模块定位**：面向负责大模型推理引擎（SGLang / vLLM / TensorRT-LLM）高吞吐与低延迟调优的解决方案架构师。本文档基于 2026-10-27 至 2026-10-28 合入的 10 个核心提交，深入拆解 Rubin (`sm_107`) NVLink 自定义 AllReduce 阈值、DeepSeek-V4.1 Engram 表在 DP Attention 下的 `all_to_all` 通信量压缩、MXFP4 分专家 Marlin 服务、投机解码 LoRA `lm_head` 对齐、HiSparse 驻留表维度转置与 KVCR 远端缓存 LRU 顺序修复。
> **证据等级**：`implementation`（源码固定 commit 级核验）+ `author-benchmark`（上游 PR 实测数据）+ `own-offline`（本仓库 Python 标准库可复现建模验证）。

---

## 1. 核心架构问题与 10 项上游证据总览

| 序号 | 仓库与 Commit | 日期 | 核心机制与架构变更 | 证据分类 |
| :--- | :--- | :--- | :--- | :--- |
| B1 | `sgl-project/sglang` [`08b9ec6`](https://github.com/sgl-project/sglang/commit/08b9ec6083b419b2e13c431f64da5f5841d75405) | 2026-10-28 | `custom_all_reduce_v2`：SM107 (Rubin VR) TP4/TP8/TP16 启用 `two_shot_push = 16 MB` 并大幅下调 1-shot 切换阈值 | `implementation` |
| B2 | `sgl-project/sglang` [`e35b5bc`](https://github.com/sgl-project/sglang/commit/e35b5bccf0a51fb2ab5ff13dc72f295be9c49621) | 2026-10-28 | `engram.py`：DeepSeek-V4.1 Engram DP 查表新增 `auto` 集合通信选择（整除时优先 `all_to_all`，通信字节降至 $1/\text{TP}$） | `implementation` + `author-benchmark` |
| B3 | `sgl-project/sglang` [`faa56e2`](https://github.com/sgl-project/sglang/commit/faa56e21a2d9493ef3d49428c74a3cb907f0ef00) | 2026-10-28 | `fused_moe_triton_kernels`：修复 Triton 3.9 下 `swap_ab` 标量块缩放因子 `a_scale * b_scale[None, :]` 广播崩溃 | `implementation` |
| B4 | `sgl-project/sglang` [`bb4bb01`](https://github.com/sgl-project/sglang/commit/bb4bb01df4ab0d1e9d17b194151295376a0b3838) | 2026-10-28 | `compressed_tensors_w4a16_mxfp4_moe`：通过 FP4 Marlin (`S_TYPE=kFE8M0fnu`, `group_blocks=2`) 服务分专家存储的 MXFP4 MoE | `implementation` |
| B5 | `sgl-project/sglang` [`2eab19a`](https://github.com/sgl-project/sglang/commit/2eab19a385bf2982a9dc7a4408b068a13bf93bc4) | 2026-10-28 | `deepseek_v4_backend`：当注意力后端 `needs_cpu_seq_lens=True` 时拒绝 `glue_draft_metadata` 融合 | `implementation` |
| B6 | `vllm-project/vllm` [`b71c1d5`](https://github.com/vllm-project/vllm/commit/b71c1d59cfbe1aaeb8be3507bc20545fcbe40072) | 2026-10-28 | `kimi_k3/amd/model.py`：ROCm Kimi-K3 根模型转发 `process_weights_after_loading()` 构建折叠 `q_b_proj / W_UK` 权重 | `implementation` |
| B7 | `vllm-project/vllm` [`06e172b`](https://github.com/vllm-project/vllm/commit/06e172bd14d0c7e5490934968f5611df4a2a0fb4) | 2026-10-28 | `v1/worker/gpu/lora_utils.py`：投机解码 MRV2 下使用 `lora_ids.repeat(num_logits)` 对齐 `lm_head` LoRA 与 Logits 行数 | `implementation` |
| B8 | `vllm-project/vllm` [`66f1d0f`](https://github.com/vllm-project/vllm/commit/66f1d0fd64d1da071fa939909a9c89df1be35891) | 2026-10-28 | `hisparse`：将 `residency` 转置为 `[resident_groups, max_num_reqs, max_num_pages]` 并按 `request_state_indices` 寻址预填充行 | `implementation` |
| B9 | `vllm-project/vllm` [`e3883b7`](https://github.com/vllm-project/vllm/commit/e3883b7444b8ecbd5f0c41d1fc12fb967bc460e8) | 2026-10-27 | `kvcr/manager.py`：通过 `align_sequence` 与 `processed_tokens` 将 KVCR 远端缓存 LRU 热度与请求前缀顺序严格对齐 | `implementation` |
| B10 | `NVIDIA/TensorRT-LLM` [`f4bc8ee`](https://github.com/NVIDIA/TensorRT-LLM/commit/f4bc8ee407803a7682b76d00df4aa7df88e2afc4) | 2026-10-28 | `test_moe_backend.py`：修复 route-map 越界读取后重新启用 B300 (SM103) TRTLLM-Gen 块缩放 MoE 用例并区分外部路由 | `implementation` |

---

## 2. 深度技术剖析与公式推导

### 2.1 SGLang SM107 自定义 AllReduce 阈值与 DeepSeek-V4.1 Engram DP 集合通信（`08b9ec6`, `e35b5bc`）

1. **SM107 (Rubin VR) 2-Shot Push 启发式（`08b9ec6`）**：
   - 在 `_sm107_configs` 中，为 TP4、TP8、TP16 全面开启 `two_shot_push = 16 MB`，并将 `1shot_push` / `1shot_pull` 的上限阈值大幅下调，使中等载荷提前进入带宽利用率更高的 `2shot_push`：
     - **TP4**：Graph 从 `6.938 MB` 降至 `1.000 MB`，Eager 从 `9.812 MB` 降至 `1.000 MB`；
     - **TP8**：Graph/Eager 均从 `2.438 MB` 降至 `384 KB`（下调 **6.5013x**）；
     - **TP16**：Graph/Eager 均从 `832 KB` 降至 `256 KB`（下调 **3.25x**）。
2. **DeepSeek-V4.1 Engram DP 查表 `auto` 集合通信（`e35b5bc`）**：
   - DeepSeek-V4.1 的 Engram 表具有 $C = \text{engram\_n\_heads} \times (\text{engram\_max\_ngram\_size} - 1) = 24$ 个 Hash 列，词表行按 TP 组切分。在 DP Attention 下，设全组总 Token 数为 $N$，嵌入维度为 $D$：
     - `all_reduce` / `reduce_scatter`：每个 Rank 对全部 $N$ 个 Token 查找全量 $C=24$ 列中属于本 Rank 分片的行（未命中置零），随后对形状 `[N, C, D]` 执行归约，`reduce_scatter` 每 Rank 跨卡发送字节数为：
       $$B_{\text{RS}} = \frac{\text{TP}-1}{\text{TP}} \cdot N \cdot C \cdot D \cdot \text{sizeof(dtype)}$$
     - `all_to_all`：当 $C \bmod \text{TP} = 0$（即 $\text{TP} \in \{2, 3, 4, 6, 8, 12\}$）时，每个 Rank 的整片恰好对应 $k = C / \text{TP}$ 个 Hash 列。因此每个 Rank 只需查找自己持有的 $k$ 列，再通过 `all_to_all_single` 将每个 DP Rank 所属 Token 的结果发回：
       $$B_{\text{A2A}} = \frac{\text{TP}-1}{\text{TP}} \cdot N \cdot \frac{C}{\text{TP}} \cdot D \cdot \text{sizeof(dtype)} = \frac{1}{\text{TP}} B_{\text{RS}}$$
   - **上游实测结果（`author-benchmark`）**：在 GB300、B300 与 H200（DP 2–8，Host Table，每 Rank `512–65,536` Tokens）共 144 个形状中，`all_to_all` 在 **121 个形状**上夺冠，在 $\ge 8,192$ Tokens/Rank 时提速 **6%–60%**。

### 2.2 SGLang 分专家 MXFP4 Marlin MoE 与 Triton 3.9 Swap-AB 修复（`faa56e2`, `bb4bb01`, `2eab19a`）

- **Triton 3.9 `swap_ab` 标量块缩放修复（`faa56e2`）**：Triton 3.9 禁止对 0D 标量张量执行 `a_scale[:, None]` 切片。当 `swap_ab=True` 且 `BLOCK_SIZE_N <= group_n` 时，`a_scale` 是标量权重块缩放因子，`b_scale` 为 1D 激活缩放因子，现改为 `accumulator += tl.dot(a, b) * (a_scale * b_scale[None, :])`。
- **分专家 MXFP4 Marlin MoE（`bb4bb01`）**：新增 `CompressedTensorsW4A16Mxfp4MoE`，支持以 `experts.{e}.{gate,up,down}_proj.*` 分专家格式保存的 `compressed-tensors` MXFP4 权重。在 C++ Marlin 模板中显式引入 `s_type_id`：NVFP4 使用 `kFE4M3fn`（`group_blocks = 1`，允许 `dequant_skip_flop = true`），而 MXFP4 使用无符号 2 的幂次指数 `kFE8M0fnu`（`group_size = 32, group_blocks = 2`，静态断言禁用 `dequant_skip_flop = false`）。
- **DSV4 Draft Metadata Glue 防护（`2eab19a`）**：当注意力后端需要 CPU 侧序列长度（`self.needs_cpu_seq_lens=True`，如 FlashAttention-3 / FlashMLA 动态元数据构建）时，`glue_draft_metadata` 主动拒绝将步进循环融合进前向内核，防止 `seq_lens_cpu` 滞后于 GPU 侧自增的 `seq_lens`。

### 2.3 vLLM Spec-Decode LoRA 对齐、HiSparse 状态槽寻址与 KVCR 前缀热度（`b71c1d5`, `06e172b`, `66f1d0f`, `e3883b7`, `f4bc8ee`）

1. **投机解码 MRV2 `lm_head` LoRA 对齐（`06e172b`）**：在投机解码下，每个请求产出 $1 + \text{num\_drafts}$ 行 Logits（由 `num_logits = np.diff(cu_num_logits_np)` 给出）。原代码将 `prompt_lora_mapping` 设为每请求 1 个 ID，导致 `lm_head` Punica 内核将第 1 个请求的 Draft 行误用后续请求的 LoRA 权重。修复后改为 `tuple(lora_ids.repeat(num_logits))`。
2. **HiSparse 驻留表转置与状态槽映射（`66f1d0f`）**：
   - 将 `residency` 张量从 `[max_num_reqs, num_groups, max_num_pages]` 转置为 `[num_groups, max_num_reqs, max_num_pages]`，使 `residency[resident_source_index]` 直接成为连续行步长的 2D 表，消除每个解码步的 `index_select` 拷贝；
   - 在 `sparse_mla_index_remap_kernel` 中传入 `idx_mapping = request_state_indices`，将批次内请求行号 `req` 映射为真实状态槽位 `req = tl.maximum(tl.load(idx_mapping_ptr + req), 0)`，修复请求重排后的跨请求 KV 页污染。
3. **KVCR 远端缓存前缀热度对齐（`e3883b7`）**：
   - 正常 LRU Touch 顺序使请求前缀块最早被 Touch、尾部块最晚被 Touch，导致重压力下前缀块先于尾部块被淘汰，破坏前缀缓存连续性；
   - 修复后在异步 Load/Store 完成时按前缀顺序调用 `align_sequence([k for k, pos in positions.items() if pos <= end_token])`，并在请求结束时按已确认提交 Token 上界：
     $$\text{processed\_tokens} = \min\big(\max(0, \text{computed\_tokens} - \text{in\_flight}), \max(\text{prompt\_tokens}, \text{num\_tokens} - 1)\big)$$
     执行带当前时间戳的 `align_sequence(..., use_current_time=True)`。

---

## 3. 本仓库离线建模验证（`own-offline`）

对应实验目录：`experiments/sglang_sm107_2shot_engram_dp_mxfp4_vllm_hisparse_kvcr_trtllm_phase120/`

- **运行命令**：`python experiments/sglang_sm107_2shot_engram_dp_mxfp4_vllm_hisparse_kvcr_trtllm_phase120/run.py --check`
- **关键输出校验**：
  - `sm107_2shot_allreduce.tp8_1shot_threshold_reduction_ratio = 6.5013`（`2.438 MB -> 384 KB`）；
  - `dsv41_engram_dp_collective.divisible_tp_sizes = [2, 3, 4, 6, 8, 12]`，TP=8 时 `byte_ratio_vs_reduce_scatter = 0.125`；
  - `vllm_and_trtllm_fixes.spec_lora_mapping_matches_logits_rows = true`（8 行 Logits 精确对应 8 个 LoRA ID）；
  - `vllm_and_trtllm_fixes.kvcr_processed_tokens = 112`，排除了未提交的 in-flight 块 `b3 (pos=128)`。
