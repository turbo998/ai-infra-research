# SGLang BCG 共享缓冲与调用方预分配、Ascend NPU Bitmask、vLLM Triton 自适应验证草稿预算、Batch-Invariant WNA16、Jamba 前缀锚定流式工具解析、Mooncake 降序块合并与 TensorRT-LLM `@routes` 注册表（Phase 98）

> **English Version**: [sglang-bcg-nemotron-holder-lse-npu-bitmask-vllm-triton-av-wna16-jamba-mooncake-trtllm-phase98.en.md](sglang-bcg-nemotron-holder-lse-npu-bitmask-vllm-triton-av-wna16-jamba-mooncake-trtllm-phase98.en.md)
> **优先级**: `107` · **问题 ID**: `sglang-bcg-nemotron-holder-lse-npu-bitmask-vllm-triton-av-wna16-jamba-mooncake-trtllm-phase98` · **证据类型**: `implementation` + `own-offline`

---

## 1. 核心架构问题与结论速览

在分段/可中断 CUDA Graph（Breakable CUDA Graph, `bcg`）、投机解码自适应验证（Adaptive Verification, AV）、确定性批不变 MoE 推理、流式工具调用增量切片以及跨节点 Mooncake KVConnector RDMA 传输中，Phase 98 通过 10 个零重叠提交确立了以下关键设计范式：

1. **SGLang 可中断 CUDA Graph (`bcg`) 的全链路调用方预分配缓冲契约 (`sglang#46553`, `d4939cc267`; `#46551`, `5aeb6dfdb6`; `#46535`, `2c250563ae`)**：
   - 在 `bcg.py` 的调试模式（`_debug_mode`）下，Eager Break 内部直接把 `self.debug_Forward_fn()` 的输出拷贝进 `self._shared_output_buffer`，消除每段图边界的临时输出分配；
   - 在 `nemotron_h.py` 中，`NemotronHMoE` 直接从调用方传入的 `shared_expert_output` holder 读取 Prefill 共享专家结果，不再从子函数返回独立引用；
   - 在 `radix_attention.py` 中，调用方在进入 `_eager_attention` 前预分配 `lse = q.new_empty((q.shape[0], self.tp_q_head_num), dtype=torch.float32)`，由被调函数原位写入 LSE，使 16 次 Eager Break 的额外张量分配从 `48` 次降至 `0` 次。
2. **SGLang Ascend NPU 精简通信器与原生 `apply_token_bitmask` 约束解码 (`sglang#45950`, `3b058f28c2`; `#36074`, `fa622d311d`)**：移除 `communicators.py` 中冗余的 71 行 `NpuCommunicator` 包装并完善 `aiter` IPC 缓冲池在 CUDA Graph 捕获下的分块规约说明；同时在 `llguidance_backend.py` 中接入 `torch.ops.npu.apply_token_bitmask(logits, vocab_mask)`（`151,936` 词表压缩为 `4,748` 个 `int32` 位图字）。
3. **vLLM 单核 Triton 投机解码自适应验证（AV）草稿预算分配 (`vllm#63421`, `41f5146b5b`)**：用单个融合 Triton 核（`_count_survival_at_least` + `assign_draft_token_budget`）替代原先依赖 `torch.compile` 的 `select_draft_token_budgets` 图编译路径，省去约 `4.8 s` Inductor 编译预热并在单趟扫描中完成累积存活率阈值计数。
4. **vLLM `VLLM_BATCH_INVARIANT=1` 下的 WNA16 MoE Triton 确定性派发护栏 (`vllm#61038`, `1fd6337e22`)**：在 `should_moe_wna16_use_cuda()` 中优先检查 `vllm_is_batch_invariant()`，当开启批不变模式时强制返回 `False`，避免 WNA16 切换到非确定性的 CUDA 核。
5. **vLLM Jamba 流式工具参数公共前缀锚定与 Mooncake 降序块对归一化合并 (`vllm#55669`, `3f45273050`; `#63142`, `2841f08b4a`)**：
   - Jamba 流式解析器使用 `find_common_prefix(prev_args, cur_args)` 锚定已发送参数前缀，避免函数名与首段参数同帧到达时漏发后续键值（完整还原 `{"city": "Dallas", "state": "TX"}`）；
   - Mooncake KVConnector 新增 `_normalize_paired_block_order(src, dst)`，在按块大小展开前先按 `(local_block_id, remote_block_id)` 升序重排，将原本因降序分配（如 `[9, 8, ..., 2]`）而被拆成 `8` 个碎片描述符的传输合并为 `1` 个连续 RDMA 描述符（描述符数量减少 `8.0x`）。
6. **TensorRT-LLM `modeling_v2` 声明式 `@routes` 架构路由索引 (`TensorRT-LLM#20789`, `158252825d`)**：通过 `_router_index.py` 中的 `@routes("GptOssForCausalLM", ...)` 装饰器实现单一事实来源的架构路由分发。

---

## 2. 一手源码证据与定量指标对比

| 编号 | 证据 ID / 仓库提交 | 关键实现路径与硬件契约 | 核心定量收益 / 正确性不变量 |
| :--- | :--- | :--- | :--- |
| 1 | `sglang-bcg-write-debug-mode-forward-output-buffer-inside-46553-d4939cc-2026` (`d4939cc267`) | `bcg.py`: Eager Break 内原位写入 `_shared_output_buffer` | 消除 BCG 调试分段边界的临时输出张量分配 |
| 2 | `sglang-bcg-read-nemotron-h-prefill-shared-expert-output-46551-5aeb6df-2026` (`5aeb6dfdb6`) | `nemotron_h.py`: 从调用方 holder 直接读取 Prefill 共享专家输出 | 消除子函数返回新张量引用导致的图捕获断裂 |
| 3 | `sglang-bcg-write-radixattention-eager-break-lse-into-cal-46535-2c25056-2026` (`2c250563ae`) | `radix_attention.py`: 调用方预分配 `(tokens, tp_q_head_num)` FP32 LSE | 16 次 Eager Break 额外分配从 `48 -> 0` 次 |
| 4 | `sglang-cleanup-remove-redundant-npucommunicator-and-clar-45950-3b058f2-2026` (`3b058f28c2`) | `communicators.py` + `custom_all_reduce.py`: 移除冗余 `NpuCommunicator` | 净删 `71` 行冗余代码并明确 IPC staging 图捕获契约 |
| 5 | `sglang-npu-add-ascend-npu-bitmask-path-in-llguidance-bac-36074-fa622d3-2026` (`fa622d311d`) | `llguidance_backend.py`: `torch.ops.npu.apply_token_bitmask` | `151,936` 词表仅需 `4,748` 个 `int32` 字即可在 NPU 原生掩码 |
| 6 | `vllm-perf-spec-decode-replace-torch-compile-in-av-draf-63421-41f5146-2026` (`41f5146b5b`) | `rejection_sampler.py`: Triton `_count_survival_at_least` 核 | 消除 `~4.8 s` `torch.compile` 预热，单趟计算草稿 Token 预算 |
| 7 | `vllm-batch-invariance-keep-legacy-wna16-moe-on-triton-61038-1fd6337-2026` (`1fd6337e22`) | `fused_moe.py`: `if vllm_is_batch_invariant(): return False` | `VLLM_BATCH_INVARIANT=1` 下 100% 保持 Triton 确定性 WNA16 路由 |
| 8 | `vllm-bugfix-anchor-jamba-streamed-tool-arguments-to-se-55669-3f45273-2026` (`3f45273050`) | `jamba_tool_parser.py`: `find_common_prefix` 锚定已发送参数前缀 | 100% 恢复跨 chunk 流式 JSON 参数（零字段截断） |
| 9 | `vllm-mooncake-coalesce-descending-block-ids-in-kvconne-63142-2841f08-2026` (`2841f08b4a`) | `mooncake_connector.py`: `_normalize_paired_block_order` 升序重排 | 8 个降序连续块从 `8` 个碎片合并为 `1` 个 RDMA 描述符 (`8.0x` 缩减) |
| 10 | `tensorrt-llm-feat-modeling-v2-register-architecture-routing-wi-20789-1582528-2026` (`158252825d`) | `_router_index.py`: `@routes(...)` 装饰器注册架构路由函数 | 统一 `modeling_v2` 架构家族路由表（`0` 重复手工字典） |

---

## 3. 架构师设计练习与可证伪问题

1. **计算练习（Mooncake 块分配顺序与 RDMA SGE/WR 聚合）**：当块分配器采用 LIFO 栈弹出空闲块时，连续分配的 8 个 KV 块 ID 为 `[9, 8, 7, 6, 5, 4, 3, 2]`，远端对应块为 `[19, 18, 17, 16, 15, 14, 13, 12]`。为什么仅检查 `src[i] == src[i-1] + 1` 的 `group_concurrent_contiguous` 会生成 8 个独立 RDMA 传输请求？先对 `(src, dst)` 按 `src` 升序配对排序后为什么能无损合并为 1 个连续传输？
2. **可证伪假设**：在启用 `VLLM_BATCH_INVARIANT=1` 时，若允许小 Batch（如 `B <= 16`）自动切换到 `moe_wna16` CUDA 核以换取更低延迟，是否仍能满足严格 Batch Invariance？源码与数值实验证明：**不能**，因为 CUDA WNA16 核与 Triton WNA16 核的分块累加顺序不同，会导致跨 Batch Size 的 Logits 出现位级偏差。

---

## 4. 配套离线可复现实验

- **实验路径**: `experiments/sglang_bcg_nemotron_holder_lse_npu_bitmask_vllm_triton_av_wna16_jamba_mooncake_trtllm_phase98/run.py`
- **校验命令**: `python experiments\sglang_bcg_nemotron_holder_lse_npu_bitmask_vllm_triton_av_wna16_jamba_mooncake_trtllm_phase98\run.py --check`
- **单测覆盖**: `tests/test_phase98_experiments.py`
