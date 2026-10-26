# SGLang DSV4/V4.1 Prefill Graph 桥接与候选表、Kimi-K3 TGV BF16 GEMM、Wan 扩散直写 Ulysses 分块、vLLM MRV2 内存释放与 DecodeBenchConnector（Phase 96）

> **证据分类**：`implementation`（固定上游源码提交）+ `own-offline`（本项目纯 Python 3.12 标准库离线确定性验证，非 GPU 硬件实测 profiling）。

## 1. 核心上游源码实证（10 项零重叠一手提交）

| Source ID | 仓库 / 提交 | 关键实现路径与架构机制 |
|---|---|---|
| `sglang-kernels-beam-model-support-pr45254-2026` | `sgl-project/sglang` `66f79f1a90` (PR #45254) | 新增 `BeamForCausalLM` 融合 MoE Triton kernel 配置，并在 DeepSeek-V4 sliding-window attention (`SWA`) FP8 KV 量化前先向 BF16 舍入以对齐 QAT 与 DSA prefill-CP 路径。 |
| `sglang-dsv4-replay-tail-caller-prefill-graph-bridges-pr46516-2026` | `sgl-project/sglang` `3d2a781b3e` (PR #46516) | DeepSeek-V4 将 replay-tail 输出通过 `_write_bridge` 直接写入调用方持有的 `_bridge_buffers`，保留旧捕获桶的小容量缓冲区生命期，避免已捕获 prefill CUDA graph 引用悬空指针。 |
| `sglang-kimi-k3-tgv-bf16-gemm-narrow-mla-gate-pr46468-2026` | `sgl-project/sglang` `a0618d5e65` (PR #46468) | Kimi-K3 将 DSpark draft `(7168, 1792)`、`(7168, 512)` 与稠密 MLP `(8448, 7168)` BF16 GEMM 路由至 CuTeDSL TGV 内核，并将 `mla_output_gate_kernel` 向量宽度从 `kVecN = 8` 收窄至 `kVecN = 4`（`AlignedVector<bf16_t, 4>`，8 字节加载）。 |
| `sglang-dsv41-candidate-table-caller-prefill-graph-buffers-pr46399-2026` | `sgl-project/sglang` `e8019014b0` (PR #46399) | DeepSeek-V4.1 稀疏 indexer 将候选表直接写入调用方持有的 `_graph_tables`（最新桶排在列表首位，旧桶保留供已捕获 graph 使用），移除 `is_capturing()` 分支。 |
| `sglang-dsv41-align-bf16-prefill-indexer-score-width-pr45794-2026` | `sgl-project/sglang` `65f19fce48` (PR #45794) | 对齐 DeepSeek-V4.1 BF16 prefill indexer 分数宽度，并在 Context Parallelism (`CP`) 下当 `inputs.req_rows` 缺省时由 `rows_per_request_device` 现场重构 rank-local 请求行索引。 |
| `sglang-diffusion-wan-normed-qk-direct-ulysses-exchange-pr46247-2026` | `sgl-project/sglang` `cf66f7ed73` (PR #46247) | Wan 视频扩散模型通过 `fused_rmsnorm_interleaved_rope_into_blocks(..., blocked=True)` 与 `fill_all` 将归一化+RoPE 后的 Q/K 直接按 `[M, block_cols]` 写入 Ulysses IPC all-to-all 发送/接收槽位。 |
| `vllm-preserve-fp16-legacy-fused-moe-lora-pr59931-2026` | `vllm-project/vllm` `2f1cb151fd` (PR #59931) | 修复 `_fused_moe_lora_kernel` 在合入 TMA 描述符路径后将 `tl.load` 分支硬编码转为 `bfloat16` 的回归，统一将两侧 `tl.dot` 操作数转为匹配的输入 `dtype`（保留 `float16` 精度）。 |
| `vllm-multimodal-release-consumed-mrv2-cpu-inputs-pr57368-2026` | `vllm-project/vllm` `373614f66d` (PR #57368) | 在 Model Runner V2 (`MRV2`) 多模态编码器消费完 CPU 特征张量后立即释放引用，消除长会话下主机内存持续驻留泄漏。 |
| `vllm-kv-connector-decode-bench-report-all-groups-loaded-pr62242-2026` | `vllm-project/vllm` `beeed65c10` (PR #62242) | `DecodeBenchConnector` 实现 `get_loaded_kv_cache_group_ids()` 并返回完整的 `self._kv_cache_config.transfer_group_ids`，修复混合全注意力+滑动窗口模型在合成 decode 压测中的跨组加载上报缺失。 |
| `trtllm-unify-multimodal-request-data-item-metadata-pr20391-2026` | `NVIDIA/TensorRT-LLM` `9ffa6f4395` (PR #20391) | 统一 TensorRT-LLM 多模态预处理与 PyTorch 执行器运行时的 `MultimodalParams` 及逐条目 item metadata 契约。 |

## 2. 数学建模与架构师计算练习

### 2.1 Wan 扩散模型直写 Ulysses 交换缓冲区的 HBM 流量节省
设序列长度 $M = 4096$，头数 $H = 32$，头维 $D = 128$，Ulysses 并行度 $U = 4$，每个 BF16 张量大小为：
$$S_{QK} = 4096 \times 32 \times 128 \times 2\text{ B} = 32\text{ MiB}$$
- **基线路径**：`RMSNorm+RoPE` 读 Q/K（$2 S_{QK}$）并写完整中间张量 `mY`（$2 S_{QK}$），随后 Ulysses 打包再读 `mY`（$2 S_{QK}$）并写分块缓冲区（$2 S_{QK}$），总计 $8 S_{QK} = 256\text{ MiB}$。
- **直写分块路径（PR #46247）**：`fused_rmsnorm_interleaved_rope_into_blocks` 读 Q/K（$2 S_{QK}$）并直接写入 `[M, block_cols]`（`block_cols = 1024`）的 Ulysses 交换槽位（$2 S_{QK}$），总计 $4 S_{QK} = 128\text{ MiB}$，消除全部中间完整张量落盘（`intermediate_full_qk_writes = 0`），**HBM 流量降低 $50.0\%$**。

### 2.2 vLLM MRV2 多模态主机内存释放与ハイブリッド KV Group 上报
对于并发 `active_mm_requests = 64`、单请求多模态特征 `8.0 MiB` 的长上下文工作负载，编码后立即释放 MRV2 CPU 张量可将残留主机内存从 `512.0 MiB` 降至 `0.0 MiB`。

## 3. 可复现离线实验与验证

运行命令：
```bash
python experiments/sglang_beam_dsv4_bridges_kimi_tgv_dsv41_table_wan_ulysses_vllm_lora_mrv2_trtllm_phase96/run.py --check
```
实验产物 `summary.json` 已验证：
- `stale_graph_pointer_reads == 0`，`tgv_routed_shapes_count == 6`，`new_mla_gate_kvec_n == 4`（`8` 字节对齐向量）；
- `baseline_qk_hbm_mb == 256.0`，`direct_blocks_qk_hbm_mb == 128.0`，`hbm_traffic_saving_fraction == 0.5`；
- `unreleased_mrv2_cpu_mb == 512.0 -> released_after_encoder_cpu_mb == 0.0`，`all_groups_loaded == True`。
