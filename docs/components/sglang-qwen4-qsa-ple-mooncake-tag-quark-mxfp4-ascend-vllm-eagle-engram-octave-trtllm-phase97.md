# SGLang Qwen4 QSA/PLE 预分配桥接、Mooncake 会话标签、Quark MXFP4 共享专家融合、Ascend 零 D2H DSA、vLLM 终态前缀快照、EAGLE 前瞻块哈希、DSv4.1 Peer-HBM Engram、ROCm Octave 3-Bit KV 与 TRT-LLM 确定性 mHC（Phase 97）

> **English Version**: [sglang-qwen4-qsa-ple-mooncake-tag-quark-mxfp4-ascend-vllm-eagle-engram-octave-trtllm-phase97.en.md](sglang-qwen4-qsa-ple-mooncake-tag-quark-mxfp4-ascend-vllm-eagle-engram-octave-trtllm-phase97.en.md)
> **优先级**: `106` · **问题 ID**: `sglang-qwen4-qsa-ple-mooncake-tag-quark-mxfp4-ascend-vllm-eagle-engram-octave-trtllm-phase97` · **证据类型**: `implementation` + `own-offline`

---

## 1. 核心架构问题与结论速览

围绕大模型推理引擎（SGLang、vLLM、TensorRT-LLM）在图捕获兼容性、投机解码前缀缓存、3-bit KV 压缩、跨卡 Engram 表直接读取与确定性执行上的演进，Phase 97 提炼出 10 项关键机制：

1. **Qwen4 QSA Top-K 与 PLE 分段桥接的调用方预分配输出 (`sglang#46543`, `5e420a1258`)**：在 Breakable CUDA Graph 分段捕获下，`qwen4_exp` 将 QSA Top-K 索引与 Per-Layer Embedding (PLE) 批次断点直接写入调用方持有的预分配张量（`_qsa_prefill_topk_bridges`），避免跨图段返回新分配临时张量导致的内存池失效。
2. **`is_cpu()` 环境变量短路消除 `torch.compile(fullgraph=True)` 图断裂 (`sglang#46640`, `5a1f174700`)**：此前 `is_cpu()` 先调用 `is_host_cpu_x86()`（触发 `platform.machine()` / `os.uname()`），导致 Dynamo 在追踪 `RMSNorm.forward_native` 时中断全图编译。现改为先判断 `os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1"`，GPU 路径系统调用次数降至 `0`。
3. **Mooncake 与 UMBP KV Offload 会话级标签透传 (`sglang#44611`, `5987ec82ed`)**：`MooncakeDirectLinker` 与 `UMBPDirectLinker` 通过 `_replicate_config_supports("session_id")` 自省，将 `HiCacheStorageExtraInfo.session_id` 写入 `ReplicateConfig`，支撑按会话粘性调度与驱逐。
4. **AMD Quark MXFP4 Qwen3.5/3.8 共享专家在线量化融合 (`sglang#41899`, `0cd2a63b36`)**：Quark 导出的 Qwen3.5/3.8 MXFP4 权重将共享专家保留为 BF16。开启 `--enforce-shared-experts-fusion` 时，加载器在 `process_weights_after_loading` 中通过 `quark_moe_bf16_to_mxfp4` 将共享专家在线量化为 OCP MXFP4（每 32 元素 1B E8M0 Scale + 16B FP4 数据，权重体积压缩 `3.7647x`），使共享专家与路由专家合入单次融合 MXFP4 MoE 核发射。
5. **Ascend NPU 镜像免拷贝 DSA 图重放零 D2H 同步 (`sglang#45949`, `d6ccb495da`)**：在 `ForwardBatch` 上维护主机端 `max_seq_len_host`，使 `AscendAttnBackend` 在 DSA 图重放时无需调用 `.max().item()` 即可计算 `block_table` 切片宽度，每步 D2H 同步从 `1` 降至 `0`。
6. **vLLM 请求终态冻结已处理前缀计数 (`vllm#57646`, `c13d4a5a46`)**：`_VllmReqContext` 在请求完成、抢占或中止的 `finalize()` 时刻将 `(num_computed_tokens, num_in_flight_tokens, num_prompt_tokens, num_tokens)` 冻结为 `_RequestTokens` 快照，防止调度器清理状态后异步 KV Offload 处理器读到归零的计数。
7. **vLLM EAGLE 投机解码前瞻块哈希 (`vllm#61589`, `2fb320d830`)**：`BlockStored` 事件新增 `lookahead_token_ids: list[list[int]] | None`，将未满块的已提交前缀与草稿前瞻 Token 拼接后再计算完整块哈希，提升 EAGLE 多步草稿的前缀缓存命中率且保持 MessagePack 线协议向后兼容。
8. **vLLM DeepSeek-V4.1 GPU Engram 跨卡 HBM 直读替代 DP 集合通信 (`vllm#62091`, `85e199e60c`)**：当 GPU 驻留 Engram 表分片开启 P2P/IPC 时，非本地行（`peer_rows`）直接通过 Peer HBM 指针读取，消除每层 DP All-to-All / All-Gather 集合通信（在 `DP=8` 下跨卡通信量降低 `8.0x`）。
9. **vLLM `ROCM_OCTAVE` 原生 3-Bit KV Cache 后端 (`vllm#59774`, `00b6c4c565`)**：在 `gfx942`/`gfx1100` 上引入 `VLLM_OCTAVE_KV`（原生 3-bit KV 存储、旋转、Prefill 与 Decode 核），相比 FP16/BF16 实现 `5.3333x` 原始位宽压缩。
10. **TensorRT-LLM `mhc_fused_hc` 强制确定性派发 (`TensorRT-LLM#20228`, `8ea175e248`)**：当 `torch.are_deterministic_algorithms_enabled()` 或 `FORCE_DETERMINISTIC` 开启时，绕过自动调优的非确定性算子变体，切回确定性 Sinkhorn/mHC 实现。

---

## 2. 一手源码证据与定量指标对比

| 编号 | 证据 ID / 仓库提交 | 关键实现路径与接口契约 | 定量收益 / 正确性不变量 |
| :--- | :--- | :--- | :--- |
| 11 | `sglang-qwen4-qsa-topk-ple-caller-held-outputs-pr46543-2026` (`5e420a1258`) | `qwen4_exp.py`: `_qsa_prefill_topk_bridges` 调用方预分配张量直写 | Breakable CUDA Graph 分段捕获零悬空张量分配 |
| 12 | `sglang-fix-cpu-engine-detection-torch-compile-fullgraph-pr46640-2026` (`5a1f174700`) | `utils.py`: `is_cpu()` 先检查 `SGLANG_USE_CPU_ENGINE == "1"` | GPU 路径 `os.uname()` 调用降至 `0`，修复 `fullgraph=True` |
| 13 | `sglang-mem-cache-tag-mooncake-offloads-session-pr44611-2026` (`5987ec82ed`) | `mooncake_direct_linker.py`: `session_id` 注入 `ReplicateConfig` | 支持会话级 KV Offload 粘性副本路由 |
| 14 | `sglang-amd-quark-qwen38-quantize-bf16-shared-expert-mxfp4-pr46342-2026` (`0cd2a63b36`) | `quark.py`: `quark_moe_bf16_to_mxfp4` 在线量化 BF16 共享专家 | 共享专家权重压缩 `3.7647x`（含 E8M0 Scale），单核融合发射 |
| 15 | `sglang-npu-ascend-d2h-free-blocktable-gather-dsa-graph-pr45949-2026` (`d6ccb495da`) | `ascend_backend.py`: `max_seq_len_host` 驱动 Block-Table 切片 | Mirror-free DSA 图重放每步 D2H 同步 `1 -> 0` |
| 16 | `vllm-kv-offload-expose-processed-prefix-finalization-pr57646-2026` (`c13d4a5a46`) | `manager.py`: `_VllmReqContext` 终态冻结 `_RequestTokens` | 异步 KV Offload 100% 捕获真实已计算前缀（如 `3072` Tokens） |
| 17 | `vllm-spec-decode-lookahead-block-hashes-eagle-prefix-pr61589-2026` (`2fb320d830`) | `kv_events.py`: `BlockStored.lookahead_token_ids` 前瞻补全块哈希 | 跨块边界草稿 Token 可提前生成完整 16-Token 块哈希 |
| 18 | `vllm-dsv41-read-engram-tables-peer-hbm-instead-dp-pr62091-2026` (`85e199e60c`) | `deepseek_v4_engram.py`: IPC/P2P `peer_rows` HBM 直读 | `DP=8` 下消除冗余全量广播，跨卡读取流量降低 `8.0x` |
| 19 | `vllm-rocm-octave-kv-native-3bit-kv-cache-backend-pr59774-2026` (`00b6c4c565`) | `rocm_octave_attn.py`: `ROCM_OCTAVE` 原生 3-bit KV Cache | 相比 FP16 达到 `5.3333x` 位宽压缩（`16 / 3`） |
| 20 | `trtllm-honor-deterministic-mode-fused-mhc-dispatch-pr20228-2026` (`8ea175e248`) | `mhc_fused_hc`: 检查 `are_deterministic_algorithms_enabled()` | 确定性模式下 100% 规避非确定性 Autotune 变体 |

---

## 3. 配套离线可复现实验

- **实验路径**: `experiments/sglang_qwen4_qsa_ple_mooncake_tag_quark_mxfp4_ascend_vllm_eagle_engram_octave_trtllm_phase97/run.py`
- **校验命令**: `python experiments\sglang_qwen4_qsa_ple_mooncake_tag_quark_mxfp4_ascend_vllm_eagle_engram_octave_trtllm_phase97\run.py --check`
- **单测覆盖**: `tests/test_phase97_experiments.py`
