# SGLang DeepSeek-V4.1 解码 MXFP8 自调优与 ComfyUI Flux ControlNet 残差序列并行分片、TensorRT-LLM 旧版 Executor 清理与 BOLT 双归档流水线、以及 vLLM BERT Token 分类 LoRA 与 NIXL 共享 GPU 编排解析

> **更新日期**：2026-10-22 · **阶段**：Phase 53 (`priority: 62`) · **分类**：`inference-engine`  
> **关联实验**：[`experiments/sglang-dsv41-mxfp8-comfyui-flux-trtllm-exec-vllm-bert-lora/run.py`](../../experiments/sglang-dsv41-mxfp8-comfyui-flux-trtllm-exec-vllm-bert-lora/run.py)

---

## 1. 一手上游来源（First-Hand Sources）

| 仓库与提交 | 日期 (UTC) | 核心技术贡献 |
|---|---|---|
| [`sgl-project/sglang@028de2c5d4`](https://github.com/sgl-project/sglang/commit/028de2c5d485385bee3b68f6463a0a887f42e4d2) | 2026-10-21 | `[Perf] dsv41`：将 DeepSeek-V4.1 FlashInfer CuTeDSL MXFP8 线性层的 `mxfp8_prefill_autotune_min_tokens` 从 `4096` 降至 `1`，在解码尺寸启用调优 tactic（#45630） |
| [`sgl-project/sglang@62b4127668`](https://github.com/sgl-project/sglang/commit/62b4127668e6b4f3a8b8617a34b60679695a8f6f) | 2026-10-21 | `[Feat] comfyui`：Flux 集成模式下支持 ControlNet 双流/单流残差对象的引用去重序列化、序列并行（SP）分片与跨步 Session 缓存隔离（#43698） |
| [`NVIDIA/TensorRT-LLM@a0141d0e56`](https://github.com/NVIDIA/TensorRT-LLM/commit/a0141d0e56af8fad6df709151f78c8769f47335d) | 2026-10-21 | `[TRTLLM-17060][refactor]`：彻底移除 35 个旧版 Executor MPI 标签与 `RequestWithId`/`orchestratorUtils.h`/`createNixlTransferAgent` 兼容层（#20043） |
| [`NVIDIA/TensorRT-LLM@00c184e1f8`](https://github.com/NVIDIA/TensorRT-LLM/commit/00c184e1f81e05ca55b22494c400f9a48d1811ee) | 2026-10-21 | `[None][fix]`：对齐 Nightly Release 与 Post-Merge 的 BOLT 行为，开启 `ENABLE_BOLT_PREMERGE_CONSUME = true` 并保留未插桩主包供 `BoltProfileGen` 采样（#20612） |
| [`vllm-project/vllm@e8e43c6a09`](https://github.com/vllm-project/vllm/commit/e8e43c6a09e4a5fd047cf02b5923db9b2d88ded9) | 2026-10-21 | `[LoRA]`：新增 `TokenClassificationHeadWithLoRA` 与 `AllPool(output_sizes_fn)` 动态每请求标签维度切片，支持 BERT Token 分类 LoRA 适配器（#61391） |
| [`vllm-project/vllm@a0a645abda`](https://github.com/vllm-project/vllm/commit/a0a645abdad134bafbc1a8954dd436e4fb0a5e61) | 2026-10-21 | `[CI][KV Connector]`：当 `GPUS_NEEDED > num_gpus` 时在共享 GPU 上先等待 NIXL Prefiller 启动完成再启动 Decoder，避免显存剖析冲突（#62879） |
| [`vllm-project/vllm@c84edac4cb`](https://github.com/vllm-project/vllm/commit/c84edac4cba45476755f9c462425ffc8290ab6aa) | 2026-10-21 | `[ROCm][CI]`：在 ROCm EAGLE3 投机解码精度测试中强制开启 `VLLM_BATCH_INVARIANT=1`，消除 `hipBLASLt` 按批次形状切换 GEMM 解导致的贪婪接受率翻转（#62893） |

---

## 2. 核心架构机制深挖

### 2.1 SGLang DeepSeek-V4.1 解码 MXFP8 Tactic 解锁与 Flux ControlNet 残差去重分片（`028de2c5d4`、`62b4127668`）

1. **DeepSeek-V4.1 全尺寸 MXFP8 自调优（`028de2c5d4`）**：在 `python/sglang/srt/models/deepseek_v4.py` 的 `autotune_prefill_kernels` 中，当启用 `method.block_fp8_as_mxfp8` 且后端为 FlashInfer CuTeDSL 时，原先将 `mxfp8_prefill_autotune_min_tokens` 设为 `4096`，导致所有 `< 4096` token 的解码与中小批次预填充被迫回退到固定（pinned）tactic。PR #45630 将阈值从 **`4096` 降至 `1`**（仅确定性模式保留 pinned tactic），使全部 14 个典型 token 桶（`1..8192`）均能命中经实测调优的 MXFP8 tactic（从 `2/14` 提升至 **`14/14`**）。
2. **ComfyUI Flux 集成模式 ControlNet 残差流（`62b4127668`）**：
   - **对象身份去重与批次切片**：ComfyUI 常将同一个 ControlNet 残差张量对象重复绑定到多个 Transformer block。`_slice_batch_row` 通过 `_memo[id(value)]`、`_pack_control` 通过 `detached[id(residual)]` 以及 `prepare_control_residual_kwargs` 通过 `prepared[id(residual)]` 保持共享张量引用，使跨进程 IPC 序列化与 `shard_latents_for_sp` 序列并行切分仅对唯一张量执行一次（在 19 个 double-block + 38 个 single-block 共 57 个槽位共享 10 个唯一残差张量时，序列化负载降低 **`5.7x`**）。
   - **跨步缓存污染防护**：将 `"comfyui_control"` 加入 `_SESSION_SKIP_EXTRA`，确保单步去噪携带的 ControlNet 残差不会被缓存并泄漏到后续未携带残差的推理步中。

### 2.2 TensorRT-LLM 旧版 Executor 兼容层清理与 BOLT 流水线对齐（`a0141d0e56`、`00c184e1f8`）

1. **旧版 Executor 彻底下线（`a0141d0e56`）**：在 `cpp/include/tensorrt_llm/runtime/utils/mpiTags.h` 中删除了从 `kDecoderStepNewOutputTokensHost (0)` 到 `kTrtGptModelInflightBatching*` 的 **35 个旧版 MPI 通信标签**（仅保留 `MpiTag::kDefault = 0`），并一并移除 `requestWithId.{h,cpp}`、`requestUtils.{h,cpp}`、`orchestratorUtils.h` 与 `cache_transmission/nixl_utils/interfaces.h` 共 6 个遗留兼容源文件。
2. **Nightly Release BOLT 双归档机制（`00c184e1f8`）**：在 `jenkins/L0_MergeRequest.groovy` 中将 `ENABLE_BOLT_PREMERGE_CONSUME` 置为 `true`，并通过 `isBoltReleasePipeline() && ENABLE_BOLT_POSTMERGE_VARIANT` 控制 Nightly Release 与 Post-Merge 同步发布独立的 BOLT 优化变体包，同时保证未插桩的主 Tarball 原样保留以作为 `BoltProfileGen` 的输入源。

### 2.3 vLLM BERT Token 分类 LoRA、共享 GPU NIXL 编排与 ROCm EAGLE3 批次不变性（`e8e43c6a09`、`a0a645abda`、`c84edac4cb`）

1. **BERT Token 分类 LoRA（`e8e43c6a09`）**：从 `ClassificationHeadWithLoRA` 抽象出 `BaseClassificationHeadWithLoRA` 并派生 `TokenClassificationHeadWithLoRA`。当通过 `max_lora_cls_labels`（如 `16`）并发服务不同标签数的 token 分类适配器（如每个适配器分别含 `3`、`9`、`5` 个标签）时，`TokenPooler` 内部的 `AllPool(output_sizes_fn=classifier.get_output_num_labels)` 按请求顺序对分割后的隐状态执行 `hs[:, :size]` 切片，精准返回 `[[12, 3], [8, 9], [15, 5]]` 形状的每 token 分类 logits。
2. **共享 GPU NIXL 预填充/解码串行化（`a0a645abda`）与 ROCm EAGLE3 批次不变性（`c84edac4cb`）**：当 `GPUS_NEEDED = NUM_PREFILL_INSTANCES * PREFILLER_WORLD_SIZE + NUM_DECODE_INSTANCES * DECODER_TP_SIZE > num_gpus` 时，强制等待所有 Prefiller 端口就绪后再启动 Decoder，防止 Decoder 显存剖析将并发加载中的 Prefiller 权重计入自身开销；在 ROCm EAGLE3 投机解码测试中注入 `VLLM_BATCH_INVARIANT=1`，消除 `hipBLASLt` 因同步/异步批次大小不同选择不同 GEMM 内核而引发的投机接受率漂移。
