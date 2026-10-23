# SGLang Rust Server 队列 Abort 直达与多模态 Finalize、TensorRT-LLM SM100 QuACK SwiGLU Epilogue 融合与 NVFP4 预填充同步量化阈值、vLLM `cooperative_topk` 真实行长单趟分块与 CPU GDN Spec-Decode 卷积状态对齐深度解析

- **研究日期**：2026-10-24（Phase 57, Priority 66）
- **证据等级**：`implementation` + `author-benchmark`（源码审计）+ `own-offline`（确定性标准库算例验证）
- **上游基准提交**：
  - `sgl-project/sglang@5a86d6ee17e1f2fd34cfa6d703ea38c581ab23fa`（PR `#45899`，2026-04-20）
  - `sgl-project/sglang@eb8ef8084231078c24fa3d8dbfc8a74d0f8f90e4`（PR `#45888`，2026-04-20）
  - `sgl-project/sglang@74c08f3dc5d98dddf052ba03122fc084a31a0269`（PR `#45914`，2026-04-20）
  - `sgl-project/sglang@82ff16dad08809a8819fa02d975d1a44cf202041` / `3e5f5505ed` / `2fbd094a8c`（PR `#45979`, `#45987`, `#45995`，2026-04-20）
  - `NVIDIA/TensorRT-LLM@90d4dcbb8ff49a64a24914b9e1eed0f74bed1d5d`（PR `#19773`，2026-04-20）
  - `NVIDIA/TensorRT-LLM@2676d6ba0aa4e689ebecc92d6fd41e4b95c1aa22`（PR `#20779`，2026-04-20）
  - `vllm-project/vllm@404c2b510e1d73df151d78e5576e1ee7ca0bcaa3`（PR `#62171`，2026-04-20）
  - `vllm-project/vllm@f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`（PR `#63339`，2026-04-20）
  - `vllm-project/vllm@b2e178eaa94bcdd5f0f5e88dec7eb65df1587cc0` / `be08fa5355`（PR `#63190`, `#63360`，2026-04-20）
- **本地验证脚本**：`experiments/sglang-rust-abort-mm-finalize-trtllm-swiglu-nvfp4-vllm-topk-gdn/run.py`

---

## 1. 架构背景与核心问题

在大规模推理服务栈（SGLang、TensorRT-LLM、vLLM）中，2026-04-20 的核心演进聚焦于**控制面死锁消除、长上下文辅助流分配器抖动防治、以及 Blackwell 融合内核与投机解码状态一致性**：

1. **SGLang Rust Server 模式下的 Abort 挂起与多模态 Hash 校验错序**：开启 `--enable-rust-server` 时，Python Tokenizer 进程并未运行，但 Scheduler 在中止排队/抢占请求（`queued` / `preempted` / `retracted`）时仍向无人消费的 `send_to_tokenizer` ZMQ 套接字发送 `AbortReq`，导致客户端 HTTP 连接永久挂起（PR `#45899`）。同时，多模态处理器在外部 caller `image_hashes` 覆盖之前就校验了 `input_ids` 布局，导致 Qwen-VL / DeepSeek-VL2 预计算 hash 校验失效（PR `#45914`）。
2. **TensorRT-LLM GB300 128K 长上下文下辅助流异步量化引发的 `cudaFree` 同步风暴（PR `#20779`）与 SM100 BF16 SwiGLU 中间读写开销（PR `#19773`）**：PR `#19960` 将 MoE 输入 NVFP4 量化放到辅助 CUDA 流 (`aux_stream`) 异步执行以加速 Kimi-K3 解码，但在 DeepSeek-R1 FP4 128K prefill (`PP4`, GB300) 上，跨流 `record_stream` 延长了数 GB 缓冲区的生命周期，触发 PyTorch 缓存分配器频繁触顶并调用 `52` 次同步 `cudaFree`，使 PP `SendRecv` 从 `0.02 s` 暴涨至 `2.98 s`，吞吐腰斩。
3. **vLLM `cooperative_topk` 误用 `stride` 计算分块导致单趟退化为双趟流式（PR `#62171`）与 CPU GDN 投机解码卷积状态错位（PR `#63339`）**：稀疏注意力 `cooperative_topk` 用 `max_model_len` 步长（`256K ~ 1M`）而非实际行长 `seq_len`（如 `40K`）除以 cluster size，导致原本仅需 `10K` 元素/CTA 的任务超出 `48K` 单趟上限而被迫走双趟 TMA 流式路径。

---

## 2. 关键实现路径与定量架构分析

### 2.1 TensorRT-LLM：NVFP4 异步输入量化 Token 阈值 (`#20779`) 与 SM100 QuACK SwiGLU 融合 (`#19773`)

#### (a) `ASYNC_INPUT_QUANT_MAX_TOKENS = 1024` 消除 `cudaFree` 同步风暴 (`#20779`)
在 `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py` 中：
- 辅助流 `quantize_input_async` 会对大尺寸输入激活张量调用 `record_stream(aux_stream)`。当 `num_tokens` 达到长上下文 prefill 规模（如 `16K ~ 128K`）时，主流与辅助流的进度差使巨型张量无法在当前层及时归还空闲池，触发分配器水位越界并强制执行阻塞全设备的 `cudaFree`。
- 源码引入硬阈值 `ASYNC_INPUT_QUANT_MAX_TOKENS = 1024`：
  $$\text{use\_async\_quant} = \text{enable\_async} \land (\text{aux\_stream} \neq \text{None}) \land (N_{\text{tokens}} \le 1024)$$
  - **Decode 阶段**（如 Kimi-K3 最大 decode batch `896 <= 1024`）：继续走 `aux_stream` 异步重叠路径，完整保留 `#19960` 的解码加速。
  - **Prefill 阶段**（`N_tokens > 1024`）：走主流同步量化，彻底消除跨流 `record_stream` 驻留。

| 指标（GB300 DeepSeek-R1 FP4 `isl=128K`, `PP4`） | `#19960` 之前 | 未修复 `main` | 修复后 PR `#20779` | 架构改善（`author-benchmark`） |
|---|---:|---:|---:|---|
| 4 轮迭代 `cudaEventSynchronize` 总耗时 | — | `7.38 s` (90 calls) | **`2.30 s` (8 calls)** | 消除跨流事件等待膨胀 (`-68.8%`) |
| 4 轮迭代阻塞 `cudaFree` 次数 / 耗时 | `0` | `52` calls (`0.36 s`) | **`0` calls (`0.00 s`)** | 100% 根除分配器同步释放风暴 |
| 4 轮迭代 PP `ncclSendRecv` 阻塞耗时 | — | `2.98 s` | **`0.02 s`** | 消除流水线气泡传染 (`149x` 降低) |
| 3 组 Context-Only Median TTFT (ms) | `1,200,228 / 1,198,861 / 1,204,054` | `1,691,192 / 1,706,087 / 1,598,272` | **`1,186,204 / 1,198,656 / 1,202,545`** | Median TTFT 降低 **`29.12%`**（最差降低 `32.9%`） |
| Disaggregated 端到端输出吞吐 (tok/s) | `~893` | `~460` | **`875.8`** | 吞吐恢复 **`1.904x`** |

#### (b) SM100/SM103 QuACK BF16 Gate/Up GEMM + SwiGLU Epilogue 融合 (`#19773`)
在 `tensorrt_llm/_torch/modules/gated_mlp.py` 与 `linear.py` 中：
- 当硬件为 Blackwell (`SM100` / `SM103`)、激活为 `swiglu_ai`、无 bias、且 `K % 8 == 0`、`I % 8 == 0` 时，`GatedMLP` 调用 QuACK `gemm_act(x, w_gate_up, activation="swiglu", concat_layout=("B",), tuned=False)`。
- **HBM 流量与精度收益**：未融合路径需写出 `[M, 2I]` BF16 中间激活（$4MI$ 字节）、再由独立 `swiglu` kernel 读入 `[M, 2I]`（$4MI$ 字节）并写出 `[M, I]`（$2MI$ 字节），合计 $10MI$ 字节；融合后直接在 Tensor Memory / 寄存器 Epilogue 中用 FP32 计算 $\text{silu}(g)\odot u$ 并仅做 **1 次** BF16 舍入写出 `[M, I]`（$2MI$ 字节），省去 $8MI$ 字节（**80% 的中间激活 HBM 读写**，在 $M=4096, I=8192$ 下单层每次省去 `256 MiB` HBM 往返）以及双重 BF16 截断误差。

### 2.2 vLLM：`cooperative_topk` 真实行长分块 (`#62171`) 与 CPU GDN Spec-Decode 卷积状态修复 (`#63339`)

1. **`cooperative_topk` 单趟分块判定修正 (`#62171`)**：
   - 在 `csrc/ops.h` 与 `vllm/_custom_ops.py` 中，将 `stride`（KV 缓冲区物理步长，常等于 `max_model_len = 256K..1M`）与真实活跃行长 `seq_len` 分离。
   - 设 cluster size $C = 4$，单趟 staging 共享内存上限为 $L_{\text{single}} = 49,152$（`48K` 元素/CTA）。当真实上下文 $S = 40,000$、物理步长 $\text{stride} = 262,144$ 时：
     $$\text{Old: } \left\lceil \frac{262,144}{4} \right\rceil = 65,536 > 49,152 \implies \text{被迫走 Two-Pass Streaming}$$
     $$\text{New: } \left\lceil \frac{40,000}{4} \right\rceil = 10,000 \le 49,152 \implies \text{命中 Single-Pass Staging}$$
   - 在 B300 上（16 rows, `40K` context, `1M` stride），`cooperative_topk` 耗时从 **`12.18 us -> 11.24 us`（`+8%` 提速）**；在 4x GB200 运行 GLM-5.3-NVFP4 AgentX 基准时，并发 `1 / 4 / 8 / 16` 下端到端耗时分别下降 **`2.8% / 2.5% / 1.5% / 1.5%`**。
2. **CPU GDN 投机解码卷积状态 GPU 布局对齐 (`#63339`)**：
   - 在 `csrc/cpu/gdn_linear_attn.cpp` 与 `short_conv.cpp` 中，当多请求 batch 内各行的草稿接受长度不同（`query_len < num_spec + 1` 或 `num_accepted > query_len`）时，旧版仅在 `seq_lens[0] == 1 + num_spec` 时走快速更新，且 fallback 路径将 `conv_state` 截断为 `width - 1` 而非 GPU 预期的 `state_len = conv_kernel + num_spec`，引发 `RuntimeError` 与 **66% 的投机解码输出失真**。新版统一维护完整 `state_len` 滑动窗口并在 `seq_lens[0] == state_len - (width - 2)` 时启用原生追加。
3. **`prompt_embeds` 多模态 LoRA 守护 (`#63190`)**：`set_active_mm_loras` 显式跳过 `modality == "prompt_embeds"`，修复预编码嵌入输入下的 `KeyError`。

### 2.3 SGLang：Rust Server Abort 闭环 (`#45899`, `#45888`)、多模态 `finalize` (`#45914`) 与 Diffusion 优化 (`#45979`, `#45987`, `#45995`)

- **Scheduler Abort 直推 Rust gRPC/Channel (`#45899`, `#45888`)**：当 `rust_server is not None` 时，`Scheduler._send_abort_to_client()` 调用 `self.rust_server.push_abort(abort_req, req)`，不再向悬空的 `send_to_tokenizer` 投递；Rust HTTP 层同时暴露原生 `POST /abort_request`。
- **多模态 `finalize` 生命周期钩子 (`#45914`)**：在 Rust 多模态管线 `encode_items` 与 caller `image_hashes` 覆盖**之后**调用 `processor.finalize(&mut entry)`，使 Qwen-VL / DeepSeek-VL2 的 token 展开与布局校验拿到最终完整 hash。
- **Diffusion 三连优化**：`#45979` 在 residency planner 未就绪时跳过无意义的 `probe_transformer` 干跑；`#45987` 在 `/v1/images` 路径支持 `return_image_bytes=True` 内存直传 PNG/JPEG 字节免磁盘落盘；`#45995` 修复 LTX-2.3 BCG 在 Sequence Parallel (`SP`) 下的视频 RoPE 起始帧偏移（从硬编码 `0` 改为 `video_sp_start = sp_rank * frames_per_rank`）。

---

## 3. 架构师设计练习与可证伪问题

1. **计算题**：在 Blackwell SM100 上运行一个 60 层 Dense/MoE 混合模型，若每层包含一个 $M=4096, I=8192$ 的 BF16 Gate/Up SwiGLU MLP，启用 QuACK `gemm_act` epilogue 融合后，单次 4096-token prefill 共减少多少 GiB 的 HBM 读写流量？
   - **解答**：单层节省 $8 \times 4096 \times 8192 = 268,435,456\text{ B} = 256\text{ MiB}$；60 层合计节省 $60 \times 256\text{ MiB} = 15,360\text{ MiB} = 15.0\text{ GiB}$。
2. **可证伪假设**：在 TensorRT-LLM 中将 `ASYNC_INPUT_QUANT_MAX_TOKENS` 设为 `2048` 还是 `1024` 更优？可通过在 GB300 上扫描 `N_tokens in {1024, 1536, 2048, 4096}` 时 PyTorch CUDA caching allocator 的 `num_alloc_retries` 是否保持为 `0` 来直接证伪。
