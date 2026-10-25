# SGLang 超大 AllReduce 切片与外部缓存连接器、vLLM MRV2 二维分块写与 RDNA3 W4A8、TensorRT-LLM 紧凑停止词与 QuACK SwiGLU 融合深度解析（Phase 86）

> **Language / 语言**: **中文 (当前)** | [English](sglang-vllm-trtllm-phase86.en.md)

## 1. 背景与核心问题

在大规模 LLM 与视频扩散模型（Video Diffusion）推理引擎中，**显存句柄泄漏、隐式 Host-Device 同步、长序列页表拷贝串行化以及采样器最坏情况显存膨胀**是决定长稳吞吐与尾延迟的四大瓶颈。Phase 86 Track 2 审计了 `sgl-project/sglang`、`vllm-project/vllm` 与 `NVIDIA/TensorRT-LLM` 的 10 项核心提交：

1. **SGLang 确定性 Custom AllReduce 超大输入切片与视频扩散零同步**（[PR #34543](https://github.com/sgl-project/sglang/commit/656331295619a4d0267b40b7da0b5cd5b549fcbc)、[PR #46047](https://github.com/sgl-project/sglang/commit/8ef5dec95faaa9c1192b96438c9d36eb988cd2db)、[PR #40595](https://github.com/sgl-project/sglang/commit/f82af9ca17c9a115fd9c1d168c4cc547d1933f93)）：消除 AMD 确定性 AllReduce 在 Chunked Prefill 下的 IPC 句柄溢出崩溃，移除视频扩散每步去噪中的隐式 Host 同步与重复上传，并支持树外（Out-of-Tree）`UnifiedCacheLinker` 插件。
2. **vLLM MRV2 二维分块页表写入、Clef 元数据合并、PCP 前向基准、UMBP 混合连接器与 RDNA3 MXFP4 W4A8 GEMV**（[PR #63613](https://github.com/vllm-project/vllm/commit/237d2ffd08044b52625453af59f1e47b9802d4ff)、[PR #63495](https://github.com/vllm-project/vllm/commit/0765565f103d5579382ad61994a041654684778b)、[PR #62698](https://github.com/vllm-project/vllm/commit/fc2b6ab45ed3e498b52d74fd880a8b40a65ba3a7)、[PR #61791](https://github.com/vllm-project/vllm/commit/c0f9fff649e732253c5472d3a4d28389605b4a01)、[PR #60413](https://github.com/vllm-project/vllm/commit/c0ba7e01ffdd446a087df759e8c8c2bda28de2ea)）。
3. **TensorRT-LLM 按请求紧凑停止词（Compact Stop Words）分配与 SM100/SM103 QuACK BF16 Gate/Up SwiGLU Epilogue 融合**（[PR #20679](https://github.com/NVIDIA/TensorRT-LLM/commit/b96ba434e8a05e2ab52d2082bd8d70b1d844739b)、[PR #19773](https://github.com/NVIDIA/TensorRT-LLM/commit/90d4dcbb8ff49a64a24914b9e1eed0f74bed1d5d)）。

---

## 2. 第一手源码级实现路径分析

### 2.1 SGLang：超大确定性 AllReduce 切片、扩散模型零同步与动态缓存连接器

- **AMD Custom AllReduce 停止对超大输入注册 IPC 缓冲（PR #34543）**：
  - 在 `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` 中，当启用确定性 AllReduce 且 Chunked Prefill 传入大于 `self.max_size`（默认 8 MiB）的张量时，旧实现走 `deterministic_all_reduce` 并对每次迭代的新临时张量调用 `self.register_buffer(inp)`，永久消耗 `rank_data` 槽位，数千步后抛出 `"Rank data buffer is overflowed"`。
  - 新实现引入 `_deterministic_all_reduce_oversized(inp)`：按 $\text{chunk} = \lfloor \text{max\_size} / \text{elem\_size} \rfloor$（BF16 下为 $4,194,304$ 元素）在展平视图上循环调用免注册的 `ops.deterministic_all_reduce_unreg(self._ptr, slice_in, slice_out)`。对于 `16384 x 4096` BF16（`128 MiB`）输入，自动切分为 `16` 个未注册切片，IPC 句柄泄漏从每步 `1` 个降为 **`0` 个**。
- **视频扩散模型消除每步 Host 同步与 H2D 上传（PR #46047）**：
  - 在 `minimax_h3_adaln_cache.py` 中，用 `index_select(0, cache_plan_index.view(1))` 替代 0 维张量下标索引 `block_params[:, cache_plan_index]`（后者在 PyTorch 中会隐式触发 `cudaStreamSynchronize` 把标量读回 CPU）；
  - 在 `minimax_h3.py` 中将 PDD（Parallel Decoding Distillation）多步融合输出头按 `(name, device, dtype)` 一次性上传至 `_pdd_device_heads`；在 `sana_wm_components.py` 中用 `torch.where(invalid[..., None, None], eye, raymats)` 替代 `if bool(invalid.any()):`。50 步去噪循环的重复 H2D 上传从 `300` 次降至 **`6` 次（50.0x 降低）**。
- **UnifiedRadixCache 树外外部连接器工厂（PR #40595）**：
  - 新增 `--unified-cache-external-linker-backend {mooncake,mori,dynamic}` 与 `--unified-cache-external-linker-config`，前端通过无设备依赖的 `UnifiedCacheLinkerConfig.from_dict()` 校验，Worker 侧通过 `create_unified_cache_linker()` 动态导入 `UnifiedCacheLinker` 子类并校验抽象方法完整性。

### 2.2 vLLM：MRV2 2D 页表分块写、Clef 合并传输、UMBP 混合缓存与 RDNA3 W4A8

- **Model Runner V2 (`buffer_utils.py`) 1024 元素二维分块并行写入（PR #62698）**：
  - 旧 `_apply_write_kernel[(n,)]` 对每个暂存差分（staged write）仅启动 1 个 Triton Program，当单请求一次追加 $2500$ 个块 ID 时，超出 `BLOCK_SIZE = 1024` 的尾部在循环中串行拷贝。
  - 新实现维护 `self._staged_write_max_len` 并启动二维网格 `[(n, _num_write_chunks(self._staged_write_max_len))]`（对于 $64$ 个请求、最大长度 $2500$，启动 $64 \times 3 = 192$ 个 Program），每个 `(pid_0, pid_1)` 独立拷贝 `[pid_1 * 1024, (pid_1 + 1) * 1024)` 区间。
- **Clef Pooling 元数据单次 H2D 合并（PR #63495）**：
  - 在 `clef_joint_schema.py` 中用 NumPy 构造 `positions, span_ids, lengths, types, option_question, option_slots, lexical_span_ids` 7 个向量，拼接后调用 **1 次** `async_tensor_h2d(np.concatenate(vectors))` 再在设备侧 `.split()`，将异步 H2D 发射次数从 `7` 次降至 **`1` 次**。
- **RDNA3/RDNA3.5 (`gfx11`) MXFP4 W4A8 `v_dot4_i32_iu8` GEMV（PR #60413）**：
  - 在 `csrc/rocm/mxfp4_w4a8_gemv_rdna3.cu` 中针对 $1 \le M \le 8$ Decode 批次，将激活按 32 元素组动态对称量化为 `int8`，利用 MXFP4 `E2M1` 权重乘以 2 后可**无损精确映射为 `int8`**（`{0, 1, 2, 3, 4, 6, 8, 12}`）的数学性质，每 32 元素组用 **8 条 `v_dot4_i32_iu8`** 指令归约后执行 1 次 FP32 缩放乘法（同时应用 E8M0 权重缩放与激活缩放）。
- **此外**：[PR #63613](https://github.com/vllm-project/vllm/commit/237d2ffd08044b52625453af59f1e47b9802d4ff) 为 `vllm bench forward` 增加 Prefill Context Parallelism (`--prefill-context-parallel-size`) 切片校验；[PR #61791](https://github.com/vllm-project/vllm/commit/c0f9fff649e732253c5472d3a4d28389605b4a01) 为 UMBP KV Connector 补齐 FullAttention + Mamba 混合规格规划、异步任务取消与 `UMBPStoreConnectorStats` 可观测性。

### 2.3 TensorRT-LLM：紧凑停止词缓冲与 SM100 QuACK SwiGLU Epilogue 融合

- **消除 PyTorch Sampler 全局停止词缓冲膨胀（PR #20679）**：
  - 旧 `FinishReasonsHandler` 分配全局稠密张量 `stop_words_cuda [max_num_stop_words, max_stop_word_length, max_num_sequences]` 与 `past_tokens_cuda [max_stop_word_length, max_num_sequences, max_beam_width]`，一旦单个请求携带长停止词列表（如 $128$ 个词、长度 $256$），全局缓冲永久扩容至 $138.4\text{ MiB}$ 且永不收缩。
  - 新实现改为每个序列槽位仅保存 $5 \times \text{int64}$（`40 B`）的 `stop_words_metadata_cuda`（记录 `[base_ptr, rules_len, history_offset, window, beam_width]`），仅为实际携带停止词的请求按需分配紧凑 `int32` 缓冲，并在 `PyExecutor._free_request_resources` 中调用 `sampler.release_request(request)` 立即释放。在 $1024$ 并发槽位、$4$ 个活跃停止词请求场景下，显存占用从 `138.41 MiB` 降至 **`42.45 KiB`（降低 3338.8x）**。
- **SM100/SM103 BF16 Gate/Up GEMM + SwiGLU Epilogue 融合（PR #19773）**：
  - 在 `gate_up_swiglu_quack.py` 与 `gated_mlp.py` 中，通过 `quack.gemm_interface.gemm_act(..., activation="swiglu", concat_layout=("B",))` 直接在 SM100 Tensor Core FP32 累加器 Epilogue 中计算 $\text{silu}(\text{gate}) \odot \text{up}$ 并一次性舍入写出 `[M, I]` BF16，完全省去 `[M, 2I]` BF16 中间张量的写出与重读（在 $M=8192, I=14336$ 时每次 FFN 节省 **`448.0 MiB`** HBM 往返流量，并消除二次 BF16 舍入误差）。

---

## 3. 可复现实验验证（`experiments/sglang_vllm_trtllm_phase86`）

运行 `python experiments/sglang_vllm_trtllm_phase86/run.py --check` 验证通过：
- **SGLang**：`128 MiB` 输入自动切分为 `16` 个免注册切片，扩散模型 50 步 H2D 上传减少 `50.0x`；
- **vLLM**：MRV2 暂存写启动 `192` 个 2D Program，Clef H2D 合并 `7 -> 1`，RDNA3 W4A8 每 32 元素组 `8` 条 `v_dot4_i32_iu8`；
- **TensorRT-LLM**：停止词缓冲压缩 `3338.8x`，QuACK 融合节省 `448.0 MiB` HBM 流量。

## 4. 架构师设计练习与可证伪问题

1. **HBM 流量与算术强度计算**：对于 $M=8192, K=4096, I=14336$ 的 BF16 `GatedMLP.gate_up_proj`，计算未融合路径写读 `[M, 2I]` 中间张量占总权重+激活 HBM 流量的百分比。
2. **可证伪假设**：在小批次 Decode（如 $M \le 32$）下，QuACK `gemm_act` 的 `concat_layout=("B",)` 双列流式 Epilogue 是否因 Tile 拆分粒度不如 cuBLAS Split-K 精细而慢于未融合 `cuBLAS + swiglu`（这也是 TRT-LLM 将 `fuse_bf16_gate_up_swiglu` 设为大模型/大 $M$ 显式 opt-in 的原因）？
