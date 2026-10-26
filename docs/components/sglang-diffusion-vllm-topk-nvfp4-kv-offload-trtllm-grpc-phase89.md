# SGLang CuTe-DSL 多块 Norm 与单遍空间并行因果 Conv3D / GLM 流式解析器、vLLM 持久 TopK 精确溢出恢复 / ModelOpt NVFP4 虚拟缩放 / ROCm AITER W4A16 专家并行 / MRV2 预热采样器剖析与 TensorRT-LLM 原生 UPB Protobuf / CuTeDSL MoE 规范标识深度解析（Phase 89）

> **English Version**: [sglang-diffusion-vllm-topk-nvfp4-kv-offload-trtllm-grpc-phase89.en.md](sglang-diffusion-vllm-topk-nvfp4-kv-offload-trtllm-grpc-phase89.en.md)  
> **证据等级**: `implementation`（固定 commit 源码审阅）+ `author-benchmark`（上游 PR 实测与单测验证）+ `own-offline`（`experiments/sglang_diffusion_vllm_topk_nvfp4_kv_offload_trtllm_grpc_phase89/run.py` 纯标准库建模）

---

## 1. 背景与架构师问题定义

在生产级多模态生成（Diffusion VAE/DiT）、超长上下文稀疏注意力（DSA TopK）与大规模 MoE 推理引擎（SGLang、vLLM、TensorRT-LLM）中，**规约数值确定性（Bitwise Determinism）、流式工具调用 $O(N^2)$ 正则回扫、直方图桶溢出（Candidate Overflow）精度崩塌、预热剖析（Warm-Up Memory Profiling）漏算采样器峰值显存、以及 Python Protobuf 反序列化阻塞 gRPC 吞吐**是导致线上长尾抖动与 OOM 的高频隐患。本篇拆解 2026-10-26 合入 `sgl-project/sglang`、`vllm-project/vllm` 与 `NVIDIA/TensorRT-LLM` 的 10 个一手 commit。

---

## 2. 核心实现路径与故障防御机制

### 2.1 SGLang Diffusion：CuTe-DSL 多块 Norm 逐位不变性、单遍空间并行因果 Conv3D 与 GLM 尾部扫描（PR #46227, #46174, #45143）
1. **CuTe-DSL Norm 单 Warp 步进多 Chunk 的逐位确定性（`sglang@095750baf9`, PR #46227）**：
   - 在 `scale_residual_norm_cutedsl.py` 中，每个 128-bit 向量加载覆盖 `CHUNK = WARP_SIZE * 8 = 256` 列（例如 `D = 5120` 对应每行 `num_chunks = 20` 个 Chunk）。
   - 为允许单个 Warp 循环处理多个 Chunk 而不改变浮点数加法结合顺序，`cta_reduce_sum` 在 SMEM 中分配 `num_chunks + 1`（而非 `num_warps + 1`）个 `Float32` 槽位：每个 Chunk 的 Warp 局部和固定写入 `acc[chunk]`，最后由 Warp 0 按 `chunk = 0 .. num_chunks - 1` 顺序规约，从而严格保证 **无论每个 Warp 处理多少个 Chunk，输出的 BF16 比特位与单 Chunk/Warp 完全一致**。
2. **空间并行因果 Conv3D 单遍构建（`sglang@c4e335f84e`, PR #46174）**：
   - 在 `parallel_conv.py` 中新增 `fill_halo_rows_` 与 `cat_pad_channels_last_3d`，将原本需先拼接时间帧/填充再单独拷贝 Halo 行的两遍高带宽显存操作合并为单遍原地填充，降低多卡 VAE 解码的显存带宽放大。
3. **GLM-4 / GLM-4.7 MoE 流式工具调用消除 $O(N^2)$ 重扫（`sglang@316ff4dce2`, PR #45143）**：
   - 当流式输出长参数（如数千字符 JSON/代码）且 `</tool_call>` 尚未闭合时，旧实现在每个新 Chunk 到达时对整段累积文本重跑 `_STREAMING_PARTIAL_PATTERN` 正则匹配。
   - 新实现通过 `self._open_call = (func_name, args_start)` 与 `self._end_scan_pos` 锁定已打开的工具调用头部，后续每个 Chunk 仅从 `_end_scan_pos` 扫描 `</tool_call>|$`。在 1000 个 3 字符 Chunk 的长参数基准中，扫描工作量缩小 **`500.5x`**（从 $O(N^2)$ 降至严格 $O(N)$）。

### 2.2 vLLM：持久 TopK 候选溢出精确恢复、NVFP4 虚拟缩放、ROCm AITER W4A16 EP、MRV2 采样器预热剖析与 H200 FP8 MoE（PR #62936, #63698, #59255, #57926, #62763）
1. **Persistent TopK 候选溢出精确恢复与非对齐视图保护（`vllm@33a73daf17`, PR #62936）**：
   - **问题**：当大量 Logits 落入同一个粗糙直方图桶（如大量 `-inf` 或半精度绑定值 Ties）导致候选数超出缓冲区容量时，旧持久 TopK 会丢失精确性；同时 `logits[:, 1:]` 等切片视图虽保持 `stride % 4 == 0`，但基地址偏移后不再满足 16 字节对齐，触发向量化加载故障。
   - **修复**：在 `csrc/topk.cu` 中强制检查 `stride % 4 == 0 && base % 16 == 0` 才启用 `vec_size = 4`；在 `csrc/persistent_topk_overflow.cuh` 与 `topk_histogram_4096_overflow.cuh` 中引入 `MediumResume`（记录 11-bit FP32 前缀 `prefix` 与剩余需求数 `remaining`）与多轮有界内存基数重扫（`exact_topk_rescan`，最终轮候选缓存在 `kExactCandidateOffset = 16 KiB` 处），并将 Medium 路径状态全部移入动态 SMEM（`kMediumHistBytes = 3072 B`），避免静态 `__shared__` 拖垮短序列占用率。
2. **ModelOpt NVFP4 MoE 虚拟权重正缩放初始化（`vllm@4f140a23de`, PR #63698）**：
   - 使用 `--load-format dummy` 进行性能压测时，随机初始化的 NVFP4 缩放因子可能含负数或零，且 `w13`（Gate/Up）全局缩放不一致会导致 FlashInfer MegaMoE（`flashinfer_moe_ep_cutedsl`）产生 `NaN/Inf`。
   - 修复后显式绑定 `dummy_weight_value`：`w13_weight_scale = w2_weight_scale = 1.0`、`w13_weight_scale_2 = w2_weight_scale_2 = 1e-3`、`w13_input_scale = w2_input_scale = 1e-2`。
3. **ROCm AITER Triton W4A16 MoE 朴素专家并行（`vllm@536854bf10`, PR #59255）**：
   - 在 `aiter_mxfp4_w4a16_moe.py` 中构造 `gate_valid = (expert_map[topk_ids.long()] >= 0).to(torch.int32)`，使 Scatter Combine 直接跳过不在当前 Rank 的 Expert，而非用零覆盖输出行，从而在 AMD `gfx950` / `gfx1250` 上打通无 All-to-All 的朴素 EP。
4. **MRV2 模型运行器按真实预热参数剖析采样器显存（`vllm@7f56f453ed`, PR #57926）**：
   - 在 `vllm/v1/worker/gpu/model_runner.py` 的 `_dummy_sampler_run` 中，除默认 Pass（编译首个真实 Step 的無采样后处理内核）外，额外使用激活了 `top_k / top_p / logprobs` 的预热参数执行采样器剖析，防止初始显存剖析低估峰值导致线上开启 `top_k/top_p` 时发生 GPU OOM。
5. **H200 FP8 Fused MoE 调优配置（`vllm@61b2527bdb`, PR #62763）**：
   - 为 `E=256, N=320, NVIDIA_H200, fp8_w8a8`（如 DeepSeek-V3/R1 TP8 下的每卡中间维 320）新增 Triton 3.8.0 调优表（小 Batch 采用 `BLOCK_SIZE_M=16, N=64, K=128, num_warps=4, num_stages=4`），并在 `benchmark_moe.py` 支持 `per_act_token_quant`。

### 2.3 TensorRT-LLM：原生 UPB Protobuf 加速 gRPC 与 CuTeDSL Grouped-GEMM MoE 规范标识（PR #19760, #20929）
1. **原生 Protobuf (`upb`) 默认启用与检测告警（`TensorRT-LLM@6448b1b17f`, PR #19760）**：
   - 在 `Dockerfile.multi` 的 `base` 与 `release` 镜像中默认设置 `ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb`，并在 `tensorrt_llm/grpc/_protobuf.py` 启动 OpenEngine / SMG gRPC 服务时检测 `api_implementation.Type()`：若误落入纯 `"python"` 实现则立即输出高优告警（接受 `"upb"` 与 `"cpp"`）。
2. **CuTeDSL Grouped-GEMM MoE 规范化注册标识（`TensorRT-LLM@79d1d922dc`, PR #20929）**：
   - 将 Blackwell CuTeDSL MoE 实现重构至 `cutedsl/` 目录，遵循统一的 `trtllm.cutedsl.<kernel>.<quant>` 命名规范（如 `cutedsl/dense_gemm.py` 中的 `TrtllmCutedslDenseGemmNvfp4Impl`），并通过单测强制校验“标识与具体实现位于同一叶子クラス且全部纳入 `IMPL_PRIORITY` 排序”。

---

## 3. 创新架构横向对比（Fireworks / Together / Cerebras / Groq）

| 维度 | SGLang / vLLM / TensorRT-LLM（Phase 89） | Fireworks AI / Together AI | Cerebras WSE-3 / Groq LPU |
| :--- | :--- | :--- | :--- |
| **长尾 TopK / 采样显存安全** | vLLM 16 KiB 有界基数重扫保证 `-inf` 绑定值下 100% 精确 TopK；MRV2 双 Pass 预热杜绝采样器 OOM | 托管 API 同样面临 `top_k/top_p/logprobs` 动态开启导致的 KV 预算挤压 | 采样通常卸载至主机 CPU 或专用后处理核，不与片上权重争抢 SRAM |
| **控制面序列化与工具调用解析** | SGLang 增量尾部扫描消除 $O(N^2)$ 正则开销；TRT-LLM 强制 `upb` 消除 Python Protobuf 瓶颈 | 高并发 Agent 工具调用场景下，流式 JSON/XML 增量解析器直接决定首包与尾包抖动 | 极低 Token 间延迟（ITL < 2 ms）使主机侧 gRPC/Protobuf 序列化更容易成为端到端瓶颈 |

---

## 4. 架构师练习与可证伪问题

1. **对齐安全性检查**：假设形状为 `(batch=8, vocab=128256)` 的 FP32 Logits 张量起始地址 `base = 0x7f00000000`，当执行切片 `logits[:, 1:]` 后，新的 `stride` 与新的首元素地址 `base'` 分别是多少？为什么仅检查 `stride % 4 == 0` 会导致 `float4`（16 字节对齐要求）加载崩溃？
   - *参考推导*：`stride = 128256`（仍满足 `128256 % 4 == 0`），但 `base' = base + 4`（`base' % 16 == 4 != 0`），直接执行 `ld.global.v4.f32` 会触发 `misaligned address` 异常。
2. **可证伪后续问题**：在开启 `top_k=50, top_p=0.95, logprobs=20` 的高并发大词表（`V=152064`）推理中，MRV2 双 Pass 预热剖析相比单默认 Pass 预热会多预留多少 GiB 的临时工作区（Workspace），从而将最大可分配 KV Cache 块数下调多少百分比？
