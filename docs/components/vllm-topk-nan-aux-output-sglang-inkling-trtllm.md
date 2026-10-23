# vLLM Cooperative Top-K NaN 内联 PTX 序保持修复、AuxOutput 跨 KV 缓存层留存对齐、DeepSeek-V4.1-Flash PCP SWA 有界重放、SGLang Inkling TMLv0 解析与 TensorRT-LLM 原生载荷迁移深度解析

> **语言 / Language**: [中文](vllm-topk-nan-aux-output-sglang-inkling-trtllm.md) | [English](vllm-topk-nan-aux-output-sglang-inkling-trtllm.en.md)  
> **Phase**: 69 (`priority: 78`, Track B — Serving, Scheduling & Speculative Decoding)  
> **核心证据等级**: `implementation` (固定 commit 源码审计) + `own-offline` (本项目确定性标准库仿真 `experiments/vllm-topk-nan-aux-output-sglang-inkling-trtllm/run.py`)

---

## 1. 架构背景与核心工程问题

在多机 Prefill Context Parallelism (`PCP`)、稀疏路由专家（Routed Experts）辅助输出缓存、高吞吐 CUDA Top-K 采样以及多模态/工具调用推理引擎中，本周上游代码库解决了五个极具代表性的生产级正确性与内存/序列化瓶颈问题：

1. **vLLM `cooperative_topk` / `persistent_topk` / `sampled_topk` 在 `NaN` 分数下的非法内存访问 (`vllm-project/vllm@109669cfb314`, `#63118`)**：为什么在 CUDA C++ 中编写无分支位翻转函数 `uint32_t bits = __float_as_uint(x); return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);` 会被 `nvcc` 编译器优化为浮点指令 `-|x|`，进而把 `NaN` (`0x7fc00000`) 规范化为 `-0.0` 的排序键 (`0x7fffffff`)，最终引发 GPU 非法内存访问（Illegal Memory Access）崩溃？
2. **vLLM `AuxOutput`（如 MoE `routed_experts`）跨本地/外部 KV 缓存层的留存上限与共享内存池动态扩容 (`vllm-project/vllm@aea0513a322e`, `#62591`) 与 MRV2 自适应验证显存剖析污染 (`@fbe133ff30d3`, `#61509`)**：当 KV 块被卸载到外部连接器（KV Connector）或因滑动窗口被部分逐出时，调度器如何防止前缀缓存命中越过已保留的 `routed_experts` 槽位？
3. **vLLM DeepSeek-V4.1-Flash 在 `PCP > 1` 下的解码器侧滑动窗口（SWA）有界重放 (`vllm-project/vllm@c7476c7a6fc8`, `#62384`)**：当 Prefill Context Parallelism 跨多卡交错切分序列时，不同 Rank 上的 SWA 活跃行（kept rows）数量不再相等，如何保证跨 Rank 集合通信的张量形状对齐？
4. **SGLang Inkling `tml-renderers` Token-ID 状态机解析 (`sgl-project/sglang@8b9f72417fa8`, `#45340`) 与 Mistral3 `LlavaForConditionalGeneration` 继承重构 (`@4542fc8f598e`, `#45656`)**：为什么必须用特殊 Token ID 而非解码后的字符串来识别 `<|end_message|>` 等控制边界？
5. **TensorRT-LLM `LlmRequest` / `executor::Result` 重型张量载荷向 Python 层迁移 (`NVIDIA/TensorRT-LLM@e139d884a920`, `#20388`) 与多模态 Processor Chat Template 回退 (`@f035671959d2`, `#20880`)**：如何消除跨进程 C++ 绑定层对 `contextLogits`、`encoderInputFeatures` 等巨型张量的重复序列化？

---

## 2. 核心实现路径与源码级机制拆解

### 2.1 vLLM Top-K 内核 `NaN` 编译器规范化陷阱与内联 PTX `mov.b32` 修复 (`#63118`)

在 `csrc/topk.cu` 与 `csrc/sampler_topk_kernels.cu` 中，vLLM 的双阶段基数桶（Radix-Bucket）Top-K 算子存在一个隐蔽的编译器代数重写漏洞：
- **阶段一（FP16 粗分桶）与阶段二（FP32 细化排序）的不一致**：
  - 阶段一将 `float` 转换为 FP16 后映射至 2048 个粗粒度桶，其中 `NaN` 的高 11 位全为 `1`，因此落入最高桶 `bin 2047`（排在所有有限数与 `+inf` 之上）。
  - 阶段二调用 `convert_to_uint32_v2(float x)` 将桶内元素映射为保序 `uint32_t`。原本的 C++ 实现使用 `__float_as_uint(x)` 并对正数置最高位 `bits | 0x80000000u`。然而 **`nvcc` 优化器将整数位或 `bits | 0x80000000u` 识别为置符号位并下降为单条浮点指令 `-|x|`**！在 IEEE-754 硬件浮点流水线中，对 quiet `NaN` (`0x7fc00000`) 执行浮点指令会将其规范化并翻转符号位，使得 `~bits` 输出 `0x7fffffff`——这恰恰是 `-0.0` 的映射键值！
  - **崩溃触发链**：当输入中包含超过 `top_k` 个 `NaN` 时，粗桶 `bin 2047` 触发溢出细化（Refinement Pass），期望在桶内找到键值为 `0xffc00000` 的元素；但由于阶段二将所有 `NaN` 映射成了 `0x7fffffff`，细化循环匹配不到任何元素，写入计数器失控并导致共享内存/全局内存越界写崩溃。
- **Opaque Inline PTX 修复**：
  ```cuda
  uint32_t bits;
  asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(x));
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  ```
  通过不透明的内联 PTX `mov.b32` 阻断 `nvcc` 将位操作折叠回浮点流水线，严格保证单调序：
  $$
  -\infty\;(\texttt{0x007FFFFF}) < -0.0\;(\texttt{0x7FFFFFFF}) < +0.0\;(\texttt{0x80000000}) < +\infty\;(\texttt{0xFF800000}) < \text{NaN}\;(\texttt{0xFFC00000})
  $$

### 2.2 vLLM `AuxOutput` 跨 KV 缓存层留存对齐 (`#62591`) 与 MRV2 显存剖析隔离 (`#61509`)

1. **`AuxOutputSchedulerConnector` 跨层留存护栏 (`#62591`)**：
   - 当开启 `return_routed_experts=True` 时，每个完整 KV 块对应一份辅助专家路由输出，存储在基于共享内存 `mmap` 的 `BlockObjectStore` 中。若同时启用外部 KV Connector，辅助存储容量翻倍为 `num_blocks * 2`。
   - 调度器维护 `retained_block_hashes` 与 `released_block_hashes`：在 `get_computed_blocks` 阶段，通过 `max_cache_hit(request)` 将本地前缀命中截断至连续拥有辅助数据的最大块边界；在外部 KV Connector 阶段，通过 `cap_external_hit(request, num_Local_hit_tokens, num_external_hit_tokens)` 拒绝非块对齐命中（如 `31` tokens）或缺失辅助数据的外部块（回退为 `0` 外部命中并重算），覆盖全部 9 种调度与滑动窗口组合。
   - **动态 Arena 扩容**：当 `BlockObjectStore._evict_until_fits` 发现所有已分配槽位仍被活跃请求引用（无法逐出）时，不再抛出 `MemoryError`，而是发出警告并将 `mmap` Arena 扩容至 $\max(\text{nbytes},\; 2 \times \text{len}(\text{arena}))$。
2. **MRV2 `adaptive_verification` 显存剖析缓存清理 (`#61509`)**：
   - 在 `vllm/v1/worker/gpu/model_runner.py` 的 `capture_model` 中，`speculator.run_Draft_cost_profiling()` 在 CUDA Graph 捕获前执行若干次 Eager 探测运行，会在 PyTorch 缓存分配器中留下约 `2 GiB` 未释放激活块。修复后紧随其后调用 `torch.accelerator.empty_cache()`，防止这部分可回收缓存被计入 `cuda_graph_memory_bytes`。

### 2.3 vLLM DeepSeek-V4.1-Flash `PCP` 下的解码器 SWA 有界重放 (`#62384`)

在 `vllm/model_executor/models/deepseek_v4_1_flash.py` 与 `vllm/v1/worker/gpu/pcp_utils.py` 中，vLLM 为支持 `PCP > 1` 的 SWA 有界重放（Bounded Replay）引入三项核心机制：
- **最小窗口不变量**：在 `init_spec_state` 中强制要求 `window >= 2 * pcp_size`（例如 `pcp_size=2` 时 `window >= 4`），确保任何被裁剪的请求在每个 PCP Rank 上至少保有 1 个本地 Token。
- **跨 Rank 尾部所有权与零填充对齐 (`replay_batch.num_padded`)**：在 `pcp_size = 2`、SWA `window = 128` 下，考虑 1 个单 Token Decode 请求与 1 个从 1024-token APC 前缀命中续写的 513-token Prefill 请求（全局有效长度 1537）。经 PCP Zigzag 交错切分后，Rank 0 持有 256 个本地 Token（其中落入最后 128 个全局位置的保留行有 `127` 行：`[0]` 与 `[130..255]`），而 Rank 1 持有 258 个本地 Token（落入窗口的保留行仅有 `2` 行：`[256, 257]`）。`DecoderReplayLayers._run` 将两侧的 `row_inputs` 统一零填充至 `replay_batch.num_padded = 129` 行（Rank 1 补 `127` 行零），在完成重放层后再切回各自的 `num_kept` 行。

### 2.4 SGLang Inkling TMLv0 解析 (`#45340`)、Mistral3 继承 (`#45656`) 与 TensorRT-LLM 原生载荷瘦身 (`#20388`, `#20880`)

- **SGLang Inkling Token-ID 状态机 (`#45340`)**：集成 `tml-renderers==0.1.0` (`o200k_base_chat`)。`InklingOutputParser.parse_delta_tokens` 直接在原始 Token ID 流上状态转移（控制 Token ID `>= 200001`），因此普通文本 Token 拼出的字面量 `"<|end_message|>"` 不会被误判为消息终止；`reasoning_effort` 严格映射至 $[0.0, 0.99]$（`none=0.0, minimal=0.1, low=0.2, medium=0.7, high=0.9, xhigh=max=0.99`，拒绝 `1.0` 与布尔值）。
- **SGLang Mistral3 子类化 (`#45656`)**：`Mistral3ForConditionalGeneration` 直接继承 `LlavaForConditionalGeneration`，消除包装层代理导致的 `model.eval()` 返回内部子模块问题。
- **TensorRT-LLM `LlmRequest` / `Result` 载荷迁移 (`#20388`) 与 Chat Template 回退 (`#20880`)**：将 `contextLogits`、`generationLogits`、`encoderOutput`、`encoderInputFeatures`、`additionalOutputs` 等大张量移出 C++ Pybind 序列化路径，使 C++ `executor::Result` 的 pickle 状态元组从 **14 个字段缩减至 11 个字段**；`resolve_hf_chat_template` 在多模态 Processor 的 `chat_template` 为 `None` 时自动回退至 `tokenizer.get_chat_template(...)`。

---

## 3. 本项目确定性离线实验验证 (`own-offline`)

运行 `python experiments/vllm-topk-nan-aux-output-sglang-inkling-trtllm/run.py --check`，产出 `summary.json`，核心验证结果如下表：

| 验证维度 | 指标 / 场景 | 基线 / 负对照 | 优化 / 护栏后结果 | 证据等级 |
|---|---|---|---|---|
| **vLLM Top-K `NaN` 排序 (`#63118`)** | `qNaN (0x7FC00000)` 映射键值 | `nvcc` `-|x|` 将 `NaN` 折叠为 `0x7FFFFFFF`（与 `-0.0` 碰撞致越界崩溃） | 内联 PTX `mov.b32` 保持 `0xFFC00000 > +inf (0xFF800000)`，全序单调 | `implementation` + `own-offline` |
| **vLLM `AuxOutput` 跨层留存 (`#62591`)** | 9 种本地/外部/滑动窗口命中矩阵 & Arena 满载 | 外部命中越过已驱逐辅助块；Arena 满时抛 `MemoryError` | 9 种场景命中精确等于 `[64, 32, 32, 32, 0, 0, 32, 0, 0]`；Arena 翻倍扩容 (`6B -> 12B`) | `implementation` + `own-offline` |
| **vLLM PCP SWA 有界重放 (`#62384`)** | `pcp_size=2, window=128` 下 Rank 0 vs Rank 1 保留行 | Rank 0 保留 `127` 行、Rank 1 保留 `2` 行，直接通信形状失配 | 统一填充至 `num_padded=129` 行（Rank 1 补 `127` 行零），强制 `window >= 2*pcp_size` | `implementation` + `own-offline` |
| **SGLang Inkling & TRT-LLM (`#45340/#20388`)** | 字面量 `<|end_message|>`、`effort=1.0`；C++ `Result` 序列化 | 字符串解析器误截断；`Result` 序列化含 14 字段重型张量 | Token-ID 解析保留字面量，拒绝 `effort=1.0`；C++ `Result` 元组缩至 **11 字段** | `implementation` + `own-offline` |

---

## 4. 面向解决方案架构师的练习与可证伪问题

1. **架构诊断题（IEEE-754 位操作与编译器优化）**：为什么在 CUDA C++ 中使用 `reinterpret_cast<uint32_t&>(x)` 或 `__float_as_uint(x)` 后立刻做最高位翻转 `bits | 0x80000000u` 会被编译器视为浮点负绝对值 `-fabsf(x)`？请对比 `cuobjdump -ptx` 下 `FSETP`/`FADD` 规范化行为与 `mov.b32` 的差异。
2. **可证伪假设**：在启用外部 KV Connector + `return_routed_experts=True` 时，若外部缓存命中返回非块对齐的 Token 数（如 `31` tokens，`block_size = 16`），`AuxOutputSchedulerConnector.cap_external_hit` 必须将该请求的外部命中截断为 `0` 并触发完整重算，否则将读取未初始化的专家路由槽位。
