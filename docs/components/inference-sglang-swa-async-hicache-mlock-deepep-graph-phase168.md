# 推理与统一显存/通信调度 Phase 168：SGLang 统一内存 SWA 尾部无同步绑页与 HiCache `mlock` 锁页、DeepEP 跨节点分发 CUDA Graph 及 vLLM Rust 工具调用 Token 计量

## 1. 背景与核心问题

在混合滑动窗口注意力（SWA）、分层主机缓存（HiCache）与跨节点 MoE 专家并行（EP）服务中，推理运行时面临三项关键瓶颈：
1. **SGLang 统一内存 SWA 扩页的主机同步停顿与 HiCache 页面老化扫描**：在 `alloc_extend_swa_tail` 为 SWA 尾部绑定新物理页时，若每次扩页均触发一次 Host Synchronization，会打断 GPU 异步流水线；同时 HiCache 的锁页主机内存（Pinned Host Buffers）若未调用操作系统 `mlock`，Linux 内核页面老化（page aging / kswapd）仍会周期性扫描这些缓冲区，引发 PCIe DMA 抖动。
2. **DeepEP 跨节点分发（Internode Dispatch）的 CPU 启动瓶颈与 TRT-LLM Ray 多卡 JIT 竞态**：在 61 层 MoE 模型解码中，跨节点 RDMA Dispatch 若无法被 CUDA Graph 捕获，每步会产生数十次主机端启动延迟；此外 TensorRT-LLM 在启动多个 Ray Worker 时若并发即时拉取 FlashInfer TRT-LLM-Gen 产物，会触发本地缓存锁冲突（`nvbugs/6907190`）。
3. **vLLM Rust 前端在 `openai-responses` 基准中漏计工具调用增量 Token**：当模型生成结构化 Tool Call Deltas 时，旧版 `vllm-bench` 仅统计普通文本 Token 而漏计工具调用参数 Token，导致吞吐与 TPOT 严重失真。

---

## 2. 上游源码机制深挖（11 条零重叠提交）

### 2.1 SGLang：SWA 无主机同步绑页、HiCache `mlock`、UE8M0 中性缩放与 Rust Server 精简
- **`573649879b` (`2026-10-31`, `#47542`)**：在 `python/sglang/srt/mem_cache/kv/alloc/chain.py` 的 `alloc_extend_swa_tail` 中直接异步绑定 SWA 尾部新页，消除每次扩页的 Host Sync。
- **`9dbc462c65` (`2026-10-31`, `#47543`)**：在 `python/sglang/srt/mem_cache/pool_host/common.py` 中对 HiCache Pinned Host Buffers 执行 `mlock`，彻底阻止内核页面老化扫描。
- **`800de62d7c` (`2026-10-31`, `#47337`) & `bf1ece1f7e` (`2026-10-31`, `#47556`)**：在 `weight_utils.py` 中为虚拟权重（dummy weights）设置中性 `UE8M0` 块缩放因子（指数偏置 `127`，对应缩放值 `1.0`），并清理 Rust Server 输出流不再读取的冗余字段。

### 2.2 DeepEP & TensorRT-LLM：跨节点 Dispatch CUDA Graph 与 Ray Worker 预取
- **`92fe2deaec` (`2025-11-05`, `#438`) & `1300811e24` (`2025-12-04`, `#502`)**：DeepEP 在 `csrc/kernels/internode.cu` 中正式支持跨节点分发（Internode Dispatch）的 CUDA Graph 捕获与重放，并同步更新 Hybrid-EP 实验分支文档。
- **`c94310bdff` (`2026-10-30`, `#21286`) & `bba275b1ab` (`2026-10-30`, `#21307`)**：TensorRT-LLM 在 Ray Workers 启动前由主进程预先拉取（Prefetch）FlashInfer TRT-LLM-Gen 算子产物，根除多卡并发初始化竞态。

### 2.3 vLLM：Rust 前端 Tool Call Delta 精确计量、原生渲染器门禁与 OLMoE LoRA
- **`32a73034d5` (`2026-10-30`, `#64557`) & `a70e48deba` (`2026-10-30`, `#64274`)**：在 `vllm-bench` `openai-responses` 中将 Tool Call Deltas 纳入生成 Token 计数，并在 Rust 前端使用原生渲染器时拒绝外部 Chat Template 冲突。
- **`bc57dc9d53` (`2026-10-29`, `#63384`)**：完成 Transformers 后端下的 OLMoE 迁移并修复 MoE LoRA 权重路由。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/sglang_swa_async_hicache_mlock_deepep_internode_graph_phase168/run.py` 验证：

| 子系统与机制 | 优化前基线 | Phase 168 表现 | 核心收益 |
|---|---|---|---|
| SGLang SWA 扩页 (`64` 步) & HiCache (`64 GiB`) | `64` 次 Host Sync；内核扫描 `64 GiB` | **`0` 次 Host Sync**；内核扫描 **`0 GiB`** | 消除 SWA 扩页停顿与 `kswapd` 页扫描抖动 |
| DeepEP Internode CUDA Graph (`61` MoE 层) & TRT-LLM | Eager CPU 启动 `18.5 μs/层` | Graph 重放 `2.5 μs/层`；`0` JIT 竞态 | **跨节点分发 CPU 开销降低 `7.4x`** |
| vLLM Rust `openai-responses` (`120` 文本 + `180` 工具 Token) | 仅统计 `120` tokens（漏计 `60%`） | 完整统计 **`300` tokens**（准确度 `100%`） | 修复 Agent 工具调用场景下的吞吐/TPOT 基准失真 |
