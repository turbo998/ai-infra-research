# 推理加速与稀疏索引 Phase 167：SGLang dsv41 SM90 Paged Decode Tile 均衡、PD 分离首拒止损与 vLLM 单次启动融合 Top-K Prefill

## 1. 背景与核心问题

在大规模混合长短上下文推理与 Prefill/Decode（PD）分离集群中，推理引擎面临三类显著的尾延迟与显存带宽瓶颈：
1. **dsv41 SM90 Paged Decode Indexer 的长序列单卡木桶效应**：当同一解码批次中包含 15 条短上下文（如 `4K`）和 1 条超长上下文（如 `128K`）时，若按请求粒度分配线程块（CTA），负责 `128K` 序列的单个 SM 需串行扫描 128 个 Tile，而其余 SM 提早空闲，导致整步 Decode 时间完全被单条长序列绑架。
2. **PD 分离解码传输的无效探测与 Mamba HiCache 的 GPU 槽位浪费**：在 PD 分离（Disaggregation）的每个解码传输轮次中，当目标 Decode 节点因显存压力拒绝首个传输请求后，若继续遍历后续待准入请求将造成无意义的调度循环；此外，混合线性注意力（Mamba）在向分层缓存（HiCache）转储检查点时，旧实现要求额外占用宝贵的 GPU KV Slot。
3. **vLLM 稀疏索引器 Prefill 的 Logits 物化开销**：在稀疏注意力/DSA Indexer 预填充阶段，若先启动打分内核将完整 `[T_prefill, N_blocks]` 分数矩阵写入 HBM，再启动第二次 Top-K 内核读取排序，会产生严重的 HBM 读写放大与双内核启动开销。

---

## 2. 上游源码机制深挖（10 条零重叠提交）

### 2.1 SGLang：dsv41 SM90 Tile 跨 SM 负载均衡、PD 分离首拒止损、AMD Mamba 无槽位 HiCache 与 GB300 DFLASH 调优
- **`2ccbdab65f` (`2026-10-31`, `#47387`)**：在 `python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py` 中重构 SM90 Paged Decode Indexer 的 Tile 划分策略，将所有序列的 KV 页 Tile 打平并均匀分发至全卡 `132` 个 SM，使单条长上下文不再决定整步耗时。
- **`566dbd2cb3` (`2026-10-31`, `#46898`)**：在 `python/sglang/srt/disaggregation/decode.py` 中优化 Decode 端传输准入循环，一旦在单次传输扫描中遇到首次准入拒绝（first refusal）立即停止后续设备准入探测。
- **`2f6e7f3475` (`2026-10-31`, `#44718`)**：在 AMD Mamba 与线性注意力路径中支持无 GPU Slot（slotless）直接向 HiCache 存储状态检查点。
- **`b20fdd2eb3` (`2026-10-31`, `#45876`)**：在 GB300 上将 DFLASH 投机草稿（draft）BF16 GEMM 路由至经调优的 `cutedsl_bf16_gemm` 分发器。

### 2.2 vLLM：单次启动融合打分+Top-K 内核与 XPU 内核发布源
- **`25fc60723c` (`2026-10-31`, `#60618`)**：在 `csrc/fused_mqa_topk/` 下新增单次启动（Single-launch）融合打分与 Top-K 筛选 CUDA 内核，在片上寄存器/共享内存中直接完成候选块打分与 Top-K 归约，**完全不物化中间 Logits 矩阵**。
- **`00ba900aa8` (`2026-10-30`, `#64633`)**：将 `vllm-xpu-kernels` 安装索引切换至 GitHub Release 页面 URL。

### 2.3 TensorRT-LLM & DeepEP：GLM4-MoE 测试覆盖、构建重试与 CI 演进
- **`04f92470a5` (`2026-10-30`, `#21256`) & `3a20b06247` (`2026-10-30`, `#21127`)**：TensorRT-LLM 新增 `Glm4MoeForCausalLM` 模型测试覆盖，并为构建阶段增加自动重试机制。
- **`3fcf25ed1b` (`2025-12-24`, `#528`) & `b57e5e212a` (`2025-12-05`, `#512`)**：DeepEP 升级 GitHub Actions 至 Node 24 兼容版本并记录实验优化分支说明。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/sglang_dsv41_sm90_disagg_stop_vllm_fused_topk_phase167/run.py` 建模验证：

| 子系统与优化机制 | 优化前基线 | Phase 167 优化后 | 核心收益 |
|---|---|---|---|
| SGLang dsv41 SM90 Paged Decode (`15×4K + 1×128K`, `132` SMs) | 单 SM 串行扫描 `128` tiles | 全卡均衡每 SM 最多 `2` tiles | **尾步耗时理论加速 `64.0x`** |
| SGLang PD 分离传输准入 & Mamba HiCache | 遍历全部 `32` 个请求；占 `1` GPU Slot | 第 `4` 个首拒即停；占 **`0` GPU Slot** | **减少 `87.5%` 无效准入探测** |
| vLLM 融合 Sparse-Indexer Prefill (`T=8192, N=4096, K=64`) | 物化 FP32 分数 `128 MiB` | 仅写出 Top-64 索引 `2 MiB` | **HBM 流量降低 `64.0x`** |
