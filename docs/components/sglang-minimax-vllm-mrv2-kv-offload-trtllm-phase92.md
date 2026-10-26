# Phase 92 Track 2：SGLang MiniMax-M3.1-Flash/AMD Softcap/Diffusion、vLLM MRV2/ROCm DSv4.1/KV Offload 与 TensorRT-LLM Unified KV

## Executive Summary

本篇覆盖 `sgl-project/sglang`、`vllm-project/vllm` 与 `NVIDIA/TensorRT-LLM` 的 10 项最新一手上游提交（`priority: 101`）：

1. **`sgl-project/sglang@a9bf8c76a0` (`#45925`)**：支持 MiniMax-M3.1-Flash，集成 MSA（`1e7bf76f26`）多头 `q8kv4` indexer 与 `--monokernel-backend msa` 插件；在 DSPark `_commit_inject_layout_kernel` 中对未提交槽位先施加 `committed = mask & (c < commit_len)` 掩码再解引用 `req_pool_ptr`；在 `_fused_kv_norm_rope_write_kernel` 中仅对前 `ROT` 维执行 RoPE 旋转（`HALF = ROT // 2`），其余 `ROT..D` 维在 RMSNorm 后直通写入。
2. **`sgl-project/sglang@d96c386335` (`#45531`)**：修复 AMD ROCm `aiter_backend.forward_decode` 将 `softcap=0` 硬编码传入 `unified_attention` 的问题，改为传递 `softcap=layer.logit_cap`（如 `50.0`），确保 Gemma/Grok 等帶 logit softcap 模型在 decode 阶段的数值一致性。
3. **`sgl-project/sglang@ce77d5e93d` (`#46057`)**：重构 SGLang Diffusion `/v1/videos` 生命周期管理：使用 `AsyncExitStack` 与请求独占临时目录，在请求被拒绝或后台任务退出后自动清理上传文件；取消预处理时等待活跃 CPU 任务结束；当 `--output-path ""` 时要求云端上传成功后才标记任务 `completed`。
4. **`sgl-project/sglang@042dc54cbf` (`#46249`)**：在 SGLang Diffusion 多副本 DP 调度器客户端 `_session_key(batch)` 中识别 `ReleaseRealtimeSessionReq` 并返回 `req.session_id`，确保实时会话清理请求精确路由至持有该会话的 DP replica。
5. **`vllm-project/vllm@4135590ffe` (`#63431`)**：在 vLLM GPU ModelRunner V2 (`MRV2`) 的 `capture_model(profile_only=True)` 内存探测阶段跳过 `adaptive_verification` 的代价剖析 dummy 运行，防止残留在分配器缓存中的激活张量被误计入 CUDA Graph 内存开销而压缩正式 KV Cache 容量。
6. **`vllm-project/vllm@9744ada582` (`#63576`)**：在 vLLM ROCm DeepSeek-V4.1 (`DSv4.1`) monokernel attention 的 `stage_combine` 中，当处于 index-source 发射模式（`"krdy" in c`）时先执行 `await_flags(c, c["krdy"], 1, lambda u: t)` 并以 `CM_DEV` 设备级缓存一致性修饰符读取 `a["klen"]`，消除因读取陈旧 `klen` 导致的活跃键值块数 `nlive` 截断。
7. **`vllm-project/vllm@14aefcd4f9` (`#62929`)**：在 vLLM KV Offload 分层管理器（`TieringOffloadingManager`）发生 cache reset 后，阻塞二级存储向一级存储的块提升（promotion），直至 reset 前在途的异步传输任务全部汇报完毕，防止陈旧的 pre-reset KV 块污染重置后的一级缓存。
8. **`vllm-project/vllm@8a4b018f1c` (`#63628`)**：在 vLLM KV Offload P2P NIXL 传输层（`add_remote_peer` / `add_remote_peer_async`）中，当构建对端描述符失败或世代号（generation）过期时自动回滚已注册的底层 NIXL agent/内存句柄，消除句柄泄漏。
9. **`NVIDIA/TensorRT-LLM@fce28f0726` (`#20178`)**：在 TensorRT-LLM Unified KV Cache Manager V2 的 `BufferConfig` 中引入 `coalescingGroup`（`CoalescingKey = std::pair<int, size_t>`），使同一 `LifeCycleId` 下的缓冲区仅在 `(coalescingGroup, expandedBufferSize)` 同时匹配时才合并进同一存储池组。
10. **`NVIDIA/TensorRT-LLM@732fd63463` (`#20915`)**：在 TensorRT-LLM PyExecutor beam search 中注册 `start_beam_histories_callable`（`PendingBeamHistories`），在执行 logits post-processors 前按父束回溯指针重构每条活跃束的真实生成序列，修复束交换后按槽位列表读取导致词频惩罚/语法约束错位的问题。

---

## 1. Upstream Commit Analysis

### 1.1 SGLang MiniMax-M3.1-Flash 单核插件、Partial RoPE 与 AMD Softcap

在 `sgl-project/sglang@a9bf8c76a0` (`#45925`) 中，MiniMax-M3.1-Flash 引入了三大底层改动：
- **MSA 单核插件**：容器镜像固定 `MSA_COMMIT=1e7bf76f26a5f53cfc2d4d297e1ec2e40955e431`，提供多头 `q8kv4` 索引器、跨步缓存视图（strided cache views）及 `--monokernel-backend msa` 入口。
- **DSPark 投机校验窗口安全掩码**：在 `_commit_inject_layout_kernel` 中，未提交或补齐槽位的请求表项可能未初始化，因此在读取 `commit_len` 后通过 `committed = mask & (c.to(tl.int64) < commit_len)` 保护后续寻址。
- **Fused KV Write Partial RoPE**：在 `_fused_kv_norm_rope_write_kernel` 中新增编译期常量 `ROT: tl.constexpr`，令 `HALF = ROT // 2`，仅旋转前 `ROT` 维，而 `ROT..D` 维在归一化后原样写入 KV Cache。

在 `sgl-project/sglang@d96c386335` (`#45531`) 中，`aiter_backend.py` 的 `forward_decode` 原本写死 `softcap=0`，导致启用 logit softcapping（$\text{softcap} \cdot \tanh(x / \text{softcap})$）的模型在 ROCm decode 阶段跳过软上限裁剪；现改为传入 `softcap=layer.logit_cap`。

### 1.2 vLLM MRV2 显存预算保护、ROCm DSv4.1 `krdy` 屏障与 KV Offload 一致性

- **MRV2 CUDA Graph Memory Profiling (`vllm@4135590ffe`, `#63431`)**：`capture_model` 分两次运行：第一次 `profile_only=True` 运行在临时 KV Cache 上以测量 CUDA Graph 占用的显存，第二次在正式分配的 KV Cache 上捕获并校准。若在 `profile_only=True` 期间运行 `adaptive_verification` 的代价剖析，其留在 PyTorch caching allocator 中的激活显存会被误算作 CUDA Graph 固有开销，直接导致正式 KV Cache 缩水。
- **ROCm DSv4.1 `stage_combine` 跨阶段同步 (`vllm@9744ada582`, `#63576`)**：在 DeepSeek-V4.1 AMD monokernel 中，当键值长度 `a["klen"]` 由上游 indexer 阶段异步写出时，`stage_combine` 必须先执行 `await_flags(c, c["krdy"], 1, lambda u: t)` 并带 `CM_DEV` 标志执行 `ld_i32(a["klen"], t, CM_DEV)`，否则读取到 L1 缓存中的旧 `klen` 会漏算新增 KV 分块。
- **KV Offload Reset Barrier & P2P NIXL Rollback (`14aefcd4f9` `#62929`, `8a4b018f1c` `#63628`)**：前者在 cache reset 后挂起所有 block promotion 直至在途 job 归零；后者在 P2P NIXL 异步注册异常或世代被抢占时对称卸载 native handles。

---

## 2. Deterministic Simulation & Quantitative Verification

通过标准库模拟器 `experiments/sglang_minimax_vllm_mrv2_kv_offload_trtllm_phase92/run.py` 验证的量化结果如下：

| 模块 / 提交 | 核心指标 | 修复前行为 | 修复后实测/模拟值 | 架构收益 |
|---|---|---:|---:|---|
| `sglang@a9bf8c76a0` (`#45925`) | `D=128, ROT=64` Partial RoPE `cos/sin` 半宽 | `64` (`D // 2` 越界) | `32` (`ROT // 2`) + `64` 维直通 | 精确匹配 MiniMax-M3.1-Flash |
| `sglang@d96c386335` (`#45531`) | AMD `aiter` decode `raw_logit=120, cap=50` 输出 | `120.0` (`softcap=0`) | `49.1669` (`softcap=50.0`) | 恢复 ROCm decode 数值等价性 |
| `sglang@042dc54cbf` (`#46249`) | `dp_size=4` 下 `ReleaseRealtimeSessionReq` 命中 owner | 轮询错发至非 owner | `4 / 4` (`100%`) 命中 owner | 消除 DP 副本实时会话残留泄漏 |
| `vllm@4135590ffe` (`#63431`) | `7` 个捕获档位下避免误扣的 KV Cache 空间 | 误扣 `168 MiB` (`84` 块) | 误扣 `0 MiB` (找回 `84` 块) | 最大化生产环境 KV Cache 容量 |
| `vllm@9744ada582` (`#63576`) | ROCm DSv4.1 `klen: 512 -> 2048 (ck=64)` `nlive` 分块 | `8` 块 (漏算 `24` 块) | `32` 块 (`await krdy` + `CM_DEV`) | 消除长上下文结合阶段注意力截断 |
| `vllm@14aefcd4f9` (`#62929`) | Cache reset 期间 `3` 个在途任务下的陈旧块提升数 | `> 0` (pre-reset 污染) | `0` (排空后才允许提升) | 严格保证 Reset 后缓存世代隔离 |
| `TensorRT-LLM@fce28f0726` (`#20178`) | 同生命周期、同展开大小 (`64 KiB`) 的 KV 与 Indexer 池组 | 误合并为 `2` 个池组 | 按 `coalescingGroup` 拆分为 `3` 组 | 隔离异构缓冲区物理池布局 |

---

## 3. Architectural Insights & Design Exercises

1. **CUDA Graph 显存测量窗口的纯粹性原则**：推理引擎通过 $\Delta \text{FreeMem} = \text{Mem}_{\text{before}} - \text{Mem}_{\text{after}}$ 估算 CUDA Graph 显存并从剩余 HBM 中划分 KV Cache。任何在测量窗口内执行但并非 Graph 拓扑一部分的剖析（如投机解码 `adaptive_verification` 曲线采样）都必须在 `profile_only=True` 时屏蔽。
2. **Beam Search 槽位历史与假设历史的分离**：在步 $t$ 发生束重排（beam swap）后，物理槽位 $b$ 在步 $0..t$ 产出的 token 列表不再构成同一条候选路径，logits 处理器必须消费通过父指针回溯重构的 `PendingBeamHistories`。

---

## 4. Verification & Falsifiable Follow-Up Questions

- **验证命令**：`python experiments/sglang_minimax_vllm_mrv2_kv_offload_trtllm_phase92/run.py --check` 与 `python -m unittest tests/test_phase92_experiments.py`。
- **可证伪后续问题**：在 vLLM KV Offload P2P NIXL 传输层中，若对端节点在并发执行两次 `add_remote_peer_async(peer_id)` 时第一次较慢完成，世代号（generation）检查是否能确保第一次注册生成的陈旧 `remote_dlist` 被立即注销而不覆盖第二次的新句柄？
