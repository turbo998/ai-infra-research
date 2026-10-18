# 大规模推理强化学习引擎（RLVR / GRPO）：verl 权重重分片、vLLM CuMem 休眠唤醒与长尾 Rollout 消除

## 1. 核心架构问题与结论（Executive Summary）

在 DeepSeek-R1 / QwQ 类推理模型的大规模后训练（RLVR / GRPO / PPO）中，系统在每个迭代步（Step）都必须在 **自回归生成阶段（Actor Rollout，由 vLLM/SGLang 驱动，通常采用小 TP + 大 DP/EP）** 与 **梯度更新阶段（Actor Training，由 FSDP2 / Megatron-LM 3D 并行驱动）** 之间切换。这带来了三大系统级瓶颈：
1. **3D 权重重分片与跨节点同步风暴（Weight Resharding Bubble）**：671B MoE 模型在 BF16 下权重体积高达 **`1,249.8349 GiB`**。若在分离式（Disaggregated）架构下仅由 Actor Rank 0 单点通过单张 400GbE RDMA 网卡（有效带宽 `45 GB/s`）向 Rollout 集群广播，每次 Step 权重同步耗时高达 **`31.8222 s`**。
2. **共置（Colocated）模式的 HBM 峰值显存溢出（OOM）**：若训练与推理共置在同一组 80 GiB GPU 上，训练态 FSDP 参数/梯度/优化器分片（`58.5 GiB`）+ vLLM 权重副本（`19.5312 GiB`）+ Rollout KV Cache 预分配池（`28.0 GiB`）合计达到 **`106.0312 GiB`**，直接击穿 80 GiB 物理显存上限。
3. **Long-CoT 长尾生成气泡（Tail-Straggler Bubble）**：在 GRPO 同步屏障下，同一批 Prompt 中 80% 的采样在 `2K ~ 6K` Tokens 结束，但少数 `32K` Tokens 的深度推理样本迫使整个集群等待最慢序列完成，导致 **`77.73%`** 的 GPU 算力周期处于空转气泡中。

基于 `volcengine/verl`（commit `4b26c788d41ce932ce7706c5ee8e249ab52f4c51`）与 `vllm-project/vllm`（commit `25f560a62c4f955672e2c6080b17ab3a48f96201`）一手源码核验与 `experiments/rl_rollout_resharding/run.py` 离线实验，核心量化结论如下：
- **五种权重同步通道的延迟跨越**：
  - 共置模式 `BucketedWeightSender`（`512 MiB` 桶 + 同卡 CUDA IPC 句柄重建）：每卡仅传输本地分片 `19.5287 GiB`，耗时 **`0.0835 s`**（相对单发 NCCL 提速 **`381.10x`**）；若因容器/NPU 限制退化为 POSIX 共享内存（`use_shm=True`），受限于 PCIe D2H+H2D 双拷，耗时升至 **`1.7954 s`**。
  - 分离式 `NCCLCheckpointEngine` 开启 `multi_sender=True`（接纳同节点 8 个 NVLink 可达的 Actor Ranks 作为并行广播中继）：将跨节点全量广播耗时从 **`31.8222 s` 压缩至 `3.9778 s`（`8.00x` 提速）**。
  - 分离式 `DeltaShardedCheckpointEngine`（稀疏 `(int32 position, bf16 value)` 增量同步）：当单步仅 `12%` 参数发生显著更新时，总传输量降至 **`449.9406 GiB`**，同步耗时进一步降至 **`2.0620 s`（`15.43x` 提速）**，且接收端峰值显存暂存区严格受控于 **`1,120.0 MiB`**（双 `512 MiB` 桶 + 单个 `MAX_ENTRY_ELEMS = 16,777,216` 的 `96 MiB` 解码块），与模型总参数量解耦。
- **vLLM `CuMemAllocator` 虚拟内存解映射（Sleep Level 1 vs Level 2）**：通过 CUDA Driver API `python_unmap_and_release` 在保持虚拟地址指针 `ptr` 不变的前提下释放物理显存页，将训练期峰值 HBM 从 **`106.0312 GiB` 降至 `58.5000 GiB`**（留出 `21.5 GiB` 安全余量）。其中 `Sleep Level 1`（备份权重至 Pinned CPU RAM）切换耗时 **`0.9118 s`**，而 `Sleep Level 2`（直接丢弃权重与 KV 物理页，唤醒后由 Trainer 通过 IPC 原地重填）将休眠/唤醒切换开销压缩至 **`0.0380 s`**。
- **异步有界陈旧度（Staleness-Bounded Async）Rollout 调度**：通过 Partial Rollout 截断回收（`16K` 步长上限）可将 GRPO 空闲气泡从 **`77.73%` 降至 `61.72%`**（墙钟时间减半）；进一步采用有界陈旧度连续异步队列可将气泡压至 **`8.50%`**，阶段吞吐提升 **`4.1095x`**。

---

## 2. 一手源码深度剖析（Implementation Mechanics）

### 2.1 `volcengine/verl`：共置分桶 CUDA IPC 与词表绑定别名过滤

1. **`BucketedWeightSender`（`verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py:73-140`）**：
   - 预分配固定大小通信缓冲桶（默认 `bucket_size_mb = 512`，即 `512 << 20` Bytes）。
   - 遍历参数生成器时，按每个张量的 `weight.element_size()` 进行字节对齐：`offset = (offset + alignment - 1) // alignment * alignment`。
   - 当 `offset + weight.nbytes > self.bucket_size` 时，调用 `get_torch_device().synchronize()`，通过 ZMQ `REQ/REP` 套接字（句柄命名规则为 `ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{local_rank}.sock`，见 `vllm_rollout.py:139-141`）发送 `{"bucket_meta": bucket_meta, "is_last": False}` 并阻塞等待接收端 `recv()` 确认后再复用同一块 `512 MiB` IPC 缓冲区。
   - **跨进程 `CUDA_VISIBLE_DEVICES` 修正（`bucketed_weight_transfer.py:44-52`）**：`rebuild_ipc(handle, device_id)` 将 `reduce_tensor` 生成的参数元组第 7 项 `list_args[6]` 覆写为当前进程的本地 `device_id`，解决了 Trainer 与 vLLM Worker 环境变量视图不一致导致的 IPC 打开失败。
2. **Tied-Embedding 别名丢弃契约（`verl/workers/rollout/vllm_rollout/weight_update_utils.py:21-70`）**：
   - 在 Qwen3.5-VL 等词表权重绑定模型中，Trainer 的 `state_dict()` 会同时导出 `model.embed_tokens.weight` 与 `lm_head.weight`。由于分桶边界可能将二者切分在不同 Bucket 中，vLLM 加载器跳过别名后会导致该 Bucket 因未加载规范名称而报错。`drop_tied_alias_updates` 先经过 `model.hf_to_vllm_mapper` 映射名称，再过滤掉所有 `VocabParallelEmbedding` 非规范别名。

### 2.2 `volcengine/verl`：分离式 `NCCLCheckpointEngine` 与 `DeltaShardedCheckpointEngine`

1. **`NCCLCheckpointEngine` 的 `cupy` 主缓冲区与 `multi_sender` 拓扑（`verl/checkpoint_engine/nccl_checkpoint_engine.py:119-200`）**：
   - **规避 PyTorch `expandable_segments` 注册冲突**：在 `prepare()`（第 `165-174` 行）中，Master 进程特意使用 `cp.zeros(self.bucket_size, dtype=cp.uint8)` 而非 `torch.zeros` 分配发送/接收双缓冲（占 `2 * bucket_size` 显存），因为当 Trainer 开启 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 时，PyTorch 虚拟内存段无法直接被 NCCL/RDMA 注册。
   - **`multi_sender=True` 根节点扇出扩展**：默认 `multi_sender=True` 会通过 `ray.get_runtime_context().get_node_id()` 识别与 Actor Rank 0 处于同一 NVLink 域（同一物理节点）的所有 Actor Workers，将它们一并吸纳进广播通信组充当中继发送方，从而把跨节点 RDMA 注入带宽放大 $8\times$。
2. **`DeltaShardedCheckpointEngine` 的确定性无死锁聚合（`verl/checkpoint_engine/delta_checkpoint_engine.py:14-195`）**：
   - **首次全量 Seed + 后续稀疏 Indices Flush**：第 1 步（Seed）以 `"dense"` 编码流式发送完整 HF 参数并初始化 Trainer 侧的 Pinned CPU 快照；后续每步与快照做字节差分，仅发送变化的 `(int32 position, value)` 对（每元素 `4 + element_size()` 字节）。
   - **为什么 `_GatherQueue` 必须按条目数（Count-Only）而非字节数触发 Flush？（第 `155-184` 行）**：
     - 在 FSDP + MoE 混合并行中，不同 Rank 上参数发生变化的稀疏元素个数（字节数）各不相同，但参数条目（Entry Count）顺序与总数在所有 Rank 上完全一致。如果按单 Rank 本地累积字节数触发 Collective Gather，不同 Rank 会在不同时间点发起集合通信，直接导致 **NCCL 死锁（Deadlock）**。因此 `_GatherQueue` 严格按 `len(entries) >= self.batch_k` 触发集合通信，并在 Gather 内部根据全局 All-Gather 到的 `counts` 按照 `max_round_bytes` 切分轮次。
   - **接收端峰值显存有界性（第 `25-26`, `141-153` 行）**：`_bucket_sliced` 将单个大参数的稀疏增量切分为不超过 `MAX_ENTRY_ELEMS` 的 `_FlushPiece`，并通过 `decode_delta_payload`（`delta_weight_transfer.py:135-195`）校验 `checksum(positions, values)` 后原地（In-Place）掩码更新到 vLLM/SGLang 权重张量上，接收端无需暂存完整模型副本。

### 2.3 `vllm-project/vllm`：`CuMemAllocator` 睡眠/唤醒机制（`vllm/device_allocator/cumem.py:104-260`）

1. **为什么必须用 C 扩展调用 CUDA Driver API？**：文件头部注释（第 `3-9` 行）记录了工程踩坑：`cuda-python` 绑定与 `ctypes` 封装 `libcuda.so` 均因 CUDA Context 不匹配而失败，唯一可靠的路径是在 C 扩展中直接调用 `cuMemCreate` / `cuMemAddressReserve` / `cuMemMap` / `cuMemUnmap` / `cuMemRelease` 并封装为 `torch.cuda.memory.CUDAPluggableAllocator`。
2. **互斥约束（第 `144-148` 行）**：`assert "expandable_segments:True" not in conf`——vLLM 的 `CuMemAllocator` 内存池与 PyTorch 原生 `expandable_segments:True` 不兼容。
3. **按 `tag` 分级的 `sleep` 与 `wake_up`（第 `172-223` 行）**：
   - 在 `use_memory_pool(tag="weights")` 与 `use_memory_pool(tag="kv_cache")` 上下文中分配的张量各自携带 `tag`。
   - 调用 `sleep(offload_tags)` 时：若 `data.tag in offload_tags`（Level 1 默认保存 `"weights"`），则分配 Pinned CPU 张量并调用 `libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)` 备份至主机内存，随后对**所有**受管分配统一执行 `unmap_and_release(handle)` 释放物理 GPU HBM（`"kv_cache"` 数据直接丢弃）。
   - 调用 `wake_up()` 时：遍历 `pointer_to_data`，调用 `create_and_map(handle)` 将新的物理页重新映射回**同一个虚拟地址 `ptr`**（因此 PyTorch Tensor 对象的 `data_ptr()` 完全不需要改变！），若有 `cpu_backup_tensor` 则拷回 GPU 并释放 CPU 备份。

---

## 3. 原创 CPU 离线实验结果（`experiments/rl_rollout_resharding/`）

### 3.1 671B MoE 模型（`1,249.8349 GiB` BF16）五种权重同步模式对比

| 同步模式 (`id`) | 传输机制 | 线上传输总量 (`GiB`) | 接收端峰值暂存 (`MiB`) | 单步同步耗时 (`s`) | 相对单发 NCCL 加速比 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `colocated_cuda_ipc_bucketed_512mb` | 同卡 ZMQ + CUDA IPC (`512 MiB` 桶) | `19.5287 GiB` / GPU | `512.0 MiB` | **`0.0835 s`** | **`381.1042x`** |
| `colocated_shm_fallback_512mb` | 同机 `/dev/shm` D2H+H2D 回退 | `39.0573 GiB` / GPU | `1024.0 MiB` | `1.7954 s` | `17.7243x` |
| `disaggregated_nccl_single_sender` | 跨节点单 Actor Rank 0 NCCL 广播 | `1249.8349 GiB` | `1024.0 MiB` | `31.8222 s` | `1.0000x` |
| `disaggregated_nccl_multi_sender_nvlink` | 跨节点 8-Actor NVLink 中继并发广播 | `1249.8349 GiB` | `1024.0 MiB` | `3.9778 s` | `7.9999x` |
| `disaggregated_delta_sparse_indices` | 跨节点稀疏 `DeltaFlush` (`12%` 变更, `int32+bf16`) | `449.9406 GiB` | `1120.0 MiB` | **`2.0620 s`** | **`15.4327x`** |

### 3.2 共置模式下 `CuMemAllocator` 休眠级别对 80 GiB GPU 显存与切换耗时的影响

| 休眠策略 (`id`) | 训练期峰值 HBM (`GiB`) | 相对 80 GiB 余量 (`GiB`) | 能否在 80 GiB 卡运行？ | 每卡 Pinned CPU 备份 (`GiB`) | Sleep + Wake 切换耗时 (`s`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `no_sleep_co_resident` | `106.0312 GiB` | `-26.0312 GiB` | **否 (`False`, OOM)** | `0.0000 GiB` | `0.0000 s` |
| `sleep_level_1_offload_weights` | `58.5000 GiB` | `+21.5000 GiB` | 是 (`True`) | `19.5312 GiB` | `0.9118 s` |
| `sleep_level_2_discard_all_direct_refit` | `58.5000 GiB` | `+21.5000 GiB` | 是 (`True`) | `0.0000 GiB` | **`0.0380 s`** |

### 3.3 Long-CoT (`2K ~ 32K` 采样分布) GRPO Rollout 调度气泡对比

| 调度策略 (`id`) | 单步 Rollout 墙钟耗时 (`s`) | GPU 空闲气泡占比 | 有效计算利用率 | 相对同步屏障加速比 |
| :--- | :---: | :---: | :---: | :---: |
| `sync_barrier_grpo` | `512.0000 s` | `77.73%` (`0.7773`) | `22.27%` | `1.0000x` |
| `partial_rollout_checkpoint_recycle` | `256.0000 s` | `61.72%` (`0.6172`) | `38.28%` | `2.0000x` |
| `async_continuous_staleness_bounded` | `124.5902 s` | **`8.50%` (`0.0850`)** | **`91.50%`** | **`4.1095x`** |

---

## 4. 解决方案架构师设计指南与可证伪问题

1. **架构师落地清单**：
   - **共置 RL 优先启用 `Sleep Level 2 + Bucketed CUDA IPC`**：在共置部署中，唤醒 vLLM 时随后就要用 Trainer 的新权重覆写旧权重，因此 `Sleep Level 1` 将旧权重备份到 CPU 再拷回 GPU 的 `0.9118 s` 完全是冗余 PCIe 流量；应直接采用 `Sleep Level 2`（丢弃权重物理页，耗时 `0.038 s`）并紧接 `0.0835 s` 的 `BucketedWeightSender` CUDA IPC 重填。
   - **环境变量防坑**：vLLM Rollout Worker 进程严禁设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（会触发 `CuMemAllocator` 断言失败）；而分离式 Trainer Rank 0 在开启 `expandable_segments:True` 时必须使用 `cupy` 独立分配 RDMA/NCCL 注册缓冲区。
2. **后续可证伪实验问题**：
   - 当 RL 训练采用 **FP8 推理 Rollout + BF16/FP32 混合精度训练** 时，在 Trainer 侧完成动态 FP8 量化后再通过 `NCCLCheckpointEngine` 发送，与发送 BF16 后在 Rollout Worker 侧通过 `vLLM` 在线量化相比，哪者在端到端 Step 时间与 KL 散度稳定性上更优？
