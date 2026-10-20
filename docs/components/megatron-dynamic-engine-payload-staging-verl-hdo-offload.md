# Megatron-LM 动态推理引擎异步载荷卸载、空闲 Rerun 状态严格加载与 veRL HDO 混合设备优化器卸载保护深挖

> **English Version**: [megatron-dynamic-engine-payload-staging-verl-hdo-offload.en.md](megatron-dynamic-engine-payload-staging-verl-hdo-offload.en.md) | **返回索引 / Back to README**: [README.md](../../README.md)

## 1. 问题定义与工程背景 (Problem Statement & Architectural Context)

在面向多模态（VLM）与强化学习后训练（RL Rollout + Megatron 训练切换）的大规模推理与训练系统中，**请求完成时的大载荷持久化（Payload Staging）**、**分布式 Checkpoint 严格加载（Strict Load）** 以及 **混合设备优化器（`HybridDeviceOptimizer`, HDO）状态卸载** 存在三个关键瓶颈与正确性陷阱：

1. **`DynamicInferenceEngine` 同步载荷写出阻塞 GPU 解码主循环与 VLM 显存滞留（`NVIDIA/Megatron-LM` `#8169`）**：
   - 当请求完成时，引擎需将大体积有效载荷（生成的 logprobs、MoE 路由索引 `routing_indices`、token IDs 以及 VLM 视觉媒体张量）通过 `RequestPayloadStager.stage()` 写入远端存储后端，并在 ZMQ 响应中仅携带轻量级 `payload_stage_metadata`。
   - 若在解码步（`step`）的记账阶段同步调用 `stage()`，一次耗时 `45 ms` 的网络存储写出会直接卡住整个 GPU 批次的解码循环（当单步有 2 个请求同时完成时，单步耗时从 `12.0 ms` 暴涨至 **`102.0 ms`**，劣化 **`8.50x`**）。
   - 反之，若将 `stage()` 放入后台线程异步执行，被后台任务持有的已完成 `DynamicInferenceRequest` 若仍保留 GPU 上的 `prompt_tokens` 与 VLM `media_tensors`（如 `imgs`），则会在高并发完成峰值期间造成数百 MiB 甚至数 GiB 的 GPU 显存滞留，且多线程并发操作引擎 ZMQ socket 会导致底层 C++ 套接字崩溃。
2. **`rerun_state_machine` 处于 `NOT_RUNNING_YET` 空闲态时严格分布式 Checkpoint 加载失败（`NVIDIA/Megatron-LM` `#7222`）**：
   - 为支持分布式保存计划缓存（cached save plans），即便作业处于空闲态（`RerunState.NOT_RUNNING_YET`），保存的 Checkpoint 中依然包含 `rerun_state_machine` 分片。
   - 然而在加载时，`rerun_state_machine.validate_state_dict(...)` 对空闲态返回 `False`，导致旧版 `checkpointing.py` 未将 `rerun_state_machine` 加入期望的 `gen_sd` 字典，在严格加载（`strict` load）时因磁盘存在“未预期键 `rerun_state_machine`”而报错退出。
3. **veRL `offload_megatron_optimizer` 强行迁移 `HybridDeviceOptimizer`（`HDO`）子优化器状态引发训练崩溃（`volcengine/verl` `#8286` / `#8237`）**：
   - Megatron-LM 的 `HybridDeviceOptimizer`（`HDO`）根据 `optimizer_offload_fraction` 将内部 `sub_optimizers` 划分为常驻 CPU 的子优化器与常驻 GPU 的融合子优化器（如 FusedAdam），并自行管理张量放置。
   - 旧版 `verl/utils/megatron_utils.py` 在 RL Rollout 切换前的 `offload_megatron_optimizer` 中遍历 `hdo.sub_optimizers` 并将其全部状态张量（`exp_avg`、`exp_avg_sq`）强行 `.to("cpu")`，不仅破坏了 HDO 内部融合算子对 GPU 驻留一半状态的设备假设，而且在恢复训练时因设备不匹配直接触发崩溃（issue `#8237`）。

---

## 2. 一手源码证据与实现机制 (First-Hand Code Evidence & Implementation Mechanics)

### 2.1 `DynamicInferenceEngine` 宿主即时拷贝 + 线程安全异步卸载闭环 (`894b5114e3`)

在 `megatron/core/inference/inference_request.py`（blob `4d6acc5920`，lines `1398–1455`）与 `megatron/core/inference/engines/dynamic_engine.py`（blob `5fab3275d8`，lines `583–590, 785–794, 1859–1958`）中：

1. **记账阶段同步执行单次 D2H 拷贝并清零请求侧 GPU 引用**：
   `OffloadedRequestPayload.from_request(request)` 在启动后台任务前，同步将 `request.prompt_tokens` 与 `request.media_tensors` 原地替换为 `.cpu()` / `.detach().cpu()` 宿主副本：
   ```python
   if isinstance(request.prompt_tokens, torch.Tensor):
       request.prompt_tokens = request.prompt_tokens.cpu()
   if request.media_tensors is not None:
       request.media_tensors = {
           name: tensor.detach().cpu()
           for name, tensor in request.media_tensors.items()
           if tensor is not None
       }
   ```
   这样，单次 D2H 拷贝同时服务于后台 `payload` 写出与后续的 ZMQ 响应序列化，且在 `_stage_and_reply` 等待网络 I/O 期间，被保活的 `request` 对象 **持有 `0 B` GPU 显存**！同时在 `_serialize_finished_request` 中将 VLM 请求的 `imgs`、`image_embeddings` 等 8 个大体积输入字段显式置为 `None`。
2. **`asyncio.to_thread` 卸载 I/O + 事件循环单线程独占 ZMQ 发送**：
   ```python
   stage_result = await asyncio.to_thread(
       stager.stage, request.uid, payload,
       finished_metadata=finished_metadata, offload_params=request.offload_params,
   )
   reply = self._serialize_finished_request(request, stage_result)
   self.socket_for_receiving_requests.send_multipart(_engine_reply_frames([reply]))
   ```
   `stager.stage()` 在工作线程池中并发运行，而 `await` 返回后的 `_serialize_finished_request` 与 `socket_for_receiving_requests.send_multipart` 严格回到引擎事件循环线程执行，彻底杜绝跨线程共享 ZMQ socket。
3. **积压峰值监控与 `reset()` / 停机排空守卫**：
   引擎通过 `self._staging_tasks: Set[asyncio.Task]` 与 `self._staging_tasks_peak` 记录在途任务数（通过 `_staging_log_suffix()` 输出 `payload staging: in-flight %d, peak %d`），并在 `reset()` 中强制检查 `if self._staging_tasks: raise RuntimeError(...)`，要求调用方必须先通过 `_drain_staging_tasks()` 排空所有在途写出。

### 2.2 `checkpointing.py` 空闲 `rerun_state_machine` 严格加载兼容 (`e564873e9b`)

在 `megatron/training/checkpointing.py`（blob `7fd3d41591`，lines `2926–2941`）中：

```python
saved_rerun_state = state_dict["rerun_state_machine"]
restore_rerun_state = rerun_state_machine.validate_state_dict(saved_rerun_state)
idle_rerun_state = (
    isinstance(saved_rerun_state, dict)
    and saved_rerun_state.get("state") == RerunState.NOT_RUNNING_YET
)
if restore_rerun_state or idle_rerun_state:
    gen_sd_rerun_state = rerun_state_machine.state_dict(
        data_iterator=None, ckpt_format=ckpt_format, force=True
    )
    ignore_rerun_state = not restore_rerun_state
```

当 `idle_rerun_state` 为真时，仍生成 `gen_sd_rerun_state` 以满足分布式严格加载的 Schema 匹配，但将 `ignore_rerun_state` 设为 `True`（`not restore_rerun_state`），避免恢复出一个虚假的 Rerun 会话。

### 2.3 veRL `offload_megatron_optimizer` 保持 `HDO` 原生状态放置 (`472dd2153a`)

在 `verl/utils/megatron_utils.py`（blob `e47ec921b8`，lines `678–704`）中，`offload_megatron_optimizer` 仍调用 `offload_megatron_copy_params(_opt)` 卸载主参数副本，但通过检查 `not all(hasattr(_opt.optimizer, attr) for attr in ("sub_optimizers", "inner_param_to_orig_param", "state"))` 显式跳过 `HybridDeviceOptimizer`，交由 HDO 自主维护其按 `optimizer_offload_fraction` 切分的 CPU/GPU 子优化器状态。

---

## 3. 合成离线实验与量化结论 (Deterministic Offline Experiment & Quantitative Findings)

通过标准库确定性算例 [`experiments/megatron_payload_staging_verl_hdo_offload/run.py`](../../experiments/megatron_payload_staging_verl_hdo_offload/run.py)（`own-offline`）验证：

| 评估维度 | 修复前基线 (Pre-Fix Baseline) | 修复后实现 (Verified Implementation) | 架构收益 / 关键变化 |
|---|---|---|---|
| **Megatron 动态引擎单步最大延迟 (`12 ms` 解码 + `45 ms` 写出)** | 同步写出阻塞解码步至 **`102.0 ms`** (累计耗时 `414.0 ms`) | 异步 `to_thread` 保持解码步 **`12.0 ms`** (解码耗时 `144.0 ms`) | 解码步峰值延迟加速 **`8.50x`**，节省 **`270.0 ms`** GPU 空转 |
| **在途异步卸载期间已完成请求 GPU 显存驻留 (`peak=4` 任务)** | 若未原地转 CPU，峰值滞留 **`146.0 MiB`** VLM 媒体与 Prompt 显存 | `OffloadedRequestPayload.from_request` 原地 `.cpu()`，滞留 **`0.0 MiB`** | 异步 I/O 与 GPU KV/激活显存池完全解耦 |
| **空闲态 `RerunState.NOT_RUNNING_YET` 严格 Checkpoint 加载** | `unexpected_keys=['rerun_state_machine']`，严格加载 **失败 (`False`)** | 纳入 `gen_sd` 且 `ignore_rerun_state=True`，严格加载 **成功 (`True`)** | 兼容分布式缓存保存计划与严格 Schema 校验 |
| **veRL `HybridDeviceOptimizer` (`fraction=0.5`) 卸载后恢复训练** | GPU 子优化器状态被迁往 `cpu`，下一步 FusedAdam **崩溃 (`True`)** | 仅卸载 `16 GiB` copy params，保留 HDO GPU 子优化器在 `cuda` (**`0` 崩溃**) | 修复 RL Rollout-Training 切换中的 `#8237` 设备失配崩溃 |

---

## 4. 解决方案架构师设计准则与可证伪问题 (Architect Takeaways & Falsifiable Questions)

1. **推理引擎异步后处理必须遵循“先在主线程剥离设备张量、再进工作线程做阻塞 I/O、最后回事件循环线程发网络帧”三段式范式**：这样既消除了慢存储对 ITL（Inter-Token Latency）的尾部尖峰冲击，又避免了 GPU 显存因慢 I/O 产生不可控的排队水位，同时保证了非线程安全通信原语（如 ZMQ socket）的单线程所有权。
2. **RL 框架卸载钩子切勿越权篡改自带异构编排能力的优化器（如 `HDO`）内部状态**：当底层优化器已按 `optimizer_offload_fraction` 静态绑定了 CPU/GPU 双子优化器算子时，外部框架只需卸载外层 FP32 拷贝参数（`copy_params`），强行改写内部 `state` 字典的设备属性会破坏融合内核的设备不变量。
