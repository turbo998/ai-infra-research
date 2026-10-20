# TorchTitan Cuspy CUPTI CUDA Graph 节点注解与调用栈捕获、Muse Spark `muse_spark_attention_qk` 融合 RoPE 编译域与 TransformerEngine JAX 单 NVLink 域融合 EP Prepare+Dispatch

> **English Version**: [torchtitan-cuspy-cupti-graph-muse-spark-rope-norm-te-nvl-fused-ep.en.md](torchtitan-cuspy-cupti-graph-muse-spark-rope-norm-te-nvl-fused-ep.en.md)  
> **证据等级**：`implementation`（上游固定 Commit 源码核验）+ `own-offline`（`experiments/torchtitan_cuspy_graph_muse_rope_te_ep/run.py --check`）

## 1. 一手源码定位与核心架构机制

1. **TorchTitan Cuspy CUPTI CUDA Graph 捕获注解与 Python 栈导出（`pytorch/torchtitan` commit `b93d5c0cf518b4404ceab998adc529fd60ebf301`, `torchtitan/distributed/cuda_graph.py` & `torchtitan/observability/profiler.py`）**：
   - 新增 `get_cuspy_cuda_graph_annotation_config(enable_event_node_ids, record_py_stacks)` 并将 `annotation_config` 透传至 `torch.cuda.graph`（涵盖 `forward_backward`、`optimizer` 与 `validation` 图包装器）；
   - 设置 `"backend": "cupti"`：利用 CUPTI 节点创建回调替代静态图边遍历（edge-walking），捕获流进入 capture 前已创建的先导节点；
   - 动态键控：默认 `"key_by": "auto"`（通过 CUPTI `sourceGraphNodeId` 关联 capture 图），而当开启 `enable_event_node_ids=True`（如 `NCCL_GRAPH_MIXING_SUPPORT` 下的 NCCL 事件记录节点）时切换为 `"key_by": "exec"` 以按执行图 ID 解析；
   - 当 `record_py_stacks=True` 时，将每个图内核的 Python 启动栈以 64 位复合键 `(graph_id << 32) | graph_node_id` 导出至 `rank{rank}_cuda_graph_py_stacks.json.gz`。
2. **TorchTitan Muse Spark `muse_spark_attention_qk` 统一编译域消除 Helion RoPE 覆盖（`pytorch/torchtitan` commit `8f8dbfb50caacdaa2e1af3ebb1d1ee67f57b9a87`, `torchtitan/models/muse_spark/model.py` & `torchtitan_recipes/models/muse_spark.py`）**：
   - 将逐头 Q/K RMSNorm、查询缩放、位置温度缩放（`clamp_min(log((positions.float() + 1.0) / floor_scale) * query_scale + 1.0, 1.0)`）以及基于 `int32` 双 BF16 实数算术的 ComplexRoPE 合并进单一 `@local_compile` 区域 `muse_spark_attention_qk`（`recompile_limit=16`, `batch_invariant=False`）；
   - 移除外部 `_MUSE_SPARK_HELION_ROPE_OVERRIDE` 及其每次调用附带的 3 个越界检查内核。作者在 GB300 上的实测（`author-benchmark`）显示：输出与梯度在 BF16 下与 Helion 逐位一致（bitwise identical），Decode 延迟在 `256 seqs` 下从 `39.0 us` 降至 **`10.4 us`（`3.7500x` 加速）**，在 `1 seq` 下从 `22.6 us` 降至 **`8.4 us`（`2.6905x` 加速）**，训练+预填+解码（含 RoPE 与无 RoPE 层）共生成 `8` 张图，严格处于 `recompile_limit=16` 预算内。
3. **TransformerEngine JAX 单 NVLink 域融合 EP Prepare+Dispatch（`NVIDIA/TransformerEngine` commit `b7db08fd73129f5af71ff2ee0eaddacd3ff1a751`, `transformer_engine/jax/cpp_extensions/ep.py` & `docs/envvars.rst`）**：
   - 在 `NVTE_EP_FUSED_PREPARE_DISPATCH=1` 下引入 `ep_prepare_and_dispatch_fwd`（底层绑定 `EpPrepareAndDispatchHandler`），将 NCCL EP count 模式下原本分离的 `ep_prepare` 与 `ep_dispatch_fwd` 两次 XLA FFI 调用合并为 1 次，并将 `handle_mem` 作为别名输出（aliased dispatch output）返回 `(recv_tokens, recv_topk_weights, handle_mem)`；
   - 该融合路径仅在 EP 位于单一 NVLink 域内受支持，且专家块内部的接收行顺序允许与非融合路径不同（专家 MLP 满足行置换等变性，Combine 归约后数值完全一致）。

## 2. 确定性离线实验核验（`own-offline`）

运行 `python experiments\torchtitan_cuspy_graph_muse_rope_te_ep\run.py --check`：
- **Cuspy CUPTI 图注解**：在 24 节点混合图（4 个预捕获流节点 + 16 个常规算子 + 4 个 NCCL 事件记录节点）中，将节点覆盖从边遍历的 `20/24` 提升至 **`24/24`**，`key_by="exec"` 解析全部 **`4/4`** 个 NCCL 事件节点，64 位栈键双射无损；
- **Muse Spark `muse_spark_attention_qk`**：复现 `256 seqs` Decode **`3.75x`** 加速与 `1 seq` **`2.6905x`** 加速，消除 3 个 Helion 边界检查算子，跨 6 组位置温度与 RoPE 旋转的最大数值偏差为 **`0.0`**；
- **TE JAX 融合 NVL EP**：XLA FFI 调用次数减少 **`50.0%`（`2 -> 1`）**，在专家内行乱序到达下 Combine 归约最大绝对误差 `< 1e-12`（实测 `0.0`）。

## 3. 架构师设计练习与可证伪问题

- **练习**：为何启用 `profiler.cuspy` 并在首个 CUDA Graph 捕获时开启 CUPTI 订阅后，同一进程内不能再挂载 Nsight Systems (`nsys`) 或 `dynolog`？
  - **解答**：CUDA Driver API 的 CUPTI Callback/Activity 订阅在单进程内具有排他性，Cuspy 在首次图捕获时建立并持有进程级 CUPTI 订阅直至训练结束，其它 CUPTI 采集器再附加会因订阅冲突失败。
- **可证伪问题**：若跨机 IB/RoCE 域强行开启 `NVTE_EP_FUSED_PREPARE_DISPATCH=1`，为何单 NVLink 域假设会失效？
