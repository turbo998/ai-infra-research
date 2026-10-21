# TransformerEngine KDA 线性注意力、TorchTitan Cuspy 剖析器与 Kimi `rms_norm_eps` 修复、LMCache/Dynamo 解耦及 Mooncake TENT 诊断

> **关联上游提交（一手源码）**：
> - [`NVIDIA/TransformerEngine@b9715e5`](https://github.com/NVIDIA/TransformerEngine/commit/b9715e55a9161c703da3b8cd1d8117c2257f356e) (`#3548`, 2026-10-14) — *KDA attention (`KimiDeltaAttention` via cuDNN FROST on SM100/SM103/SM107)*
> - [`NVIDIA/TransformerEngine@c3081a5`](https://github.com/NVIDIA/TransformerEngine/commit/c3081a5f2db9de31eab51689a25c70ea48068aa6) (`#3470`, 2026-10-14) — *[All] Guard THD learnable dSink on older cuDNN*
> - [`pytorch/torchtitan@789740c`](https://github.com/pytorch/torchtitan/commit/789740c02578d288ee6e5a5651c6f96162bf9b84) (`#5525`), [`35f4a2e`](https://github.com/pytorch/torchtitan/commit/35f4a2e166586302ae6da957656a839759b20121) (`#5450`), [`d06d37f`](https://github.com/pytorch/torchtitan/commit/d06d37f53cb206ad008fb5869ae36a5c46adba50) (`#5548`, 2026-10-20) — *Cuspy profiler backend, `torch.accelerator` migration, and Kimi K2.x `rms_norm_eps=1e-5` fix*
> - [`LMCache/LMCache@eb2f3df`](https://github.com/LMCache/LMCache/commit/eb2f3df13f6e49e8a92c464f161dc3c456da3b42) (`#5925`), [`e6534c6`](https://github.com/LMCache/LMCache/commit/e6534c635e747b2c66ac2cb5342236d0ce62f312) (`#5939`, 2026-10-21) & [`ai-dynamo/dynamo@45a4e56`](https://github.com/ai-dynamo/dynamo/commit/45a4e56a5f4dd4111d937ac36f38ac669677468d) (`#16324`, 2026-10-21) — *Isolate `lmcache.inprocess` & retire KVBM in favor of `dynamo.vllm.pd_connector`*
> - [`kvcache-ai/Mooncake@7540f58`](https://github.com/kvcache-ai/Mooncake/commit/7540f5826ed32cb755b1db027a2c609adb53f024) (`#4154`, 2026-10-21) & [`8952021`](https://github.com/kvcache-ai/Mooncake/commit/89520212377af7114bbba4177021ba092d78ed11) (`#4611`, 2026-10-20) — *[TENT] Add diagnostic core/CLI & preserve `int64`/`bool` in `typed_ragged` with empty rows*

---

## 1. TransformerEngine `KimiDeltaAttention` 与 cuDNN THD Guard 收窄

### 1.1 KDA 逐通道衰减非交换递推（`#3548`）
`KimiDeltaAttention` 将 Gated DeltaNet 的标量衰减推广为逐 Key 通道向量衰减 $\alpha_t = \exp(g_t) \in (0, 1]^{d_{qk}}$，同时保持标量写入强度 $\beta_t$：
$$S_t = (I - \beta_t k_t k_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1} + \beta_t k_t v_t^\top, \quad o_t = \text{scale} \cdot q_t S_t$$
- **非交换性核心特征**：与标量衰减不同，当 $g_t$ 在不同通道取值不同时，$\mathrm{Diag}(\alpha_t)$ 与 $(I - \beta_t k_t k_t^\top)$ **不可交换**，必须先对状态执行通道衰减，再读取已衰减状态计算擦除项；当 $g_t$ 退化为单头常数时，两种顺序完全等价（差值为 `0.0`）。
- **cuDNN FROST 引擎适配**：在 SM100/SM103/SM107 上通过 `cudnn.linear_attention.ops.kimi_delta_attention` 驱动，支持 `head_dim in (64, 128)`，状态张量采用 cuDNN 转置约定 `[batch, heads, v_head_dim, qk_head_dim]`。

### 1.2 Pre-cuDNN 9.26.0 THD Learnable Softmax 反向守卫收窄（`#3470`）
`TransformerEngine@c3081a5` 将 `cudnn_runtime_version < 92600` 下 THD `NVTE_LEARNABLE_SOFTMAX` 反向拒绝条件收窄为 `head_dim_v not in {64, 128, 256}`，恢复了已正确索引 Ragged Stats 的 64/128/256 特化内核。

---

## 2. TorchTitan Kimi K2.x `rms_norm_eps` 修复、Cuspy 剖析器与 `torch.accelerator`

- **Kimi K2.x 解码层 `rms_norm_eps=1e-5` 修复（`#5548`）**：`moonlight-16B-A3B`、`Kimi-VL-A3B` 与 `Kimi-K2.5` 复用 DeepSeek-V3 的 `build_mla_moe_layers` 时此前硬编码了 `eps=1e-6`，而官方 HF 权重配置的 `attention_norm` / `ffn_norm` 为 `1e-5`（MLA 内部 `q/kv` 归一化保持 `1e-6`）。在真实 Embedding 行上，硬编码 `1e-6` 导致第 0 层归一化输出产生中位数 **`1.0%`（Moonlight）/ `1.2%`（Kimi-VL）/ `4.8%`（K2.5）**、最高达 **`34%–40%`** 的数值偏差！
- **Cuspy 进程内 CUPTI 剖析器（`#5525`）**：新增 `CuspyProfilerConfig`，支持 CUPTI PM 硬件采样（`sm__cycles_active.avg` 等）、功耗/频率/温度计数器、CUDA Graph 依赖边与 `pftrace`（Perfetto protobuf）压缩轨迹导出。
- **`torch.accelerator` 统一流与事件（`#5450`）**：将 EMA CPU Offload（通过 `params_read` 事件仅阻塞计算流至 H2D+lerp 结束，允许尾部 D2H 与下一步计算重叠）、RL 权重异步暂存与 `PagedStashManager` 全部迁移至 `torch.accelerator` / `torch.Stream` / `torch.Event`。

---

## 3. LMCache / Dynamo 架构瘦身与 Mooncake TENT 诊断 / Ragged 精度保持

- **LMCache & Dynamo（`#5925` / `#5939` / `#16324`）**：LMCache 将旧版进程内实现彻底隔离至 `lmcache.inprocess`（含 `python -m lmcache.inprocess.standalone`）；Dynamo 从 CI/Release 与 `Cargo.toml` 中移除 7 个 `kvbm-*` Crate 与 Wheel，同时保留 `dynamo.vllm.pd_connector` 中的 `PdConnector`。
- **Mooncake TENT 离线诊断与脱敏引擎（`#4154`）**：新增 `tent diagnostics version [--json]`（独立于 `tent_interface`，绝不初始化 RDMA/GPU 运行时），定义退出码优先级 **`4`（内部错误 `tool.internal`）> `3`（采集器异常 `engine.process` / `tool.probe_missing`）> `1`（检查失败）> `0`**（`2` 为 CLI 参数错误），并通过 `Redactor::sanitizeText()` 在 `kStrict` 模式下跨结构化字段与自由日志文本维护一致的伪名映射（如 `<host-1>`、`<ip-2>`）及密钥/rkey 脱敏。
- **Mooncake `typed_ragged` 空行精度修复（`#4611`）**：NumPy 默认将空列表 `[]` 推断为 `float64`，导致 `[[1, 2], [], [2**53 + 1]]` 被提升为 `float64` 并丢失精度为 `9007199254740992.0`。`_typed_ragged_row_dtypes` 忽略空 `list/tuple` 行，完整保持 `int64` 精度与 `bool` 行的 `msgpack_ragged` 编码。

---

## 4. 确定性复现实验与验证

```bash
python experiments/te-kda-attention-torchtitan-cuspy-kimi-eps-lmcache-inprocess-mooncake-tent/run.py --check
```
