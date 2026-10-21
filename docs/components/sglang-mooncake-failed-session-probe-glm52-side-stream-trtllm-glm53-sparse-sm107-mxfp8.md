# SGLang Mooncake 失败会话指数退避探测与 NIXL UCX 绑定、GLM-5.2 侧流归并及 TensorRT-LLM GLM-5.3 `glm_kpool` 稀疏注意力深挖

> **一手上游证据（2026-10-21）**：
> - `sgl-project/sglang` PR `#45338` (`be65391abd01f711cdc443a3fe8f508b0c960357`)：`[PD] Default-enable Mooncake failed-session probe with backoff and idle pruning`
> - `sgl-project/sglang` PR `#43482` (`2e7c9d851c868678ddfcd93b4e8000cd23e1f779`)：`[PD] Support per-GPU UCX device selection in NIXL disaggregation`
> - `sgl-project/sglang` PR `#45349` (`f7b8c6fe404ca4f29c3188cb50692f7735d45a99`)：`[GLM-5.2] Record DSA side-stream work right before its join`
> - `sgl-project/sglang` PR `#44929` (`da7e3f1c125be0f1ef2101c474b679296acdb325`)：`[UMBP] Accept packed draft layer mappings in UMBPDirectLinker`
> - `NVIDIA/TensorRT-LLM` PR `#20454` (`ef913d39b3b5ba488dc0972bd400e52b41899767`)：`[https://nvbugs/6068971][perf] Speed up glm_kpool sparse attention for GLM-5.3-Flash`
> - `NVIDIA/TensorRT-LLM` PR `#19953` (`4636b6311f55ec4ffe16d44fba3601f41638ec42`)：`[https://nvbugs/6039511][feat] Enable CUTLASS MXFP8 MoE on SM107`

---

## 1. SGLang Mooncake 失败会话指数退避探测与 NIXL UCX 多网卡精准绑定（PR `#45338` / `#43482`）

### 1.1 Mooncake 失败 Decode 会话指数退避探测与空闲驱逐（PR `#45338`）
在 Prefill-Decode（PD）分离集群中，一旦某个 Decode 实例因瞬时网络抖动被 Prefill 节点列入 `failed_sessions` 黑名单，若不探测恢复，该 Decode 节点将永久丢失流量；但若以固定 `30s` 周期无限探测已下线的僵尸会话，又会在控制面上产生持续风暴。

PR `#45338` 默认开启 `SGLANG_ENABLE_FAILED_SESSION_PROBE = True`，并在 `mooncake/conn.py` 中引入 `FailedSessionProbe(last_requested, next_probe_at, interval)` 状态机：
- **指数退避（Exponential Backoff）**：初始探测间隔 `SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = 30.0s`，每次探测失败后翻倍 `interval = min(interval * 2.0, 300.0s)`（即在 `t = 30s, 90s, 210s, 450s` 触发探测）。
- **空闲超期驱逐（Idle Pruning）**：当 `now - last_requested >= SGLANG_FAILED_SESSION_IDLE_TIMEOUT_S (600.0s)`（即路由器已不再向该会话派发请求）时，在 `t = 750s` 自动从 `failed_sessions` 与 `failed_session_probes` 中彻底移除，将已下线 Decode 节点 24 小时内的无效健康检查次数从 **`2880` 次削减至 `4` 次（降低 `99.86%`）**。
- **P↔D 角色切换前置校验**：在 `__init__` 中无条件校验 `0 < interval <= max_interval` 与 `idle_timeout > 0`，防止节点从 Decode 动态切换为 Prefill 时因非法环境变量崩溃。

### 1.2 NIXL UCX 按 GPU 粒度 HCA 设备绑定（PR `#43482`）
- 在多 NIC 拓扑下，PR `#43482` 在 `nixl/conn.py` 中新增 `_resolve_ucx_device_list(ib_device, gpu_id)`，复用 `get_ib_devices_for_gpu` 提取当前 GPU 对应的 HCA 列表，严格校验不含 `:port` 后缀（因为 NIXL UCX 内部会自动追加端口 `1`），并按 NIXL UCX 要求的字面分隔符 `", "`（逗号加空格）注入 `backend_params["device_list"]`，同时拦截与 `SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS` 的冲突配置。

---

## 2. GLM-5.2 DSA 侧流（Side-Stream）紧邻归并录制与 UMBP 打包草稿层映射（PR `#45349` / `#44929`）

1. **GLM-5.2 DSA CUDA Graph 单流重放归并顺序（PR `#45349`）**：
   - 在 `dsa_indexer.py`（`_fused_q_prepare_and_store`）与 `forward_mla.py`（`forward_absorb_prepare`）中，SGLang 使用 `self.alt_stream` 并行执行 `wq_b(q_lora)` 或 `q_b_proj_forward(q)`，同时在主流执行 `wk_weights_proj(x)` 或 `self.indexer(...)`。
   - 此前代码先录制 `with torch.cuda.stream(self.alt_stream):` 分支，再录制主流算子，最后调用 `current_stream.wait_stream(self.alt_stream)`。在 CUDA Graph 捕获下，这会导致图拓扑过早分叉，破坏单流重放（single-stream replay）假设。
   - PR `#45349` 将 `with torch.cuda.stream(self.alt_stream):` 移至主流大算子（`wk_weights_proj` / `indexer`）**紧后方、紧贴 `current_stream.wait_stream(self.alt_stream)` 之前**录制，既保留了双流并发语义，又确保 CUDA Graph 重放时归并点（join）严格留在主通道上。
2. **UMBP Direct Linker 打包投机草稿层（Packed Draft Layers）映射（PR `#44929`）**：
   - 在投机解码（如 MTP / Eagle）中，单个逻辑草稿层可能对应多个物理 KV 缓冲区索引（`layer_mapping: dict[int, int | Sequence[int]]`）。PR `#44929` 令 `_ordered_layers` 与 `_range_items` 支持一对多映射（如 `{60: [0, 1], 61: 2}`），在校验物理缓冲区 `[0..pool_layer_count-1]` 双射无重复的同时，按 `dict.fromkeys` 去重保持逻辑层顺序。

---

## 3. TensorRT-LLM GLM-5.3-Flash `glm_kpool` 稀疏注意力加速与 Rubin `SM107` MXFP8 MoE（PR `#20454` / `#19953`）

`NVIDIA/TensorRT-LLM` PR `#20454` 针对 `GLM-5.3-Flash` 的 `glm_kpool` 稀疏 MLA 后端落地了三项核心性能优化：

1. **单次启动融合生成状态追加核函数 `_append_paged_state_kernel`**：
   - Decode 阶段（`request_ids is None`）原本需要多次算子启动（计算 `page = pos // tokens_per_block`、`offset = pos % tokens_per_block`、量化 latent、分别写入 `latent_pool` 与 `index_pool`）。
   - 新增 Triton 核函数 `_append_paged_state_kernel` 在 **1 次 kernel launch**（从 ~`6` 次降至 `1` 次）内同时完成页表寻址、FP8 E4M3 在线缩放量化（`tl.clamp(latent * kv_scale_orig_quant, -fp8_max, fp8_max)`）以及 `latent_pool`（576 维：512 Nope + 64 RoPE）与 `index_pool`（`3 * head_dim`）的双池写入。
2. **投机验证 2D 位置同池去重更新（`_kpool_update_kernel` `ROWS_PER_TABLE > 1`）**：
   - 在 Draft Verification 阶段，每个请求传入 `T` 个连续位置 `positions[N, T]`。若同一请求的多个候选 token 落在同一个 `KPOOL`（大小 `64`）块内，按顺序逐列更新会产生冗余写。
   - PR `#20454` 支持直接传入 2D `positions[N, T]`：当 `b % ROWS_PER_TABLE < ROWS_PER_TABLE - 1` 且 `POS[b + 1] // KPOOL == pos // KPOOL` 时，当前行直接 `return` 跳过，仅由该池内最后一个可见位置一次性完成池键（pool key）刷新（例如 2 个请求 × 4 候选 token = 8 行中，仅需执行 `3` 次池键更新，减少 **`62.5%`** 冗余计算）。
3. **原生 TRTLLM-GEN FP8 稀疏注意力解码（高达 `128` 查询头）与主机端预乘 `bmm1_scale`**：
   - 将 `can_run_native` 从仅支持小批量 BF16（`num_heads <= 16`）扩展至支持 **FP8 E4M3 KV Cache（`1 <= num_heads <= 128, head_dim == 512, tokens_per_block == 64, topk % 64 == 0`）**，彻底省去 FlashMLA 回退路径中 `[tokens, topk, 512]` 的 BF16 稀疏收集与反量化（`gather_fp8_kv_rows`）。
   - 同时在 CUDA Graph 捕获前一次性在主机端预计算 `self._native_bmm1_scale = self.softmax_scale * float(self.kv_scale_quant_orig.item())`，消除每次调用时的设备端标量乘法。
4. **Rubin `SM107` CUTLASS MXFP8 MoE 解锁（PR `#19953`）**：
   - 在 `CutlassFusedMoE` 中将 `QuantAlgo.MXFP8`（W8A8 E4M3×E4M3 + UE8M0 `1x32` 块缩放）的 `sm_constraint` 从 `{100, 103}` 扩展至 `{100, 103, 107}`。

---

## 4. 确定性复现实验验证

```bash
python experiments/sglang_mooncake_probe_glm52_trtllm_glm53_sparse/run.py --check
```
