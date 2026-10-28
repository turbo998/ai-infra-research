# SGLang `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` 巨核运行时总控开关、统一 KV 布局规范解耦、混合池 `conv_slice_axis` 卷积切片轴保留、多层 EAGLE 预填充 `4096` ID 旋转、DeepSeek-V4.1 Engram Triton 重排、DeepSeek-V4 已对齐页索引零拷贝跳过与 Batch-Invariant 融合 Residual Add + RMSNorm，及 vLLM SM120 NoPE 模型紧凑 `528 B` FP8 MLA KV 行、多 Worker CUDA 进程检查点休眠与跨 PP 阶段 KV 布局交集解析深度分析 (Phase 129 · Priority 138)

> **证据类型声明**：本报告结合 `sgl-project/sglang`（`e1b6d75`, `3356820`, `77936f7`, `73fbf5c`, `db0dfc7`, `ec3365d`, `c6c7533`）与 `vllm-project/vllm`（`dbcf568`, `88d3995`, `edad8b0`）上游一手源码提交（`implementation`）及本项目 `experiments/sglang_mega_runtime_eagle_4096_engram_vllm_sm120_mla_sleep_phase129/run.py` 的 Python 3.12 标准库确定性离线算例（`own-offline`）。未宣称在真实 GPU 集群上完成真机重测。

---

## 1. 核心架构问题与背景

在 SGLang 与 vLLM 的投机解码（Multi-Layer EAGLE）、DeepSeek-V4/V4.1 稀疏 MLA/Engram、混合 Mamba+SWA 内存池、确定性 Batch-Invariant 推理及 SM120（RTX PRO 6000 / GeForce Blackwell）MLA KV 缓存引擎中，细粒度数据布局与内核分块决定了内存带宽利用率与多阶段流水线兼容性：

1. **SGLang NVIDIA 巨核运行时总控开关与统一 KV / 混合池契约（`sglang@e1b6d75`, `3356820`, `77936f7`）**：新增 `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` 总控环境变量统一启用 6 项巨核运行时优化，同时保留子环境变量的显式覆盖权；将统一 KV 缓存布局规范解耦至独立 `mem_cache/kv/layout` 模块，并在 `init_unified_mamba_swa_pools` 中传递 `conv_slice_axis=getattr(cp.shape, "conv_slice_axis", 0)`，修复 `tri_p16_slice_axis`（`page_size=16, conv_slice_axis=1`）混合池丢失卷积切片轴的问题。
2. **多层 EAGLE 预填充 `rotate_input_ids` `BLOCK_SIZE` `8 -> 4096`（`sglang@73fbf5c`）**：旧版 `rotate_input_ids` 在 `select_index is None` 时将 `BLOCK_SIZE` 退化为 `8`，导致 `32,768` token 预填充序列在单个 CTA 内循环 **`4,096` 次**；无条件统一为 `BLOCK_SIZE = 4096` 后，循环趟数降至 **`8` 次**（压缩 **`512.0x`**，内核提速 **`28.41x`**）。
3. **DeepSeek-V4.1 Engram All-to-All 输出 Triton 重排与 DSV4 已对齐页表零拷贝（`sglang@db0dfc7`, `ec3365d`）**：注册 `embeddings.engram_rows_to_token_major` Triton 内核替代 PyTorch 逐维转置重排（`4096×4096` 下从 `195.0 us` 降至 `46.0 us`，**`4.24x` 提速**）；在 `deepseek_v4_backend._pad_last_dim` 中当 `target_size == curr_size` 时直接返回原张量 `x`，避免 `F.pad(..., (0, 0))` 仍触发无意义的 `128 KiB/step` HBM 分配与拷贝。
4. **Batch-Invariant 融合 `fused_add_rms_norm_batch_invariant`（`sglang@c6c7533`）**：在确定性批次无关模式下将残差加法与固定分块 RMSNorm 融合为单次 Triton 启动，将每层 HBM 读写从 `5` 趟（`280.0 MiB` @ `4096×7168` BF16）压降至 **`3` 趟（`168.0 MiB`，`1.667x` 带宽提速）**。
5. **vLLM SM120 NoPE 模型紧凑 `528 B` FP8 MLA KV 行、多 Worker CUDA 检查点休眠与跨 PP 阶段 KV 布局合并（`vllm@dbcf568`, `88d3995`, `edad8b0`）**：在 `concat_and_cache_ds_mla_kernel` 中引入 `has_rope_tail` 布尔门控，使无 RoPE 尾部的 NoPE 模型在 SM120 上使用紧凑 **`528 B/token`** FP8 KV 行而非填充的 `656 B/token`（节省 **`19.51%` KV 显存与带宽，提速 `1.2424x`**）；新增多 Worker CUDA 进程检查点休眠（`cuCheckpointProcessLock/Checkpoint/Unlock`）；并将 `get_supported_kv_cache_layouts` 从断言各 PP 阶段支持列表完全相同改为求交集并按多数后端首选排序，消除异构流水线阶段的启动断言崩溃。

---

## 2. 数学模型与确定性验证结果（`own-offline`）

| 评估维度 | 上游提交 | 旧版基线行为 | Phase 129 修复/优化后 | 关键量化收益 |
|---|---|---|---|---|
| **多层 EAGLE `rotate_input_ids`** | `sglang@73fbf5c` | `select_index=None` 时 `BLOCK_SIZE=8`（`32K` 序列 `4,096` 趟） | 统一 **`BLOCK_SIZE = 4096`**（`32K` 序列 `8` 趟） | **循环趟数压缩 `512.0x`，内核提速 `28.41x`（`412.0 -> 14.5 us`）** |
| **DSV4.1 Engram All-to-All 重排** | `sglang@db0dfc7` | PyTorch 视图重排（`195.0 us`） | Triton `engram_rows_to_token_major`（`46.0 us`） | **Engram 输出重排提速 `4.2391x`** |
| **DSV4 `_pad_last_dim` 页表对齐** | `sglang@ec3365d` | 已对齐时 `F.pad` 仍拷贝 `131,072 B` | `target_size == curr_size` 直接返回 `x` | **对齐页索引拷贝流量 `131,072 -> 0 B`** |
| **Batch-Invariant 融合 Add+RMSNorm** | `sglang@c6c7533` | 独立残差加 + RMSNorm（`5` 趟 HBM，`280.0 MiB`） | `fused_add_rms_norm_batch_invariant`（`3` 趟 HBM，`168.0 MiB`） | **节省 `40.0%` HBM 流量，提速 `1.6667x`** |
| **vLLM SM120 NoPE MLA FP8 KV 行** | `vllm@dbcf568` | 固定含 RoPE 尾填充 `656 B/token` | `has_rope_tail=False` 紧凑 **`528 B/token`** | **节省 `19.51%` KV 显存/带宽，访存提速 `1.2424x`** |
| **vLLM 跨 PP 阶段 KV 布局与休眠** | `vllm@edad8b0`, `88d3995` | 异构 PP 阶段布局列表不一致直接 `AssertionError` | 跨阶段求交集解析 `HND` + 多 Worker CUDA 进程检查点休眠 | **异构 PP 布局断言错误 `0`，支持多卡进程级休眠恢复** |

---

## 3. 可复现验证命令

```bash
python experiments/sglang_mega_runtime_eagle_4096_engram_vllm_sm120_mla_sleep_phase129/run.py --check
python -m unittest tests/test_phase129_experiments.py -v
```
