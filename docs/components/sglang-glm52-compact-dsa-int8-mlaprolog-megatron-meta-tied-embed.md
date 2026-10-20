# SGLang GLM-5.2 紧凑 DSA 索引池分配与动态 INT8 MLAProlog 及 Megatron-LM Meta-Device 绑定词表同步深度解析

> **English Version**: [sglang-glm52-compact-dsa-int8-mlaprolog-megatron-meta-tied-embed.en.md](sglang-glm52-compact-dsa-int8-mlaprolog-megatron-meta-tied-embed.en.md) | **返回索引**: [README.md](../../README.md) | **7-Card 评测报告**: [ai-infra-phase33-benchmark-cards.html](../reports/ai-infra-phase33-benchmark-cards.html)

## 1. 一手源码变更矩阵（2026-10-20）

| 组件 | 仓库与 Commit SHA | 发布日期 | 核心文件与函数 | 证据级别 |
| :--- | :--- | :--- | :--- | :--- |
| **SGLang GLM-5.2 Compact DSA Indexer Pool Allocation** | `sgl-project/sglang` [`3bf4cbc6cf18b539d88c08c17cb97d82ec82c951`](https://github.com/sgl-project/sglang/commit/3bf4cbc6cf18b539d88c08c17cb97d82ec82c951) (`#44436`) | `2026-10-20` | `python/sglang/srt/mem_cache/kv_cache_configurator.py::_build_ascend_mla_kv_pool`, `python/sglang/srt/model_executor/pool_configurator.py::_compute_dsa_indexer_cell_size` | `implementation` + `own-offline` |
| **SGLang GLM-5.2 Dynamic INT8 MLAProlog & BF16 Indexer Adaptation** | `sgl-project/sglang` [`10d5293546faf6469ebe7b924ecf5d9519e14c6f`](https://github.com/sgl-project/sglang/commit/10d5293546faf6469ebe7b924ecf5d9519e14c6f) (`#43469`) | `2026-10-20` | `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py::npu_mla_preprocess` | `implementation` + `own-offline` |
| **Megatron-LM Tied Embedding Materialization & Sync under Meta-Device Init** | `NVIDIA/Megatron-LM` [`fa322ce62077fa34ac6927fdc66ab2c8b88fcc30`](https://github.com/NVIDIA/Megatron-LM/commit/fa322ce62077fa34ac6927fdc66ab2c8b88fcc30) (`#8010`) | `2026-10-20` | `megatron/core/models/common/language_module/language_module.py::LanguageModule._materialize_shared_embedding_weight` | `implementation` + `own-offline` |

---

## 2. SGLang GLM-5.2-W4A8C8 紧凑 DSA Indexer 内存池分配

### 2.1 `"full"` 与 `"shared"` 混合 DSA 层的冗余索引槽分配
在 GLM-5.2 等采用动态稀疏注意力（DSA）的混合架构中，模型的 `hf_config.indexer_types` 由少量的 `"full"` 层（独立计算并写入 `index_k_with_scale_buffer`）与大量的 `"shared"` 层（直接复用前序 `"full"` 层的稀疏索引，自身从不读写 indexer 缓冲区）组成。

在 Commit `3bf4cbc6cf18` 之前，`_build_ascend_mla_kv_pool` 将紧凑索引布局 `use_compact_indexer_layout` 硬编码门控在 `is_arch35` 上，同时 `_compute_dsa_indexer_cell_size` 在 `if not is_npu_arch35(): allocate_all_layers = True` 下强制按全模型层数预算每 token 的 indexer 字节数。结果导致非 arch35 NPU 为从不访问 indexer 的 `"shared"` 层白白分配了物理内存，挤占了宝贵的 MLA KV Cache 容量。

### 2.2 解除 `is_arch35` 门控后的容量增益定量推导
Commit `3bf4cbc6cf18` 移除了上述两处 `is_arch35` 限制，使池预算计算与池物理分配统一遵循 `use_compact_indexer_layout = is_dsa_model and _should_elide_dsa_index_k(...)`：
- 以 60 层 GLM-5.2（其中 15 层 `"full"`、45 层 `"shared"`，MLA KV 维度 $576$，Indexer 维度 $128$，BF16 2B 存储）为例：
  - 每 token Indexer 内存从 $60 \times 128 \times 2 = 15{,}360\,\text{B}$ 降至 $15 \times 128 \times 2 = 3{,}840\,\text{B}$（**削减 $75.0\%$**）。
  - 单 token 总 KV+Indexer 单元字节从 $84{,}480\,\text{B}$ 降至 $72{,}960\,\text{B}$。
  - 在 $24.0\,\text{GiB}$ KV 池预算下，最大可容纳 KV token 数从 **$305{,}040$ 提升至 $353{,}204$（净增 $+15.79\%$）**；在官方单测的 6 层（2 full / 4 shared）`$1\text{ MB}$` 预算下，容量从 `118` tokens 提升至 `134` tokens（`+13.56%`）。

---

## 3. SGLang GLM-5.2 动态 INT8 `MLAProlog` 算子分派与 BF16 Indexer 适配

### 3.1 为什么仅凭 `weight.dtype == torch.int8` 无法分派 `MLAProlog`？
在 GLM-5.2 W4A8/W8A8 推理中，静态 W8A8（`NPUW8A8Int8LinearMethod`）与动态 W8A8（`NPUW8A8Int8DynamicLinearMethod`）的投影权重 `qkv_a_proj.weight` 均为 `int8`，但只有动态量化方案才满足 `MLAProlog` 融合算子 `weight_quant_mode = 2` 的输入契约（静态量化缺少动态 per-token scale 布局）。

Commit `10d5293546fa`（`#43469`）在 `mla_preprocess.py` 初始化阶段直接检查 `(self.qkv_a_proj, self.q_b_proj)` 的量化 kernel 实例类型：
- 全为 `NPUW8A8Int8DynamicLinearMethod` $\to$ `weight_quant_mode = 2`；
- 全为 `NPUMXFP8LinearMethod` $\to$ `weight_quant_mode = 3`；
- 浮点 FP16/BF16 $\to$ `weight_quant_mode = 1`；静态 INT8 则保持 `None`（不误触 `MLAProlog`）。

### 3.2 分离式 Decode 的 `w_kc` 保留与 BF16 Indexer `q_lora` 重算
该提交同时修复了动态 INT8 `MLAProlog` 下的两个关键张量契约边界：
1. **分离式 Decode 保留 `w_kc`**：当 `disaggregation_mode == "decode"` 且 `weight_quant_mode == 2` 时，禁止执行 `m.w_kc.untyped_storage().resize_(0)`，因为动态 INT8 路径仍需访问 `w_kc`。
2. **BF16 Indexer 的未量化 `q_lora` 重算**：动态 INT8 `MLAProlog` 返回的 `q_lora` 已附带 INT8 动态量化缩放，若当前层的 DSA Indexer（`m.indexer.wq_b`）仍为 BF16 算子，则显式通过 `q = m.fused_qkv_a_proj_with_mqa(hidden_states)[0][..., :m.q_lora_rank]` 与 `q_lora = m.q_a_layernorm(q)` 重算浮点 `q_lora` 并置 `dynamic_scale = None`。

---

## 4. Megatron-LM `init_model_with_meta_device` 下的绑定词表同步（`#8010`）

### 4.1 流水线首尾 Stage 在 Meta-Device 初始化下的失步缺陷
在 Megatron-LM 流水线并行（PP $\ge 2$）且启用 `share_embeddings_and_output_weights=True` 时，首 Stage（`pre_process=True`）与末 Stage（`post_process=True`）必须在初始化末尾通过 `embd_group` 执行 `all_reduce` 以保证词表权重与 LM Head 权重完全一致。

然而，旧代码在 `setup_embeddings_and_output_layer` 中写了 `if self._is_in_embd_group() and not self.config.init_model_with_meta_device:`——当启用 `init_model_with_meta_device=True` 时直接跳过了 `all_reduce`，且末 Stage 上的 Transformer Engine (TE) LM Head 绑定副本仍停留在 `meta` 设备上，导致要么触发 meta tensor CUDA 异常，要么首尾 Stage 词表权重失步。

### 4.2 `_materialize_shared_embedding_weight()` 的零 NaN 物化与属性继承
Commit `fa322ce62077`（`#8010`）新增了 `_materialize_shared_embedding_weight()` 并在 `all_reduce` 前调用：
1. **区分有无 `reset_parameters` 的物化策略**：定位持有 `weight` 的子模块 `owner`。若 `hasattr(owner, "reset_parameters")`（如 TE LM Head），使用 `torch.empty_like` 分配 CUDA 显存并调用 `owner.reset_parameters()`；若没有 `reset_parameters`（如 ModelOpt CPU checkpoint 加载流），必须使用 **`torch.zeros_like`** 而非 `empty_like`，防止未初始化显存中的随机 `NaN` 参与 `embd_group` `all_reduce` 污染首 Stage 权重。
2. **跨 TE Parameter 替换保留自定义属性**：TE 的 `reset_parameters()` 会创建新的 `nn.Parameter` 实例替换旧对象；代码遍历 `vars(old_param)` 将 `tensor_model_parallel`、`allreduce` 等 Megatron 自定义属性完整迁移至 `new_param`。

---

## 5. 可复现实验

```bash
python experiments/sglang_glm52_dsa_mlaprolog_megatron_meta_embed/run.py --check
```
