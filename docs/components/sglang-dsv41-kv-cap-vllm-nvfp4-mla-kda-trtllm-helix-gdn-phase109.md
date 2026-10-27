# SGLang DSV4.1 物理 KV 容量、vLLM NVFP4 MLA 聚集与 GLM-5.3 KDA 融合及 TRT-LLM Helix/GDN 边界分析 (Phase 109 · Priority 118)

> **English Version**: [sglang-dsv41-kv-cap-vllm-nvfp4-mla-kda-trtllm-helix-gdn-phase109.en.md](sglang-dsv41-kv-cap-vllm-nvfp4-mla-kda-trtllm-helix-gdn-phase109.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/sglang_dsv41_kv_cap_vllm_nvfp4_mla_kda_trtllm_helix_phase109/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`sgl-project/sglang@dea05ca16e` (`#46718`)、`7f8ad67373` (`#46144`) 与 `20ff40cb84` (`#41275`)**：
   - **DeepSeek-V4 捕获桩、按角色自动拓扑推导与 DSV4.1 物理 KV 容量对齐**：`#46718` 在 `deepseek_v4.py` 中新增 `_write_tail_source_kv_capture_stub`，在 CUDA Graph 捕获阶段仅设置 `get_attn_backend().tail_source_kv_layer_id = layer.layer_id` 而避免图外回退；`#46144` 在多模态扩散模型分离式部署（Disaggregated Diffusion）中按角色（Encoder 2 GPUs -> `TP=2`，Denoiser 4 GPUs -> `TP=4`）独立重算自动并行度；`#41275` 在 `pool_configurator.py` 的 `_compressed_bytes_per_full_token` 中将 `c128` 固定压缩状态字节纳入 `ratio = 2` 的单 Token 成本（从 `352.0 B` 修正为 **`480.0 B`**），消除 `16 GiB` 池下 **`36.36%` 的 Token 容量高估**与运行期 OOM。
2. **`vllm-project/vllm@741584f8ca` (`#58744`)、`3bc521d9e6` (`#61543`) 与 `90309f6b74` (`#60490`)**：
   - **NVFP4 MLA 分块上下文聚集、有界 KV 组跳过与 Qwen4Exp QSA 预填充工作区预留**：`#58744` 在 `cp_gather_and_upconvert_nvfp4_kv_cache` 中支持 `seq_starts` 并直接使用 NVFP4 内核收集 `nvfp4_ds_mla` 分块上下文 KV（每行 `612 B` 对比 BF16 暂存 `1,152 B`，**降低 `1.8824x` 聚集显存带宽**，`16,384` Token 下从 `18.0 MiB` 降至 `9.5625 MiB`）；`#61543` 在 `DecodeBenchConnector` 中跳过 `CircularBufferSpec` 与 `HiSparseHotSpec` 等非传输有界 KV 组（`0` 崩溃）；`#60490` 在 `vllm/models/qwen4_exp/nvidia/qsa.py` 显存剖析阶段显式预留 **`128.0 MiB` QSA Indexer 预填充工作区**。
3. **`vllm-project/vllm@de096755ad` (`#60991`) 与 `115b59a2da` (`#63857`)**：
   - **GLM-5.3-Flash KDA 短卷积单参数融合与异构层注意力 TP 自动检测**：`#60991` 将 GLM-5.3 KDA 的 `q|k|v` 三个短卷积融合为单一 `conv1d.weight` 参数（参数量从 `3` 合并为 **`1`**），保证 RL 权重热更新（Refit）完整覆盖且 FULL CUDA Graph 捕获的指针地址永不漂移；`#63857` 在 `tools/recipes/model_constraints.py` 中遍历全部异构层的注意力头数（如 `[32, 48, 64]`），按跨层最大公约数求出合法最大 `TP = 16`，消除仅读取第 0 层推断出 `TP = 32` 导致 `48` 头层整除失败的崩溃。
4. **`NVIDIA/TensorRT-LLM@4b027a381d` (`#20759`) 与 `ae91047ac3` (`#20008`)**：
   - **Aggregated Helix KV Cache 预填充定容与 GDN MTP FlashInfer 环形缓冲区重放**：`#20759` 为跳过显存剖析的 Aggregated Helix 模式引入 `_HELIX_AGG_FREE_GPU_MEMORY_FRACTION = 0.6`（在 `60.0 GiB` 可用显存下分配 `36.0 GiB` KV Cache 并预留 **`24.0 GiB`（`40%`）预填充激活余量**）；`#20008` 在 `MambaHybridCacheManagerV2` 与 `gdn_mixer.py` 中集成 FlashInfer 环形缓冲区缓存重放（Ring-Buffer Cached Replay）以支持 GDN 多 Token 预测（MTP），将投机状态回滚/重放耗时从 `115.0 us` 降至 **`46.0 us`（`2.50x` 加速）**。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：在混合注意力模型中，若三类层的 KV 头数分别为 $H_{\text{kv}} \in \{32, 48, 64\}$，证明在无 KV 头复制的常规张量并行切分下，最大合法 TP 度为 $\gcd(32, 48, 64) = 16$，并指出仅检查首层 $H_{\text{kv},0} = 32$ 为何会在 $H_{\text{kv}} = 48$ 层触发分片不整除错误。
- **可证伪假设**：若在 DeepSeek-V4.1 的 `ratio = 2` 压缩池定容中忽略 `c128` 固定状态开销（每全量 Token 少算 $128\text{ B}$），则在 $16\text{ GiB}$ 预算下会多规划 $36.36\%$ 的逻辑 Token 槽位，在长上下文满载时必然触发物理池越界或 OOM（被 `sglang #41275` 证伪并修复）。
