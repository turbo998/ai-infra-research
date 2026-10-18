# Multimodal Encode-Prefill-Decode (EPD) Disaggregation Experiment (`own-offline`)

本实验基于 `vLLM v1` `EncoderCacheManager` (`vllm/v1/core/encoder_cache_manager.py`) 与 `SGLang` `TransportProxyTensor` (`python/sglang/srt/managers/mm_utils.py`) 一手实现，量化高分辨率视觉/视频大模型（`Qwen2.5-VL-72B`，`3,072` 视觉 Token/请求）在四种部署架构下的延迟与吞吐边界：
1. **Colocated Monolithic (`8x H100`)**：重计算 ViT 阻塞同卡 Decode 导致 P99 TPOT 飙升至 `63.61 ms`（恶化 `3.84x`），且 ViT 执行期间 144 GB LLM 权重闲置浪费 `38.4%` HBM 周期；
2. **P/D Disaggregated (`4P + 4D`)**：消除了 Decode 抖动，但 ViT 仍在 Prefill 池造成队头阻塞（TTFT `43.25 ms`）；
3. **3-Stage EPD + `EncoderCacheManager` (`1E + 3P + 4D`)**：结合跨轮次 `mm_hash` 引用计数缓存（多轮命中率 `92.0%`）与 `CUDA IPC` 零拷贝传输（`48.0 MB` 仅需 `0.132 ms`），将 TTFT 降至 **`27.60 ms`（提速 `1.59x`）**，P99 TPOT 压至 **`16.58 ms`（降低 `3.84x`）**，吞吐提升 **`2.70x`（`14.60 RPS`）**。

## 运行与校验

```powershell
python experiments\multimodal_encoder_disaggregation\run.py
python experiments\multimodal_encoder_disaggregation\run.py --check
```
