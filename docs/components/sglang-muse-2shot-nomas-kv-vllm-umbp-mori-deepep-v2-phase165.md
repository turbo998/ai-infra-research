# SGLang Muse-Spark 融合 `2shot_push` All-Reduce 与 Nomas KV 预算回收、vLLM UMBP MORI 集成及 DeepEP EPv2 多维并行架构（Phase 165）

> **English Version**: [sglang-muse-2shot-nomas-kv-vllm-umbp-mori-deepep-v2-phase165.en.md](sglang-muse-2shot-nomas-kv-vllm-umbp-mori-deepep-v2-phase165.en.md)

## 1. 一手源码审计范围（Track 2 · 10 项零重叠提交）

- **SGLang**: `#47646` (`07a85353d6`, 2026-10-31), `#47545` (`e9c18259c7`, 2026-10-31), `#47516` (`a4a6eddd25`, 2026-10-31), `#46724` (`1c1166ad04`, 2026-10-31)
- **vLLM**: `#64222` (`a23fd99034`, 2026-10-30), `#61794` (`617135eed3`, 2026-10-30)
- **TensorRT-LLM**: `#21182` (`c749f53b16`, 2026-10-30), `#20743` (`0a7f070152`, 2026-10-30)
- **DeepEP**: `#664` (`af9a040318`, 2026-06-15), `#605` (`b306af06af`, 2026-04-29)

## 2. 核心架构机制与定量验证

1. **SGLang `muse-spark-open` MoE Finalize 与 `2shot_push` All-Reduce CuTeDSL 融合（`#46724`, `#47646`）**：
   - 将 MoE 专家加权归约（Finalize）与双跳推送（`2shot_push`）NVLink All-Reduce 融合为单核 CuTeDSL 流水线，消除核间启动间隙并重叠累加与远端写入，端到端延迟从 `30.0 us` 降至 `22.2 us`（**-26.0%**）；同时修正 `PDLTriggerSecondary` 关于 `griddepcontrol.wait` 覆盖边界的契约说明（`#47646`）。
2. **SGLang Nomas 扩散模型 KV 预算回收与请求日志水印密钥脱敏（`#47516`, `#47545`）**：
   - 修复 Nomas KV 预算计算未计入已释放缓存显存的缺陷并前置拒绝不可被 SP 整除的分段，使可用 KV 预算从 `18.0 GiB` 恢复至 `24.0 GiB`（**+33.33%** 有效余量）；同时在关闭水印时自动脱敏 `request_logger` 中的客户端水印密钥（泄露数 **0**）。
3. **vLLM UMBP 嵌入式运行时 + MORI 集成与 DeepEP EPv2 Engram/PP/CP 支持（`#61794`, `#605`）**：
   - 为 vLLM 补齐 UMBP 嵌入式运行时与 AMD MORI 互连集成测试覆盖，并分析 DeepEP EPv2 对 Engram、流水线并行（PP）、上下文并行（CP）及 `EP_NIC_NAME` 网卡绑定（`3` 类并行模式协同）的架构演进。

## 3. 确定性实验复现

```bash
python experiments/sglang_muse_2shot_nomas_kv_vllm_umbp_mori_deepep_v2_phase165/run.py --check
```
