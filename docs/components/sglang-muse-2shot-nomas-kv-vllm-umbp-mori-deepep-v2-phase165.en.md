# SGLang Muse-Spark Fused `2shot_push` All-Reduce & Nomas KV Budget Reclamation, vLLM UMBP MORI Integration & DeepEP EPv2 Multi-Parallel Architecture (Phase 165)

> **中文版**: [sglang-muse-2shot-nomas-kv-vllm-umbp-mori-deepep-v2-phase165.md](sglang-muse-2shot-nomas-kv-vllm-umbp-mori-deepep-v2-phase165.md)

## 1. Primary Source Audit Scope (Track 2 · 10 Zero-Overlap Commits)

- **SGLang**: `#47646` (`07a85353d6`, 2026-10-31), `#47545` (`e9c18259c7`, 2026-10-31), `#47516` (`a4a6eddd25`, 2026-10-31), `#46724` (`1c1166ad04`, 2026-10-31)
- **vLLM**: `#64222` (`a23fd99034`, 2026-10-30), `#61794` (`617135eed3`, 2026-10-30)
- **TensorRT-LLM**: `#21182` (`c749f53b16`, 2026-10-30), `#20743` (`0a7f070152`, 2026-10-30)
- **DeepEP**: `#664` (`af9a040318`, 2026-06-15), `#605` (`b306af06af`, 2026-04-29)

## 2. Core Architectural Mechanisms & Quantitative Verification

1. **SGLang `muse-spark-open` Fused MoE Finalize + `2shot_push` All-Reduce in CuTeDSL (`#46724`, `#47646`)**:
   - Fuses MoE expert finalize accumulation with the `2shot_push` NVLink all-reduce in a single CuTeDSL kernel, eliminating inter-kernel launch gaps and cutting combined latency from `30.0 us` to `22.2 us` (**-26.0%**), alongside accurate `PDLTriggerSecondary` contract documentation for `griddepcontrol.wait` (`#47646`).
2. **SGLang Nomas Diffusion KV Budget Reclamation & Request Logger Watermark Redaction (`#47516`, `#47545`)**:
   - Accounts for freed cached memory in Nomas KV budget calculation and rejects SP-indivisible segments up front, restoring usable KV headroom from `18.0 GiB` to `24.0 GiB` (**+33.33%**), while redacting client watermark keys in `request_logger` when watermarking is disabled (**0** leaked keys).
3. **vLLM UMBP Embedded Runtime + MORI Integration & DeepEP EPv2 Engram/PP/CP Support (`#61794`, `#605`)**:
   - Establishes vLLM UMBP embedded runtime and AMD MORI integration coverage (`2` targets) and verifies DeepEP EPv2's `3` co-designed parallel modes (EP, PP, CP with Engram and `EP_NIC_NAME` NIC binding).

## 3. Deterministic Reproduction

```bash
python experiments/sglang_muse_2shot_nomas_kv_vllm_umbp_mori_deepep_v2_phase165/run.py --check
```
