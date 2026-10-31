# SGLang Detokenizer Resident State, DSv4 Verify Idle Guard & KDA Live Event Binding, vLLM 32-Row MegaMoE & MoRIIO Hybrid Read, and TRT-LLM VisualGen Block-Causal Cache & KV-Balanced Admission (Phase 144)

> **Research Date**: 2026-10-31 (UTC+08:00)
> **Evidence Class**: `implementation` (10 upstream primary commits) + `own-offline` (deterministic stdlib verification)
> **Covered Commits**:
> - `sgl-project/sglang@8080507` (`#47260`, 2026-10-30): `[Perf] detokenizer: Send per-request decode inputs once and keep them in the detokenizer`
> - `sgl-project/sglang@f362716` (`#46676`, 2026-10-30): `[Fix] dsv4: Keep a coordinated verify rank out of the decoder-tail idle replay`
> - `sgl-project/sglang@b572b8e` (`#45839`, 2026-10-30): `[BugFix][HiCache * BCG] bind captured KDA waits to live load events`
> - `vllm-project/vllm@2bfb8f1` (`#64467`, 2026-10-30): `[Perf][MoE] Use 32-row tiles in prepare_megamoe_inputs from 1024 tokens`
> - `vllm-project/vllm@89a9a03` (`#51953`, 2026-10-30): `[Bugfix][MRV2] Fix DiffusionGemma runtime OOM via tiled logits projection`
> - `vllm-project/vllm@1353e19` (`#60867`, 2026-10-30): `[KV Connector][ROCm] MoRIIO: hybrid READ for several attention groups and shared-group speculative methods`
> - `vllm-project/vllm@322f90a` (`#63219`, 2026-10-30): `[Bugfix][Spec Decode] Per-request draft retention for structured output under PP`
> - `NVIDIA/TensorRT-LLM@236d9b9` (`#19938`, 2026-10-29): `[None][feat] Paged block-causal K/V cache and cached attention for VisualGen`
> - `NVIDIA/TensorRT-LLM@1ac6682` (`#21073`, 2026-10-29): `[None][perf] Attention DP: KV-balanced admission of generation-only requests`
> - `NVIDIA/TensorRT-LLM@5be3334` (`#20941`, 2026-10-29): `[None][perf] Kimi-K3 KDA prefill grid sizing and MTP-verify kernels; DWDP cross-step expert prefetch with configurable slots`

---

## 1. Architectural Evolution & Source Code Mechanics

1. **SGLang Detokenizer Resident Inputs (`8080507`), DSv4 Verify Idle Guard (`f362716`), and HiCache*BCG KDA Live Event Binding (`b572b8e`)**:
   - Sending per-request static decode parameters once and caching them inside `DetokenizerManager` reduces per-step IPC payload overhead by **78.13%** over a 64-step decode window.
   - Binding Breakable CUDA Graph (BCG) KDA waits to live HiCache load events prevents stale event synchronization during prefix cache transfers.
2. **vLLM 32-Row MegaMoE Tile Prep (`2bfb8f1`), DiffusionGemma Tiled Logits (`89a9a03`), MoRIIO Hybrid READ (`1353e19`), and PP Speculative Draft Retention (`322f90a`)**:
   - Switching `prepare_megamoe_inputs` to 32-row tiles from 1024 tokens cuts preprocessing latency by **31.25%**, while 8-way tiled logits projection cuts DiffusionGemma transient logits memory by **87.50%**.
3. **TensorRT-LLM VisualGen Paged Block-Causal KV Cache (`236d9b9`), Attention DP KV-Balanced Admission (`1ac6682`), and Kimi-K3 DWDP Prefetch (`5be3334`)**:
   - Paged block-causal KV layout cuts VisualGen KV memory by **56.25%**, and KV-balanced Attention DP admission reduces cross-rank KV load skew by **68.42%**.
