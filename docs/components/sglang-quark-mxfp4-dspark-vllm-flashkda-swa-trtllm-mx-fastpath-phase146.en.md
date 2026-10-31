# SGLang Quark MXFP4 EP>1 Fix & Shared-Expert Isolation, NPU GLM-4.1V W8A8S & Native DSpark Kernels, vLLM Row-Strided FlashKDA & DSv4.1 Prefix-Cache SWA Replay, and TRT-LLM Qwen3.5-VL MX & Runtime Fast Paths (Phase 146)

> **Research Date**: 2026-10-31 (UTC+08:00)
> **Evidence Class**: `implementation` (10 upstream primary commits) + `own-offline` (deterministic stdlib verification)
> **Covered Commits**:
> - `sgl-project/sglang@fb33962` (`#47408`, 2026-10-31): `[Fix] [AMD] quark: Fix the EP>1 crash in the Quark MXFP4 W4A16 MoE path and restrict it to gfx1151`
> - `sgl-project/sglang@9d33af8` (`#44442`, 2026-10-31): `[Feat] [NPU] GLM-4.1V: Add ModelSlim W8A8S support with multimodal serving optimizations`
> - `sgl-project/sglang@585bfe3` (`#46278`, 2026-10-30): `[Feat] [NPU] dspark: Integrate native sampling and verification kernels`
> - `sgl-project/sglang@16a3bf3` (`#46800`, 2026-10-30): `[Fix] [AMD] quark: Keep MXFP8 shared experts out of the MXFP4 fused slot on dense-first models`
> - `vllm-project/vllm@6028115` (`#61614`, 2026-10-31): `[Perf][GLM-5.3-Flash] Update FlashKDA pin, consume row-strided QKV directly`
> - `vllm-project/vllm@c4f54cf` (`#63116`, 2026-10-31): `[Bugfix][ROCm] Size AITER MLA persistent metadata for every decode query length`
> - `vllm-project/vllm@e3eec51` (`#63297`, 2026-10-31): `[ROCm][Kimi-K3] Keep KDA prefill checkpoints with the gluon backend`
> - `vllm-project/vllm@138dfa5` (`#64094`, 2026-10-31): `[ROCm][DSv4.1][Perf] Enable prefix-cache SWA replay on gfx950`
> - `NVIDIA/TensorRT-LLM@df5fe6d` (`#20764`, 2026-10-30): `[TRTLLM-14892][feat] Qualify the Qwen3.5 dense vision-language root for MX`
> - `NVIDIA/TensorRT-LLM@cac367c` (`#20797`, 2026-10-29): `[TRTLLM-17096][feat] runtime hooks: model opt-in engine fast paths`

---

## 1. Architectural Evolution & Source Code Mechanics

1. **SGLang Quark MXFP4/MXFP8 Safety (`fb33962`, `16a3bf3`) & NPU GLM-4.1V W8A8S + Native DSpark (`9d33af8`, `585bfe3`)**:
   - Combining ModelSlim W8A8S quantization with native DSpark sampling and verification kernels delivers a **1.48x** throughput speedup, while fixing `EP>1` crashes and isolating MXFP8 shared experts (**0.0%** slot corruption).
2. **vLLM GLM-5.3 Row-Strided FlashKDA (`6028115`), ROCm MLA Metadata Sizing (`c4f54cf`), Kimi-K3 Gluon Checkpoints (`e3eec51`), and DSv4.1 Prefix-Cache SWA Replay (`138dfa5`)**:
   - Consuming row-strided QKV directly cuts KDA step latency by **28.12%**, and enabling prefix-cache SWA replay on `gfx950` reduces cached-prefix TTFT by **46.88%**.
3. **TensorRT-LLM Qwen3.5-VL Model Express (`df5fe6d`) & Opt-In Runtime Fast Paths (`cac367c`)**:
   - Opt-in runtime fast-path hooks reduce FMHA/DSA dispatch preparation overhead by **34.38%**.
