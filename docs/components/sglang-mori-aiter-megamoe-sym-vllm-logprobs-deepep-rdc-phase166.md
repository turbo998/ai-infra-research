# SGLang AMD MoRI AITER 激活契约直连与 MegaMoE 对称缓冲区前置分配、vLLM V2 分块 Prompt Logprobs 峰值显存优化及 DeepEP NVSHMEM 3.5 CUDA RDC 修复（Phase 166）

> **English Version**: [sglang-mori-aiter-megamoe-sym-vllm-logprobs-deepep-rdc-phase166.en.md](sglang-mori-aiter-megamoe-sym-vllm-logprobs-deepep-rdc-phase166.en.md)

## 1. 一手源码审计范围（Track 2 · 10 项零重叠提交）

- **SGLang**: `#37367` (`ef4f3f1151`, 2026-10-31), `#43110` (`1f834c1964`, 2026-10-31), `#47577` (`6e4d53fbc4`, 2026-10-31), `#43899` (`b8d86350d2`, 2026-10-31)
- **vLLM**: `#64841` (`42258ca4af`, 2026-10-31), `#64223` (`515b8753e3`, 2026-10-30)
- **TensorRT-LLM**: `#21320` (`6086b1c5d5`, 2026-10-30), `#21257` (`7aa8f85270`, 2026-10-30)
- **DeepEP**: `#578` (`567632dd59`, 2026-02-03), `#564` (`29d31c0957`, 2026-01-20)

## 2. 核心架构机制与定量验证

1. **SGLang AMD MoRI 基于 AITER 激活契约解析与 MegaMoE 对称缓冲区前置分配（`#37367`, `#47577`）**：
   - 直接根据 AITER 激活契约解析 MoRI Token 分发路径，消除冗余格式推断与重排，分发延迟从 `16.0 us` 降至 `13.0 us`（**-18.75%**）；同时在计算 KV 缓存容量前预先分配 MegaMoE 对称通信缓冲区（迟发 OOM 降为 **0**），并修复 HiCache 派生 Sidecar 跨源传输解析（`#43110`）。
2. **vLLM ModelRunner V2 分块 Prompt Logprobs 峰值显存优化（`#64841`）**：
   - 在计算长上下文 Prompt Logprobs 时按 Chunk 流式归约而非一次性物化全序列 Logits，在 8-Chunk 场景下将峰值显存从 `4.0 GiB` 降至 `0.5 GiB`（**-87.5%**）。
3. **DeepEP NVSHMEM >= 3.3.20 强制开启 CUDA RDC 与 NVSHMEM 3.5 RC QP 选路更新（`#564`）**：
   - 修复链接源码编译版 NVSHMEM 时内核与库各自实例化 `nvshmemi_ibgda_device_state_d` 的问题，强制开启 CUDA 可重定位设备代码（RDC）将设备状态符号实例从 `2` 收敛为 **1**，并适配 NVSHMEM 3.5 的 RC QP 索引布局。

## 3. 确定性实验复现

```bash
python experiments/sglang_mori_aiter_megamoe_sym_vllm_logprobs_deepep_rdc_phase166/run.py --check
```
