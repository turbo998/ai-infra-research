# SGLang M3 Reasoning-Effort Enum Guard & LTX-2.5 / Pika-H3-Flex Diffusion, vLLM GB300 AgentX & Full MI355 MoE, TRT-LLM PyExecutor & DeepEP SM90 (Phase 162)

> **Evidence Classification**: Primary upstream commit analysis (`implementation`: `140bb6dc8d70`, `dc79ae47ca74`, `37fc17b8d60a`, `d90e35cffa21`, `59d58fd261b1`, `d2d550c3fc12`, `7969af8decf9`, `d870ded367b1`, `01dc3aaac820`, `60d44037a702`) + deterministic Python 3.12 standard-library model (`own-offline`: `experiments/sglang_m3_enum_ltx25_vllm_gb300_mi355_trtllm_deepep_sm90_phase162/run.py`).

## 1. Core Architectural Findings & Implementation Paths

1. **SGLang M3 `reasoning_effort` Enum Guard (`140bb6dc8d70`), LTX-2.5 `1920x1088` Default Canvas (`dc79ae47ca74`) & Pika-H3-Flex 6-Step LoRA (`37fc17b8d60a`)**:
   - Rejects out-of-enum `reasoning_effort` inputs for M3 (`100.0%` rejection of invalid values), defaults LTX-2.5 two-stage generation to its native `1920x1088` canvas, and adds `Pika-H3-Flex` (MiniMax-H3 + 6-step LoRA) support.
2. **vLLM MiniMax-M3.1-Flash GB300 InferenceX AgentX Presets (`d90e35cffa21`) & Full MI355 MoE Kernel Execution (`59d58fd261b1`, `d2d550c3fc12`)**:
   - Runs ROCm MI355 MoE kernels across all `8` XCDs of a full MI355 rather than `4`-XCD DPX partitions (`2.0x` aggregate compute/HBM bandwidth in `own-offline` modeling).
3. **TensorRT-LLM `pyexecutor` Connector Split (`d870ded367b1`, `7969af8decf9`) & DeepEP SM90 Compatibility (`60d44037a702`, `01dc3aaac820`)**:
   - Eliminates SM90 Hopper PTX compatibility errors in DeepEP (`0` errors).

## 2. Deterministic Verification

```bash
python experiments/sglang_m3_enum_ltx25_vllm_gb300_mi355_trtllm_deepep_sm90_phase162/run.py --check
```
