# TorchTitan DeepSeek-V3 HF Save Map & GPT-OSS Dual-Layout Adapter, Dynamo Elastic-EP NIXL & TRT-LLM `ob1` Snapshot Transport, and Mooncake Accessor / NIXL Trace V2

## Overview / 概述

- **English**: Deterministic Python 3.12 standard-library verification model (`own-offline`) covering six upstream commits landed on 2026-10-20:
  1. `pytorch/torchtitan@1d581795cd` (`#5065`): Filtering `fqn_to_index_mapping` in `DeepSeekV3StateDictAdapter` (`91,991` released entries -> `45,395` for `num_mtp_layers=0` and `46,183` for `num_mtp_layers=1` by dropping `45,808` FP8 `.weight_scale_inv` entries and `788` `model.layers.61.*` entries when `num_mtp_layers=0`, and cloning `_MTP_SHARED_COPIES` on `to_hf`).
  2. `pytorch/torchtitan@da54a64ede` (`#5064`): Dual-layout `GptOssStateDictAdapter` (`gate_up_proj [E, D, 2F]` input-first BF16 vs `gate_up_proj_blocks [E, 2F, D/32, 16]` output-first MXFP4).
  3. `ai-dynamo/dynamo@02da7f4486` (`#16865`): Preventing SGLang Elastic-EP workers from crashing on the first MoE forward over NIXL (`SGLANG_DEEPEP_BF16_DISPATCH=1`, `SGLANG_NIXL_EP_BF16_DISPATCH=1`, `--moe-runner-backend deep_gemm`, `SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`, `--moe-dense-tp-size 1`).
  4. `ai-dynamo/dynamo@c8b3a8e0e5` (`#16372`): Forcing OpenMPI `OMPI_MCA_pml=ob1` and `OMPI_MCA_btl=tcp,self` in TRT-LLM snapshot mode so frozen ranks do not drop connections on unanswered UCX TCP keepalive probes.
  5. `kvcache-ai/Mooncake@cc6b43e989` (`#4717`) & `ai-dynamo/nixl@56c1c2f706` (`#2445`): Mooncake per-key metadata accessor contract with shard-monotonic `route::Generation` and NIXL Trace Plugin API V2 (`pushContext` / `popContext`).
- **中文**：针对 TorchTitan DeepSeek-V3 HF 保存索引过滤与 GPT-OSS 双布局适配器、Dynamo SGLang Elastic-EP NIXL 首轮 MoE 前向防崩溃与 TRT-LLM OpenMPI `ob1` 快照传输、以及 Mooncake 逐键元数据访问器与 NIXL Trace V2 插件契约的确定性验证模型。

## Run / 运行方式

```bash
python experiments/torchtitan_dsv3_hf_save_map_gpt_oss_mxfp4_dynamo_elastic_ep/run.py --check
```
