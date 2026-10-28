# Phase 118 Track C Experiment Suite

Deterministic Python stdlib simulation covering 10 upstream distributed training, EP JIT, and transfer-engine commits (`priority: 127`):
- Megatron-LM FP32 token-scaling factor for low-precision gradient buffers (`53015a0d7b`, #8532)
- TransformerEngine MXFP8 ONNX block quantization attributes (`9d4bd38678`, #3592)
- TransformerEngine AArch64 NCCL EP linking with BOLT (`f68a64a373`, #3627)
- verl Qwen3.5-35B FSDPTurbo `all2all` dispatcher on NPU (`4d3ed12a76`, #8400)
- Mooncake EP CUDA JIT compilation (`dfd1b716d5`, #4581)
- Mooncake TENT MLU benchmark support (`74944a7772`, #4703)
- Mooncake Store `BatchProbeKey` lease-free default (`19e0ab28f1`, #4729)
- Dynamo Operator LWS decoupling from Volcano (`acef8afe0a`, #17070)
- NIXL UCX `ROCM_IPC_MIN_ZCOPY=0` for AMD XGMI (`bc24e80c91`, #2371)
- NIXL `prometheus_mp` default shared directory (`9fbfcbb6f2`, #2402)

## Usage

```bash
python experiments/megatron_fp32_token_scale_te_mxfp8_mooncake_ep_jit_dynamo_nixl_phase118/run.py --write
python experiments/megatron_fp32_token_scale_te_mxfp8_mooncake_ep_jit_dynamo_nixl_phase118/run.py --check
```
