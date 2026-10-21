# vLLM `FusedMMInputNorm`, ROCm HY-V4, Scale-Out `derender`, Rust Histogram Buckets, TRT-LLM Video Hash & SGLang NPU Init

Deterministic Python 3.12 standard-library model reproducing:
- `vllm-project/vllm@90c4a1a73e` (`#62767`): `FusedMMInputNorm` `persistent=False` buffers under `DummyModelLoader`.
- `vllm-project/vllm@833349105b` (`#58509`): ROCm HY-V4 preview `triton_ihc_post_pre_rms_norm` & `AiterMxfp8Experts` SiLU gate mode.
- `vllm-project/vllm@817fa3518c` (`#60082`): Scale-out `/v1/generate` `output_mode="derender"` and `ParseContext` validation.
- `vllm-project/vllm@c1210ccbe5` (`#62826`) & `18e8791134` (`#62823`): Rust frontend `--custom-histogram-buckets` (`8` families) & `response_format` precedence.
- `NVIDIA/TensorRT-LLM@12accb2d5e` (`#19920`) & `sgl-project/sglang@2815eed49c` (`#43039`): Offline video `raw_bytes_hash` parity (`200 MiB` cap) and NPU `init_npu_backend` ordering.

## Run

```bash
python experiments/vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang/run.py --check
```
