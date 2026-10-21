# TRT-LLM MiniMax-M3 3-Term BF16 Router & FLUX FP8 MHA, SGLang Quark AITER ASM FP4 & Rust Health-Ack, and vLLM MoRIIO Step-Barrier Read Await

Deterministic verification script for:
- `NVIDIA/TensorRT-LLM@522daa8cdf` (PR `#20699`), `5d6fbb2062` (PR `#18396`), & `5f90e8939d` (PR `#20387`)
- `sgl-project/sglang@6e98c84c60` (PR `#39598`), `ed268dd320` (PR `#45628`), & `ca72dad953` (PR `#45475`)
- `vllm-project/vllm@db3b0f0fae` (PR `#60808`) & `35d01660ff` (PR `#62244`)

## Run
```bash
python experiments/trtllm_minimax_m3_router_flux_sglang_quark_vllm_moriio/run.py --check
```
