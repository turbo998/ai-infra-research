# SGLang Extend Host Prompt Tracking, SO_REUSEPORT & Rust Health Probe, TensorRT-LLM Responses MM & vLLM Stream Processor Verification

Deterministic Python 3.12 standard-library verification of:
- `sgl-project/sglang@a3d00eff58b6c22ea7539f882cc10c537634200b` (`#45487`): Host `contains_last_prefill_chunk` computation and TBO child inheritance.
- `sgl-project/sglang@7e5256d1ba3aaa628bb8b756fafb904a76a1123a` (`#45482`): Per-worker `SO_REUSEPORT` listeners for Uvicorn tokenizer workers (`uvicorn>=0.51`).
- `sgl-project/sglang@ca72dad953171ea4f580d89f70f999fc89501067` (`#45475`): Rust server `push_health_ack()` (`frame_health_ack`) answering skipped probes on the response ring.
- `sgl-project/sglang@68623cae9256ad8ee00269c8c1d52e4c2915daa5` (`#45670`): Mamba ping-pong `track_position` index gathering.
- `NVIDIA/TensorRT-LLM@1c7fadcfbe50911d784658e902d1bd003d89fdd7` (`#20253`) & `271ed9f22ab85e5ce2b45e326f5b41cbb249bd93` (`#20231`): KV V2 / CUDA graph warmup cleanup and Responses API multimodal image input.
- `vllm-project/vllm@6ae1051bd9120c9e2974a8ad502b4c9716db6b9f` (`#62741`) & `5ea2c7ff1e64b36b45bf6bd502367c85633b2d8c` (`#62706`): Rust `--exclude-tools-when-tool-choice-none` / `max_tool_calls` and shared `ChatStreamProcessor`.

## Run
```bash
python experiments/sglang-extend-host-prompt-reuseport-rust-probe-trtllm-v2-vllm-stream/run.py --check
```
