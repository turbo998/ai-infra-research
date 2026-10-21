# FlashInfer SM120 MegaMoE Register-Spill Elimination, TRT-LLM-Gen Multi-Wave KV Split, Kimi-K3 AttnRes Round-2 & NVFP4 Attention Packaging, and TRT-LLM Zero-Copy Video Encoding

Deterministic Python 3.12 standard-library reproduction of:
- `flashinfer-ai/flashinfer@cd6c1885a8` (`#6770`): SM120 Cake MegaMoE CUDA 13.3/13.4 register-spill elimination (`_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE`, per-GIN-backend modules, `m128` 1-step unroll, `ACTIVE_ROWS_FIELD_MAX = 0x7FFF` 15-bit signed ABI cap `<= 32640`).
- `flashinfer-ai/flashinfer@a3f2525e39` (`#6331`): TRT-LLM-Gen GQA decode opt-in multi-wave KV split (`kMaxNumWavesForKvSplit = 4` when single-wave splits `<= 2`, `1.86x` geomean on B200 64k lognormal KV, `2.02x` peak at `B=38` 288k lognormal).
- `flashinfer-ai/flashinfer@3b65505f40` (`#6904`) & `edfb3f7f88` (`#6902`): Cake Kimi-K3 AttnRes round-2 (`133` programs, `161` libraries, `582` timed rows) and NVFP4 persistent-tile attention self-contained `cake-build/1` manifest.
- `NVIDIA/TensorRT-LLM@c018848da6` (`#20575`): VisualGen zero-copy `memoryview(video_np.reshape(-1))` ffmpeg pipe avoiding a 711.9 MiB host allocation and GIL-held copy.

## Run

```bash
python experiments/flashinfer-sm120-megamoe-spill-trtllm-multiwave-kimi-k3-nvfp4/run.py --check
```
