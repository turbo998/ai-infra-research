# SGLang Mamba DCP Chunk Grid & Kimi-K3 SM107 Split-K, vLLM RL NCCL Rendezvous & TensorRT-LLM Packed Ulysses Unscatter (Phase 87)

Deterministic Python 3.12 stdlib model auditing:
- `sgl-project/sglang#41701` (`8252e3af63`): Chunked prefill truncation alignment on Mamba chunk grid (`64`) via LCM so DCP can donate checkpoints at every chunk boundary (`4 vs 1` donations across a 1024-token prefill).
- `sgl-project/sglang#46464` (`1b112fd62a`): Kimi-K3 FP32 warp split-K GEMM for the MoE front projection (`N=6016, K=7168`, router slice `1536:2432` = `896` experts) on SM107 (`m in {8, 16, 32}`).
- `vllm-project/vllm#63460` (`c9dbc92630`) & `#63496` (`8e11a8a49e`): Shared concurrent NCCL trainer/worker rendezvous initialization and early config-time rejection of MegaMoE backends with LoRA.
- `NVIDIA/TensorRT-LLM#17147` (`1586d85375`) & `#20270` (`87342f4474`): Fused `ulyssesPackedQkvPostUnscatterKernel` (`[P, B, Sp, 3, H, D] -> 3 x [B, P*Sp, H, D]` in a single `uint4` vectorized kernel launch with `int32` overflow checks) and PyExecutor runner surface slimming.

## Run
```bash
python experiments/sglang_vllm_trtllm_phase87/run.py
python experiments/sglang_vllm_trtllm_phase87/run.py --check
```
