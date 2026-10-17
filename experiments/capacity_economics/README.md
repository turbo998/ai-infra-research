# Capacity & Cost Economics Experiment (`own-offline`)

This synthetic, pure-Python experiment (`experiments/capacity_economics/run.py`) evaluates end-to-end AI infrastructure capacity sizing, training token economics, inference SLO goodput, Slurm GPU energy integration, and the analytical capacity/latency models of four AI Infra innovators:

1. **Training Token Economics & MFU vs. Quantization (`evaluate_training_economics`)**:
   - Reproduces TorchTitan `torchtitan/models/utils.py` (`get_dense_model_nparams_and_flops` and `get_moe_model_nparams_and_flops` at `pytorch/torchtitan@086bf6c166ec85c1298eb5596fa9bf95f6a2d840`) and `torchtitan/components/metrics.py` (`MetricsProcessor.log`):
     $$\text{FLOPs/token} = 6 \cdot N_{\text{matmul}} + 6 \cdot L \cdot H \cdot (2 d_{\text{head}}) \cdot S$$
     where for MoE models $N_{\text{matmul}} = (N_{\text{dense}} - N_{\text{emb}}) + N_{\text{router}} + N_{\text{shared}} + \lfloor N_{\text{routed}} \cdot k / E \rfloor$.
   - Enforces TorchTitan's `has_quantization` guardrail (`metrics.py:405-408`): when FP8/MX quantization is enabled, `mfu_bf16_pct` is reported as `None` (`not-assessed-quantized` in CSV) because dividing mixed-precision FLOPs by BF16 peak FLOPs (`gpu_peak_flops_bf16`) produces an invalid utilization ratio.
   - Reproduces Slurm `src/plugins/acct_gather_energy/gpu/acct_gather_energy_gpu.c` (`_get_additional_consumption` at `SchedMD/slurm@ad46cad8e0969e8e5945b88f0a21f6bed0f9a3bf`): integrates GPU energy per polling window via the trapezoidal rule $\Delta E = \lfloor (t_1 - t_0)(W_0 + W_1)/2 \rfloor$ Joules.

2. **Inference SLO Goodput & Prefill/Decode Disaggregation (`evaluate_inference_capacity_economics`)**:
   - Parameterizes request arrival burstiness using vLLM `benchmarks/benchmark_serving.py` (`get_request` Gamma arrival distribution $\text{Gamma}(\text{shape}=\beta, \theta=1/(\lambda \beta))$ with $\text{CV}_a^2 = 1/\beta$, and `calculate_metrics` `request_goodput` SLO filter at `vllm-project/vllm@25f560a62c4f955672e2c6080b17ab3a48f96201`).
   - Sizes separate Prefill and Decode instance pools (`benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py`) vs. colocated instances subject to prefill-decode interference, and quantifies the TTFT goodput collapse when sizing naively at 92% utilization under bursty arrivals ($\beta = 0.25$).

3. **AI Infra Innovators Capacity & I/O Models (`evaluate_innovator_economics`)**:
   - **Fireworks AI** (`fw-ai/benchmark@94cd111967c7be51a35777c475d7c5d2b66ec94a`, `llm_bench/prefill_load_test.py`): `cached_tokens + 1` worker warmup, inverse dynamic batch sizing `max(1, min_tokens_to_batch // (prompt_tokens - cached_tokens))`, and multi-worker straggler throughput bound `max(server_durations)`.
   - **Together AI / Sequoia** (`Infini-AI-Lab/Sequoia@0ff3ff71475ca148f2dfb7545e0de5caf5c67b39`, `Engine/offload_engine.py`): 2-slot ping-pong `LlamaLayerBuffer` CPU-to-GPU PCIe layer streaming and speculative tree acceptance amortization.
   - **Groq** (`groq/groqflow@720e024414be401a7853931a2622a26446883746`, `groqflow/groqmodel/groqmodel.py`): `GroqModel.estimate_performance()` 900 MHz deterministic cycle counter, `* num_chips_used` multichip pipeline stage multiplier, and 24 GB/s + 10 $\mu\text{s}$ PCIe transfer overhead.
   - **Cerebras** (`Cerebras/sdk-examples@d19f2ca3e51285fc5904c78643b38819a2f26c44`, `benchmarks/bandwidth-test/run.py` & `src/bw_sync_kernel.csl`): West/East `memcpy` reserved PE columns, 46 KB per-PE hardware FIFO column sizing, 2D Manhattan wavefront timestamp compensation `time_ref - (px + py)`, and 850 MHz wafer I/O bandwidth calculation.

## Usage

Regenerate outputs:

```powershell
python experiments\capacity_economics\run.py
```

Verify deterministic outputs without modifying files:

```powershell
python experiments\capacity_economics\run.py --check
```
