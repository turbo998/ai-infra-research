# LLM Serving CPU Overhead & Zero-Bubble Overlap Scheduler Experiment (`vLLM V1 ZMQ IPC` & `SGLang Overlap Scheduler`)

Reproducible analytical model comparing 4 CPU-GPU scheduling architectures when serving `Llama-3.3-70B-FP8` at `Batch=128` (`10.0 ms` pure GPU forward step):
- **Tier 1 (`Monolithic Single-Process FastAPI + Scheduler + Sync D2H Detokenizer`)**: `7.0 ms` exposed CPU bubble per step (`17.0 ms` TPOT, `58.8%` GPU duty cycle, `7,529 TPS`).
- **Tier 2 (`Multi-Step Scheduling --num-scheduler-steps 8`)**: `2.4 ms` exposed CPU bubble (`12.4 ms` TPOT, `80.6%` GPU duty cycle, `10,322 TPS`).
- **Tier 3 (`Multiprocess ZMQ IPC Decoupling`)**: `1.1 ms` exposed CPU bubble (`11.1 ms` TPOT, `90.1%` GPU duty cycle, `11,531 TPS`).
- **Tier 4 (`vLLM V1 Multiprocess ZMQ + SGLang Async Overlap Scheduler`, Recommended)**: `0.08 ms` (`80 µs`) exposed CPU bubble (`10.08 ms` TPOT, `99.2%` GPU duty cycle, `12,698 TPS` — `+68.7%` higher throughput than Tier 1).

## Run & Verify

```bash
python experiments/llm_serving_cpu_overhead_vllm_v1_multiprocess_zmq_sglang_overlap_scheduler_selection/run.py
python experiments/llm_serving_cpu_overhead_vllm_v1_multiprocess_zmq_sglang_overlap_scheduler_selection/run.py --check
```
