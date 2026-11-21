# LLM Serving Admission Control, Backpressure & Graceful Degradation Selection Experiment (`Predictive TTFT Admission + SLM Overflow Downgrade` vs. `Unbounded FIFO Queue`)

Reproducible overload resilience and Goodput model comparing 4 admission control and load-shedding architectures under a `2.2x` flash traffic surge (`320 RPS` arriving at a `160 RPS` primary `70B` cluster).

## Run
```bash
python experiments/llm_serving_admission_control_backpressure_queue_shedding_ttft_sla_selection/run.py
python experiments/llm_serving_admission_control_backpressure_queue_shedding_ttft_sla_selection/run.py --check
```
