# Enterprise LLM Observability & FinOps Chargeback Selection Experiment (`OpenTelemetry GenAI + Langfuse/Phoenix Tail Sampling`)

Reproducible analytical model comparing 4 LLM observability and token FinOps chargeback architectures across `10,000,000` daily LLM requests (`8K` average prompt + RAG context):
- **Tier 1 (`Synchronous Inline DB Logging + Naive Pre-Count Billing`)**: `+68.0 ms` hot-path latency penalty, `32.0%` FinOps billing error (ignores prefix cache discounts & mid-stream aborts), `$38,500/month` storage cost.
- **Tier 2 (`Async 100% Un-Sampled Full-Prompt ClickHouse/ES`)**: `+1.2 ms` latency, `$38,500/month` storage cost (`54 TB/month`).
- **Tier 3 (`5% Blind Head-Based Random Sampling`)**: `$2,400/month`, **but drops `95%` of production error traces and thumbs-down events**.
- **Tier 4 (`OTel GenAI Async Ring-Buffer + Tail-Based Sampling [100% Metrics/Errors + S3 Blob Payloads] + Cache-Aware FinOps Ledger`, Recommended)**: **`+0.15 ms` hot-path overhead**, **`0.02%` billing error (`99.98%` FinOps accuracy)**, `100%` error/feedback capture, and **`$4,200/month` storage (`-89.1%` cost savings)**.

## Run & Verify

```bash
python experiments/llm_observability_opentelemetry_genai_langfuse_phoenix_finops_chargeback_selection/run.py
python experiments/llm_observability_opentelemetry_genai_langfuse_phoenix_finops_chargeback_selection/run.py --check
```
