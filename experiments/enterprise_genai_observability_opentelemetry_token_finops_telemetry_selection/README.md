# Enterprise GenAI Full-Stack Observability (`OpenTelemetry GenAI Semantic Conventions`, `ClickHouse` & `WCU` Token FinOps) (`#483`)

Reproducible analytical model comparing Legacy HTTP APM (`145 min` MTTR), Synchronous Full-Prompt SaaS Logging (`$68,500/mo` storage bill, `+38.5ms` latency), Prometheus Metrics Only, and `OpenTelemetry GenAI` Metadata-Everywhere + Tail-Sampled Payloads + `ClickHouse WCU` FinOps (`3.4 min` MTTR, `99.6%` chargeback accuracy, `$3,800/mo`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_genai_observability_opentelemetry_token_finops_telemetry_selection/run.py --check`
