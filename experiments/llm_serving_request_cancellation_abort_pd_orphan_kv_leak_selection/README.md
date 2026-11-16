# Round 302: LLM Serving Request Cancellation Propagation & Disaggregated PD Orphaned KV Leak Selection

Deterministic evaluator comparing 4 request cancellation & cross-node KV lifecycle architectures under a `20%` client timeout/abort rate at `64K` context across wasted zombie GPU compute (`%`), leaked Decode KV memory (`GB/hr`), and effective goodput (`req/min`).
