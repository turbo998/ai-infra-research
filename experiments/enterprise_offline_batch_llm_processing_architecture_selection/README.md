# Enterprise High-Throughput Offline Document & Data Batch LLM Processing Selection (`R478`)

Deterministic evaluation suite comparing four enterprise offline batch LLM processing architectures across `60 Billion` monthly batch tokens (`10,000,000` nightly knowledge-extraction / classification records, `8-hour` overnight completion SLA):

1. **Tier 1**: Synchronous Real-Time PAYGO / PTU API in `asyncio.gather()` Loops (`$3.80/1M tokens`, `38.5%` online Copilot SLA degradation from HTTP 429 contention, `14.5h` completion, `$228,000/mo` TCO)
2. **Tier 2**: 100% Cloud-Managed Asynchronous `Batch API` (`Azure OpenAI Global Batch / AWS Bedrock Batch` @ 50% Discount) (`$1.90/1M tokens`, `0.0%` online contention, `9.2h` completion, `$114,000/mo` TCO)
3. **Tier 3**: Dedicated Self-Hosted `Ray Data + vLLM Offline` on On-Demand GPUs (Unsorted) (`$1.25/1M tokens`, `5.8h` completion, `$75,000/mo` TCO)
4. **Tier 4**: Off-Peak Idle `PTU` Backfill + Prefix-Sorted `Ray Data + vLLM` on Spot GPUs + Cloud `Batch API` Overflow (`$0.58/1M tokens`, `0.0%` online SLA degradation, `3.4h` completion, `94.8%` 24x7 fleet utilization, `$34,800/mo` TCO)

## Usage

```bash
python experiments/enterprise_offline_batch_llm_processing_architecture_selection/run.py
python experiments/enterprise_offline_batch_llm_processing_architecture_selection/run.py --check
python -m unittest tests/test_enterprise_offline_batch_llm_processing_architecture_selection.py
```
