# Enterprise Real-Time Streaming Feature Store & Online/Offline Skew Prevention (`R432`)

Deterministic evaluation suite comparing four real-time feature engineering and serving architectures across a high-throughput payment fraud & recommendation workload (`45,000 QPS` peak inference, `45` features/request, `120M` entities, `180,000` Kafka events/sec):

1. **Tier 1**: Dual-Codebase Offline SQL + Online Java/Python Microservice Rewrite (`18.4%` online/offline skew, `14.2%` future leakage, `48.0ms` P99 latency, `3,600s` lag, `$310,000/mo` TCO)
2. **Tier 2**: Nightly/Hourly Micro-Batch Spark to Redis (`6.8%` skew, `4.5%` future leakage, `14.5ms` P99 latency, `900s` lag, `$225,000/mo` TCO)
3. **Tier 3**: Naive Flink Streaming to Unbounded In-Memory Redis (`4.9%` skew, `2.1%` future leakage, `19.2ms` P99 latency, `2.5s` lag, `$168,000/mo` TCO)
4. **Tier 4**: Unified Feature DSL (`Feast / Tecton`) + `Apache Flink` Tiled Pane Aggregator + `ASOF Point-in-Time Join` + `Log-and-Wait` + Tiered `Redis / DragonflyDB` (`0.0%` skew, `0.0%` future leakage, `3.8ms` P99 latency, `0.35s` freshness lag, `$49,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_streaming_feature_store_flink_feast_online_offline_skew_selection/run.py
python experiments/enterprise_streaming_feature_store_flink_feast_online_offline_skew_selection/run.py --check
python -m unittest tests/test_enterprise_streaming_feature_store_flink_feast_online_offline_skew_selection.py
```
