# vLLM V1 UMBP KV Connector & THUDM/slime Straw In-Place Checkpoint Rewind (`vllm_umbp_kv_connector_slime_rewind`)

Deterministic Python 3.12 standard-library verification of vLLM V1's UMBP shared KV connector (`RankTopology`, `UMBPNamespace`, `KVLayoutPlanner`, `UMBPStoreConnectorScheduler`, and `UMBPStoreConnectorWorker` at `82b7dad5e3`) and THUDM/slime's in-place Straw checkpoint rewind with dataset cursor fallback (`4f204660ab`).

## Run

```bash
python experiments/vllm_umbp_kv_connector_slime_rewind/run.py --check
```
