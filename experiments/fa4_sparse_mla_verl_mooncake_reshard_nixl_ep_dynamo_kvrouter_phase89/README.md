# Phase 89 Track C: FlashAttention-4 Sparse MLA, verl Multi-Node DP, Mooncake Tiled COO & HA Store, NIXL EP & EFA, and Dynamo KV-Router

Deterministic Python 3.12 standard-library model reproducing the architectural invariants across the 10 Phase 89 Track C training/KV-fabric commits (`Dao-AILab/flash-attention`, `volcengine/verl`, `kvcache-ai/Mooncake`, `ai-dynamo/nixl`, and `ai-dynamo/dynamo`).

## Usage

```bash
python experiments/fa4_sparse_mla_verl_mooncake_reshard_nixl_ep_dynamo_kvrouter_phase89/run.py
python experiments/fa4_sparse_mla_verl_mooncake_reshard_nixl_ep_dynamo_kvrouter_phase89/run.py --check
```
