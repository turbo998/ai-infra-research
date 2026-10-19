# SGLang `HiRadixCache`, `MLAHostDedupBroadcaster`, and Lockstep L3 Prefetch Simulator

Deterministic simulator verifying `MLAHostDedupBroadcaster` (`mla_host_dedup.py`), `L2TransferEngine` (`l2_transfer.py`), `StoragePrefetchRetries` (`storage_prefetch.py`), and `PoolHitPolicy` (`hicache_storage.py`) in `sgl-project/sglang`.

## Run & Check

```bash
python experiments/sglang_hiradix_cache_controller/run.py --check
```
