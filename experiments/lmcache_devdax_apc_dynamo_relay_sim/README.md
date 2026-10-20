# LMCache Device-DAX Range Partitioning & APC Attribution + Dynamo Cuckoo DC Recovery & Migration Rebasing Simulation

Deterministic Python 3.12 standard-library model of `LMCache/LMCache` commits `e8dc1a1dae` and `168130df9c` plus `ai-dynamo/dynamo` commits `909ebee0c4` and `cc02d8b1c2`:

- **LMCache Device-DAX/CXL L1 Offset Partitioning (`utils.py`, `devdax_memory_allocator.py`)**: Validates `/sys/dev/char/{major}:{minor}/{size,align}` geometry (`2 MiB` alignment, `512 GiB` capacity) and disjoint range partitioning `[0, 256 GiB)` + `[256 GiB, 512 GiB)` prior to `mmap` to prevent `SIGBUS`.
- **LMCache APC-Covered Prefix Exclusion (`lookup.py`)**: Excludes `start_chunk * chunk_size` (`768` tokens) from `looked_up_tokens` (`2,048 -> 1,280`), eliminating fake `37.5%` L1 hit rates on LMCache misses and preventing `100`-token sub-chunk double counting via `max(engine_hit_tokens, covered_tokens + hit_tokens)`.
- **Dynamo Cuckoo DC Relay & Migration Rebasing (`cuckoo/dc.rs`, `migration.rs`)**: Replays `Removed`/`Cleared` events in cached snapshot tails (`2` zombie canonical candidates pruned) and rebases migrated prompt/completion usage (`prompt: 7 -> 3`, `completion: 2 -> 6`, `cached: 7 -> 3`).

## Usage

```bash
python experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py
python experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py --check
```
