# LMCache Device-DAX/CXL L1 Offset Partitioning & APC Hit Attribution, plus Dynamo Cuckoo DC Snapshot Removal Replay & Migration Token Rebasing

> **中文版**: [lmcache-devdax-offset-apc-attribution-dynamo-relay.md](lmcache-devdax-offset-apc-attribution-dynamo-relay.md)  
> **Evidence Class**: `implementation` (First-hand source verification) + `own-offline` (Deterministic stdlib simulation)  
> **Upstream Repositories & Commits**: `LMCache/LMCache` commits [`e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3`](https://github.com/LMCache/LMCache/commit/e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3) & [`168130df9c8fafa376b650814598179c212d3e66`](https://github.com/LMCache/LMCache/commit/168130df9c8fafa376b650814598179c212d3e66) (2026-10-20); `ai-dynamo/dynamo` commits [`909ebee0c4b63988bea6ab474a46913073e4a2dc`](https://github.com/ai-dynamo/dynamo/commit/909ebee0c4b63988bea6ab474a46913073e4a2dc) & [`cc02d8b1c2b60a9cbe291b5264456897c64d7e7c`](https://github.com/ai-dynamo/dynamo/commit/cc02d8b1c2b60a9cbe291b5264456897c64d7e7c) (2026-10-20)  
> **Experiment**: [`experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py`](../../experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py) (`--check` passed)

---

## 1. Architectural Motivation & Engineering Bottlenecks

In multi-instance CXL/Device-DAX KV cache pools and cross-datacenter KV routing planes, four subtle failure modes arise:
1. **Linux Device-DAX `fstat` Blindness & Deferred `SIGBUS`**: When multiple LMCache servers partition a shared 512 GiB CXL Device-DAX (`/dev/dax0.0`), `os.fstat(fd).st_size` returns `0` for character devices, and the kernel only checks bounds/alignment on page fault—raising a fatal `SIGBUS` on first touch if `offset + size > capacity` or if `offset`/`size` violates device alignment (`2 MiB`).
2. **APC-Covered Prefix Inflating LMCache L1 Hit Rates**: When the inference engine's own Automatic Prefix Cache (APC) already holds `start_chunk` chunks, LMCache only probes `[start_chunk, end)`. Previously, `l1_found_count += job.start_chunk` credited the engine's APC prefix as an LMCache L1 hit, inflating L1 hit rates even when LMCache missed 100% of looked-up chunks.
3. **Dynamo Cuckoo Filter (`DcCkfState`) Snapshot-Tail Removal Drops**: `LocalKvIndexer` recovery returns a cached `TreeDump` plus a buffered event tail containing `KvCacheEventData::Removed` or `Cleared` events. Rejecting non-`Stored` events in `DcCkfRankReplacement::push_event` broke recovery or admitted zero-owner zombie canonical hashes.
4. **Migrated Stream Prompt Token Double Counting (`RetryManager`)**: When a worker disconnects after streaming $k$ tokens, `RetryManager` replays those $k$ tokens in the retry prompt. Without rebasing, the new worker reports `prompt_tokens = P + k` and `cached_tokens = P + k`.

---

## 2. First-Hand Source Analysis (`LMCache/LMCache` & `ai-dynamo/dynamo`)

### 2.1 LMCache Device-DAX sysfs Geometry Pre-Check & Disjoint Arena Ranges (`utils.py`, `devdax_memory_allocator.py` @ `e8dc1a1dae`)

- **`get_dax_device_geometry(fd)` & `check_dax_range(device_path, fd, offset, size)`**:
  Resolves `/sys/dev/char/{major}:{minor}/{size,align}` for character devices and validates before `mmap`:
  $$\text{offset} \bmod G_{\text{mmap}} = 0, \quad \text{offset} + \text{size} \le \text{capacity}, \quad (\text{offset} \bmod \text{align} = 0) \land (\text{size} \bmod \text{align} = 0)$$
- **`_DevDaxArena.overlaps(offset, size)`**: Rejects overlapping ranges `offset < arena.offset + arena.size and arena.offset < offset + size` on the same `(is_chr, major, minor)` device while allowing disjoint slices (`[0, 256 GiB)` and `[256 GiB, 512 GiB)`).

### 2.2 LMCache APC-Covered Prefix Exclusion & Sub-Chunk De-Duplication (`lookup.py` @ `168130df9c`)

- **Pure Lookup Hit-Rate Attribution**:
  $$\text{looked\_up\_tokens} = \text{requested\_tokens} - \text{start\_chunk} \times \text{chunk\_size}$$
  `_consume_prefetch_result` attributes `l1_chunks = min(l1_found_count, found_count)` and `l2_chunks = found_count - l1_chunks` strictly before shifting `found_count += job.start_chunk`.
- **Whole-Request Hit De-Duplication (`request_hit_tokens`)**:
  $$\text{request\_hit\_tokens} = \max\big(\text{engine\_hit\_tokens},\; \text{covered\_tokens} + \text{hit\_tokens}\big)$$
  Taking `max` rather than `+` prevents double counting the sub-chunk tail inside `engine_hit_tokens`.

### 2.3 Dynamo `DcCkfRankReplacement` Removal Replay & `RetryManager::rebase_usage` (`dc.rs` @ `909ebee0c4`, `migration.rs` @ `cc02d8b1c2`)

1. **Snapshot-Tail `Removed` / `Cleared` Replay (`lib/kv-router/src/indexer/cuckoo/dc.rs`)**: `DcCkfRankReplacement::push_event` removes deleted hashes from `source_lineage` (or clears both maps on `Cleared`), and `DcCkfState::install_replacement` prunes zero-owner candidates via `replacement.canonical_candidates.retain(|canonical| owner_increments.contains_key(canonical))`.
2. **Post-Migration Usage Rebasing (`lib/llm/src/migration.rs:rebase_usage`)**: With client prompt length $P$ and replayed token count $R = \text{token\_ids.len()} - P$:
   $$P' = P_{\text{worker}} - R, \quad C' = C_{\text{worker}} + R, \quad H' = \min(H_{\text{worker}}, P'), \quad \text{completion\_tokens\_details} = \text{None}$$

---

## 3. Deterministic Offline Verification (`lmcache_devdax_apc_dynamo_relay_sim`)

Running `python experiments/lmcache_devdax_apc_dynamo_relay_sim/run.py --check` verifies:

| Subsystem & Check | Legacy / Invalid Input | Verified Deterministic Outcome |
|---|---|---|
| CXL Device-DAX (`512 GiB, align=2 MiB`) | 1) Disjoint `[256 GiB, 512 GiB)`; 2) Overlapping `[128 GiB, 384 GiB)`; 3) `+1 MiB` unaligned; 4) Past capacity | 1) `admitted = True`; 2/3/4) Rejected prior to `mmap` (zero runtime `SIGBUS`) |
| LMCache APC Exclusion (`2,100` tokens) | `start_chunk = 3` (`768` tokens); legacy reported **`37.5%`** fake L1 hit rate on 0-chunk miss | `looked_up_tokens = 1,280`, miss `l1_hit_rate = 0.0%`; on `2 L1 + 1 L2` hit, `request_hit_tokens = 1,536` (`100`-token overlap prevented) |
| Dynamo `DcCkfRankReplacement` | `Stored([11,12,13]) -> Removed([12,13,99]) -> Stored([14])` | Prunes `2` zombie candidates `[1012, 1013]`, retaining **`[1011, 1014]`** |
| Dynamo `RetryManager::rebase_usage` | Client prompt `3` tokens, `4` replayed across 2 migrations, worker reports `(P=7, C=2, cached=7)` | Rebases to **`(P'=3, C'=6, total=9, cached'=3)`** and clears partial `completion_tokens_details` |
