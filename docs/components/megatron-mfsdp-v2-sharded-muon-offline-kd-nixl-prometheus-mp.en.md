# Megatron-LM MFSDP v2 Sharded Muon, Offline KD v2 Monolith & Per-Document CP Zigzag, and NIXL Multi-Process Prometheus Exporter

> **中文版**: [megatron-mfsdp-v2-sharded-muon-offline-kd-nixl-prometheus-mp.md](megatron-mfsdp-v2-sharded-muon-offline-kd-nixl-prometheus-mp.md)

## 1. Primary Source Evidence & Core Claims

| Repository & Commit | Evidence Mode | Source Paths | Mechanism & Quantitative Finding |
|---|---|---|---|
| `NVIDIA/Megatron-LM` `#8313` (`91d774de79`, 2026-10-20) | `implementation` | `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/muon.py`, `tests/unit_tests/distributed/mfsdp_v2/test_muon.py` | Implements MFSDP v2 `FsdpMuon` with sharded `DTensor` momentum (`RowAtomic`), balanced owner-compute Newton-Schulz orthogonalization via `assign_owner_work` + `gather`/`scatter`, and bitwise loss parity (`atol=0, rtol=0`) against unsharded Muon. |
| `NVIDIA/Megatron-LM` `#5482` (`fce1e2051f`, 2026-10-20) | `implementation` | `megatron/training/distillation/{logits_saver.py,utils.py}`, `megatron/training/checkpointing.py` | Upgrades Offline Logits KD to `LOGPROBS_FORMAT_VERSION = 2` with sample-range tars (`dp{d}__{start}-{end}.tar`), 1-bit/element 17th-bit packing via `v2_pack_indices` (`-87.5%` 17th-bit storage), per-document CP zigzag slicing/reassembly (`_document_zigzag_index`), `check_logits_saver_failure()`, and `tp0/cp0/dp0` tracker ownership. |
| `ai-dynamo/nixl` `#2405` (`60dd8b7cd0`, 2026-10-20) | `implementation` | `src/core/nixl_plugin_manager.cpp`, `docs/telemetry.md` | Maps `NIXL_TELEMETRY_EXPORTER=prometheus` to `prometheus_mp` (`libtelemetry_exporter_prometheus_mp.so`) and removes the single-process exporter so all TP ranks on a host export behind one shared scrape endpoint (fixing NVBug 6575930). |

## 2. Implementation Path, Formulas & Distributed Protocols

### 2.1 MFSDP v2 `FsdpMuon` Sharded Momentum & Owner-Compute Orthogonalization (`#8313`)
`FsdpMuon` updates sharded 2D parameter momentum locally across all four weight-decay modes (`decoupled`, `independent`, `l2`, `palm`) and optional Nesterov interpolation, gathers `momentum_inputs` to balanced owner ranks (`assign_owner_work`), runs `self.orthogonalize` on owners, scatters updates to non-owners while owners slice `full_updates[i].view(-1).narrow(0, offset, weight.numel())`, and refreshes model weights via `sync_model_weight_from_main_weight()`.

### 2.2 Offline Logits KD v2: 17-Bit Index Packing & Per-Document CP Zigzag (`#5482`)
- **`v2_pack_indices` / `v2_unpack_indices`**: Packs the 17th vocabulary bit across the concatenated per-iteration monolith `[seq, samples_per_dp_per_iter, K]` via `np.packbits` (`1 bit/elem` vs `torch.bool` `1 byte/elem`, saving `87.5%` on the 17th bit and `29.17%` total index bytes).
- **`_document_zigzag_index`**: Splits each packed document $[s_i, e_i)$ into $2 \cdot \text{CP}$ equal chunks and assigns chunks $[r, 2\cdot\text{CP}-r-1]$ to CP rank $r$, skipping zero-length right-padded boundaries ($e_i \le s_i$).
- **Tracker Ownership**: Attaches `progress_finalize_fn` under `skip_weight_ckpt` to `tp_rank == 0 and cp_rank == 0 and dp_rank == 0` on the last PP stage rather than `is_last_rank()`.

### 2.3 NIXL Multi-Process Prometheus Alias (`#2405`)
`loadTelemetryPlugin("prometheus")` resolves `file_name = "prometheus_mp"` and caches the single `dlopen` handle under both `"prometheus"` and `"prometheus_mp"`, aggregating all local ranks via `/dev/shm/nixl-telemetry-<euid>-<port>`.

## 3. Deterministic Verification

- **Verifier**: `python experiments\megatron_mfsdp_v2_muon_offline_kd_nixl_prometheus\run.py --check`
