# Megatron-LM MFSDP v2 Sharded Muon, Offline KD v2 Monolith & Per-Document CP Zigzag, and NIXL Multi-Process Prometheus Exporter

Deterministic Python 3.12 standard-library model verifying:
1. Megatron-LM PR #8313 (`91d774de79`): `FsdpMuon` sharded `DBuffer` momentum, 4 weight-decay modes (`decoupled`, `independent`, `l2`, `palm`), Nesterov `lerp`, and owner-compute orthogonalization + `narrow` local slice application.
2. Megatron-LM PR #5482 (`fce1e2051f`): Offline Logits KD v2 (`LOGPROBS_FORMAT_VERSION = 2`) 17-bit index packing (`v2_pack_indices` 1-bit/element 17th bit saving 87.5% vs `torch.bool`), per-document CP zigzag slicing/reassembly (`_document_zigzag_index`), and `tp_rank == 0 and cp_rank == 0 and dp_rank == 0` progress tracker ownership.
3. NVIDIA NIXL PR #2405 (`60dd8b7cd0`): `loadTelemetryPlugin()` aliasing `prometheus` -> `prometheus_mp` (`libtelemetry_exporter_prometheus_mp.so`) behind one shared scrape endpoint.
