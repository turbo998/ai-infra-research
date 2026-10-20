# Mooncake RDMA Auto-GID Re-Probing & Bounded Scatter Resharding Simulator

Deterministic Python 3.12 standard-library simulator reproducing `PeerAutoGidStateTable` 7-class candidate ranking, two-endpoint failure verification (`kRetrySameGid` -> `kAdvanceGid`), peer advisory epoch handshakes, and `MooncakeTransferEngineExecutor` bounded in-flight Scatter resharding from `kvcache-ai/Mooncake` commits `5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10`, `0977fc7066ece59f93440a1db59cb805906fe182`, and `67df8c414ad04ec415f69da351e6fc92774e8a6f`.

## Run & Verify

```bash
python experiments/mooncake_gid_reprobe_reshard_sim/run.py --check
```
