# Mooncake RDMA Auto-GID 7-Class Probing (`PeerAutoGidStateTable`), Two-Endpoint Failure Confirmation State Machine, and Bounded In-Flight Scatter Resharding (`MooncakeTransferEngineExecutor`)

> **中文版**: [mooncake-rdma-auto-gid-reprobe-bounded-reshard.md](mooncake-rdma-auto-gid-reprobe-bounded-reshard.md) | **Experiment**: [`experiments/mooncake_gid_reprobe_reshard_sim/README.md`](../../experiments/mooncake_gid_reprobe_reshard_sim/README.md)

### Executive Summary

Mooncake's RDMA transfer engine (commit `5ea6cf0d95`), Reshard executor (commit `0977fc7066`), and Store client service (commit `67df8c414a`) introduce a **7-tier GID candidate classifier (`AutoGidCandidateClass`, promoting RFC1918 `10/8` private IPv4-mapped GIDs above `fe80::` link-local IPv6)**, an **epoch-isolated two-endpoint failure confirmation state machine (`PeerAutoGidStateTable::recordCompletion`, returning `kRetrySameGid` on the first failing endpoint and advancing via `kAdvanceGid` with `auto_gid_reprobe_from` peer advisory only when a strictly newer `endpoint_id > first_failed_endpoint_id` fails)**, and a **bounded in-flight Scatter sliding window (`max_inflight_batches`) with `_TransferSubmissionState.POISONED` DMA lifetime isolation**. Together, these mechanisms eliminate RoCEv2 multi-subnet/overlay GID blackholes and lift 64 GiB cross-node weight resharding goodput on 400GbE RoCEv2 from `313.03 Gbps` (`1.7562 s` at `W=1`) to **`357.85 Gbps` (`1.5363 s` at `W=4`, `+14.3%` speedup within a bounded `1,024 MiB` pinned window)**.

### Upstream Source Responsibility & Locations

| Module & File | Upstream Repo & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `mooncake-transfer-engine/include/transport/rdma_transport/rdma_gid_probe.h` | `kvcache-ai/Mooncake` commit `5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10` / blob `eb0c5fc7710d425a510ece1fc651c452945c5176` | Defines 7-tier `AutoGidCandidateClass`, 5-state `AutoGidDataPathAction`, circular cursor `reselectAutoGidCandidate`, and `PeerAutoGidStateTable`. |
| `mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp` | `kvcache-ai/Mooncake` commit `5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10` / blob `78916020bbd0284ec45f7546409677db2f8b41bf` | Populates `auto_gid_reprobe_from` in `fillLocalHandshakeDesc` and intentionally retains DMA notification buffers via `.release()` if QP/MR teardown fails. |
| `mooncake-reshard/python/mooncake/reshard/transfer_engine/executor.py` | `kvcache-ai/Mooncake` commit `0977fc7066ece59f93440a1db59cb805906fe182` / blob `ec05a8ba8f19ab07c149ab92d3511432a831fc71` | Implements `MooncakeTransferEngineExecutor._execute_batches` bounded `max_inflight_batches` window, Scatter ticket validation, and `POISONED` submission state. |
| `mooncake-store/src/client_service.cpp` | `kvcache-ai/Mooncake` commit `67df8c414ad04ec415f69da351e6fc92774e8a6f` / blob `fab103e035ed84ac0079a2bf343bb9db7898cd68` | Implements `GetContiguousSliceRange` pointer coalescing and protocol-specific MR limits (`GetTransportRegistrationLimit` for `efa`/`cxi`/`ub`). |

### Core Implementation Path, Memory Layout & Synchronization Contracts

1. **7-Tier RoCEv2 GID Ranking (`rdma_gid_probe.h:32-47`)**:
   - Splits `kNetworkPrivateV4 = 2` (RFC1918 `10/8`, `172.16/12`, `192.168/16` and CGNAT IPv4-mapped GIDs) above `kNetworkDegraded = 3` (link-local `fe80::` and container overlays), ensuring datacenter RoCEv2 fabrics select routable/same-subnet RFC1918 addresses before unroutable `fe80::` entries.
2. **Two-Endpoint Confirmation & Peer Advisory Re-Probing (`rdma_gid_probe.h:145-183, 299-350`)**:
   - `recordCompletion` filters stale completions via `attempt.selection_epoch != state.selection_epoch` (`kIgnoreDuplicate`).
   - On the first failing endpoint (`first_failed_endpoint_id == 0`), it records `first_failed_endpoint_id = attempt.endpoint_id` and returns `kRetrySameGid`.
   - Duplicate failures from the same or older endpoint (`attempt.endpoint_id <= first_failed_endpoint_id`) return `kIgnoreDuplicate`.
   - Only when a reconstructed endpoint with `attempt.endpoint_id > first_failed_endpoint_id` fails on the same epoch does it return `kAdvanceGid`, advancing the cursor in the next `prepare()` and advertising `auto_gid_reprobe_from` so the remote peer also advances its GID selection (granting `state.remaining_advances = 1` even if the peer's local advance budget was `0`).
3. **Bounded In-Flight Scatter Execution & DMA Poisoning (`executor.py:355-515`)**:
   - `_execute_batches` maintains `published: deque[_PublishedBatch]` capped at `max_inflight_batches`, draining the oldest ticket via `drain_published(published.popleft())` before submitting a new Scatter batch.
   - If any completion drain leaves the transfer state unresolved (`TransferCompletionUnknownError` or `TransferCompletionInterrupted`), `TransferSubmission` transitions to `_TransferSubmissionState.POISONED`, preventing subsequent batches or premature buffer reclamation while RNIC DMA may still be in flight.

### Key Equations, Units & Engineering Assumptions

1. **Bounded Sliding-Window Reshard Goodput & Footprint**:
   $$T_{\text{reshard}}(W) = N_{\text{batches}} \cdot \left( \frac{B_{\text{batch}}}{\text{BW}_{\text{wire}}} + (t_{\text{fence}} + t_{\text{submit}}) \cdot \max\!\left(0.04,\, 1 - \min(1.0, 0.42(W - 1))\right) \right)$$
   $$M_{\text{window}}(W) = W \times B_{\text{batch}}, \qquad \text{Goodput}(W) = \frac{N_{\text{batches}} \cdot B_{\text{batch}} \cdot 8}{T_{\text{reshard}}(W)}$$

### Deterministic Verification (`experiments/mooncake_gid_reprobe_reshard_sim`)

Executed via `python experiments/mooncake_gid_reprobe_reshard_sim/run.py --check` (`experiments/mooncake_gid_reprobe_reshard_sim/results/summary.json`):

| `max_inflight_batches` ($W$) | Peak In-Flight Observed | Completed Batches | 64 GiB Reshard Time (`total_reshard_time_s`) | Effective Goodput (`effective_goodput_gbps`) | Pinned Window Footprint (`pinned_window_footprint_mib`) | State on Unknown CQE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` (Sync Baseline) | `1` | `256` | `1.7562 s` | `313.03 Gbps` | `256 MiB` | `POISONED` |
| `2` | `2` | `256` | `1.6600 s` | `331.18 Gbps` | `512 MiB` | `POISONED` |
| **`4` (Sweet Spot)** | **`4`** | **`256`** | **`1.5363 s`** | **`357.85 Gbps` (`+14.3%`)** | **`1,024 MiB`** | **`POISONED`** |
| `8` | `8` | `256` | `1.5363 s` | `357.85 Gbps` | `2,048 MiB` | `POISONED` |
| `16` | `16` | `256` | `1.5363 s` | `357.85 Gbps` | `4,096 MiB` | `POISONED` |

### Architect Design Exercises & Falsifiable Questions

1. **Why must `recordCompletion` require `attempt.endpoint_id > state.first_failed_endpoint_id` before advancing the GID rather than counting 2 failed CQEs?**
   - **Answer**: A single multi-QP endpoint with `max_wr_depth` in-flight WRs flushes dozens of error CQEs when a link drops. Counting CQEs would exhaust `max_advances` on the first transient flap; requiring a strictly higher `endpoint_id` proves a freshly reconstructed endpoint also failed on the same GID epoch.
2. **Falsifiable Hypothesis**: Can `RdmaEndPoint::~RdmaEndPoint()` safely free `notify_.send_buffer` and `notify_.recv_buffer` when `deconstructLocked()` fails to destroy a QP or deregister an MR?
   - **Verification**: `rdma_endpoint.cpp:105-112` explicitly calls `.release()` on both unique_ptrs to leak the buffers intentionally rather than risking silent host memory corruption from a still-attached RNIC DMA context.
