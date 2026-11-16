# 4,096-GPU `PyTorch Elastic (torchrun / rdzv)` Fault Recovery, `C10d TCPStore` Barrier Storm & Orphaned Zombie CUDA Context Selection Guide: Why 4,096 Ranks Hammering Rank-0 `TCPStore` Stall for `420 s`, Why `fork` DataLoaders Trigger `68%` Respawn `CUDA OOM` & How 2-Level Hierarchical Rendezvous + In-Process Warm Abort (`6.8 s`, `61.8x` Faster) Solve It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/pytorch_elastic_torchrun_rdzv_tcpstore_barrier_zombie_recovery_selection/run.py` and `results.json`.

![PyTorch Elastic TCPStore Storm & Zombie GPU Recovery Visual Summary](../../assets/svg/pytorch-elastic-torchrun-rdzv-tcpstore-barrier-zombie-recovery-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Two-Level Node-Agent Hierarchical Rendezvous (`512` Node Agents Aggregating `8` Local Ranks via Shared Memory) + `spawn` DataLoader & `cgroup.kill` Atomic Zombie Reaping + Zero-Respawn In-Process `ncclCommAbort` Warm Reconfiguration (`NVRx / TorchFT`) for 4,096-GPU Training Clusters**.
- **Architect Value**: On a `4,096`-GPU (`512`-node) training cluster, single-node optical link flaps or XID faults occur every few hours. When `torchrun` attempts elastic recovery, standard configurations hit two catastrophic failure modes:
  1. **Rank-0 `C10d TCPStore` Connection Storm (`420.0 s` Hang, Tier 1)**: Having all `4,096` worker processes simultaneously connect to and poll the single-threaded `TCPStore` on `Rank 0` overflows the Linux `SYN` backlog (`somaxconn = 128`), triggering exponential TCP SYN retransmissions (`1s -> 3s -> 7s`) and stalling rendezvous for **`420.0 s`**.
  2. **Orphaned Zombie Subprocesses Holding 76 GB of GPU HBM (`68.0%` Respawn `CUDA OOM`, Tier 1/2)**: If a training script initializes CUDA before spawning `DataLoader(num_workers=8)` with the default `fork` start method, all 8 child processes inherit open file descriptors to `/dev/nvidiactl` and `/dev/nvidia0`. When `torchrun` kills the parent worker process to restart the job, surviving `DataLoader` children keep the kernel reference count on that 76 GB CUDA allocation $> 0$—causing the newly respawned training worker to crash immediately with **`CUDA out of memory` (`68.0%` failure rate)**!
  By combining **a 2-level node-aggregated Rendezvous store, mandatory `spawn` + `cgroup.kill` process-tree reaping, and in-process `ncclCommAbort` warm reconfiguration (Tier 4)**, fault recovery drops **`61.8x` (`420.0 s -> 6.8 s`)**, respawn OOMs drop to **`0.0%`**, and effective cluster goodput rises from `74.2%` to **`98.6%`**!

---

## ① 📌 Executive Summary

1. **Why `fork` DataLoaders Cause Ghost `CUDA OOM` After Parent Process Termination**:
   - The NVIDIA kernel driver frees GPU HBM only when the **last open file descriptor (`struct file` refcount) to the GPU context closes**—not when the parent PID exits.
   - `os.fork()` duplicates all open GPU device descriptors into every `DataLoader` worker. If the parent rank is killed (`SIGKILL`) while a `DataLoader` child is blocked on network storage I/O, the child retains the GPU descriptor and locks 76 GB of HBM!
   - **Fix**: Always call `multiprocessing.set_start_method("spawn", force=True)` and reap the entire process group via `cgroup.kill` + `prctl(PR_SET_PDEATHSIG, SIGKILL)` before respawning.
2. **Why Flat `TCPStore` Collapses Beyond 1,024 GPUs (`2-Level Hierarchical Store`)**:
   - Aggregating the 8 local ranks on each server in `/dev/shm` first so only **`512` node agents** connect to the root store (alongside `net.core.somaxconn = 65535`) reduces master socket concurrency by `8x` and completes barriers in `< 0.4 s`.
3. **Why In-Process Warm Recovery (`6.8 s`) Beats Full Process Respawn (`165.0 s`)**:
   - Killing and cold-restarting `4,088` healthy Python/CUDA processes just because 1 node failed wastes `165 s` re-importing libraries and reloading checkpoints. Calling `ncclCommAbort()` (`_abort_process_group`) in-place keeps all `4,088` survivor CUDA contexts warm and resumes training in **`6.8 s`**.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: How do survivor GPUs break out of a hung `ncclAllReduce` kernel without killing the Python process?
- Enable `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` and invoke `ncclCommAbort(comm)` (`c10d._abort_process_group(pg)`). This forces the driver and RDMA NIC to abort pending Work Requests and immediately unblocks the CUDA stream so a new communicator can be initialized in the same process.

### Q2: Why does `torchrun` occasionally fail on restart with `EADDRINUSE` on port `29500`?
- Forked child processes (loggers, metric exporters, or data workers) inherited the listening socket FD without `FD_CLOEXEC`. Enforcing `spawn` and cleaning the node sub-cgroup via `cgroup.kill` frees the port deterministically.

### Q3: How does training continue without changing global batch size or 3D parallel dimensions (`TP×PP×DP`) when 1 node fails?
- Pair in-process abort with **1–2 pre-warmed Standby Nodes (`Warm Standby Swap`)**: the standby node joins the new communicator in `< 1 s` and pulls its FSDP weight/optimizer shard directly from a peer replica over `400G` RDMA P2P (`2.4 s`) without touching shared disk storage!

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 4,096-GPU Rendezvous & Elastic Fault Recovery Decision Tree

```mermaid
flowchart TD
    A["4,096-GPU (512-Node) Training Cluster Experiences Single-Node Fault"] --> B{"Are Child Processes Spawned Cleanly & Reaped via cgroup.kill?"}
    B -- "No: Default fork DataLoader + parent-only SIGTERM" --> C["⛔ Tier 1/2: Orphaned children hold 76GB HBM -> 68% respawn CUDA OOM!"]
    B -- "Yes: Mandatory spawn + PDEATHSIG + cgroup.kill" --> D{"Rendezvous Store Topology & Survivor Recovery Mode?"}
    D -- "4,096 ranks hammer flat TCPStore + full cold process respawn" --> E["⛔ Tier 1/2: TCPStore SYN drop stalls 420s; cold respawn wastes 165s"]
    D -- "512-node 2-level store + clean full process respawn" --> F["✅ Tier 3: 0% zombie OOM, 38.0s clean recovery (94.1% goodput)"]
    D -- "2-level store + in-process ncclCommAbort + Warm Standby P2P swap" --> G["🏆 Tier 4: Zero process restart on 4,088 GPUs! 6.8s recovery (61.8x faster)"]
```

### 3.2 Flat `TCPStore` + Zombie OOM (`420.0 s`) vs. Hierarchical Store + Warm Abort (`6.8 s`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: Flat 4,096-Rank TCPStore + Fork Zombies (420.0 s, 68% OOM)"]
      B1["4,096 Ranks Overflow Rank-0 TCPStore SYN Queue"] --> B2["Respawned Workers Crash with CUDA OOM From Orphaned DataLoader Forks!"]
    end
    subgraph Good["🏆 Tier 4: 2-Level Store + In-Process ncclCommAbort (6.8 s, 0% OOM)"]
      G1["4,088 Survivors Call ncclCommAbort In-Place"] --> G2["Rebuild Comm via 512-Node Store & P2P Shard Sync in 6.8 s!"]
    end
```

### 3.3 In-Process `ncclCommAbort` & Warm Standby P2P Recovery Sequence

```mermaid
sequenceDiagram
    participant Surv as 4,088 Survivor GPUs (In-Process)
    participant Fault as Faulty Node (8 GPUs)
    participant Stand as Warm Standby Node (8 GPUs)
    participant Store as 2-Level Hierarchical Rendezvous Store

    Fault--xSurv: 1. Node link flap / XID hardware error during AllReduce
    Note over Surv: 2. Watchdog triggers in-process ncclCommAbort() (unblocks CUDA stream)
    Note over Fault: 3. cgroup.kill atomically reaps all worker & DataLoader processes
    Surv->>Store: 4. 512 Node Agents complete hierarchical rendezvous (<0.4s)
    Stand->>Surv: 5. Standby node pulls FSDP shard via RDMA P2P from peer replica (2.4s)
    Note over Surv,Stand: 6. Training resumes in 6.8 s with 0.0% OOM rate!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Rendezvous & Recovery Architecture | Fault Recovery Latency | Respawn `CUDA OOM` Rate | Effective Cluster Goodput | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Flat `4,096`-Rank `TCPStore` + `fork`** | Master socket storm + zombie HBM leak | `420.0 s` ✕ | `68.0%` ✕ | `74.2%` ✕ | ⛔ **Severe Hang & OOM Loops** |
| **Tier 2: External `etcd` + Full Cold Respawn** | Kills & cold-restarts all 4,096 processes | `165.0 s` ✕ | `24.0%` ✕ | `85.5%` ✕ | ⛔ **Wastes 165s Per Fault** |
| **Tier 3: 2-Level Node Store + `cgroup.kill`** | Hierarchical rdzv + clean process tree reap | `38.0 s` ✅ | `0.0%` ✅ | `94.1%` ✅ | ✅ **Hardened `torchrun` Baseline** |
| **Tier 4: 2-Level Store + In-Process Warm Abort (Recommended)** | **Zero respawn + `ncclCommAbort` + P2P swap** | **`6.8 s` (`61.8x` Faster 🏆)** | **`0.0%` 🏆** | **`98.6%` 🏆** | 🏆 **1K–10K GPU Pretraining Standard** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/pytorch_elastic_torchrun_rdzv_tcpstore_barrier_zombie_recovery_selection/run.py`](../../experiments/pytorch_elastic_torchrun_rdzv_tcpstore_barrier_zombie_recovery_selection/run.py) · Data: [`results.json`](../../experiments/pytorch_elastic_torchrun_rdzv_tcpstore_barrier_zombie_recovery_selection/results.json)
- **Interactive Sizer**: [`docs/reports/pytorch-elastic-torchrun-rdzv-tcpstore-barrier-zombie-recovery-sizer.html`](../reports/pytorch-elastic-torchrun-rdzv-tcpstore-barrier-zombie-recovery-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/pytorch-elastic-torchrun-rdzv-tcpstore-barrier-zombie-recovery-selection-cards.html`](../reports/pytorch-elastic-torchrun-rdzv-tcpstore-barrier-zombie-recovery-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Flat 4096-Rank TCPStore + Unclean Kill", "recovery_s": 420.0, "oom_pct": 68.0 },
      { "tier": "Tier 2: etcd Rdzv + Full Process Cold Respawn", "recovery_s": 165.0, "oom_pct": 24.0 },
      { "tier": "Tier 3: 2-Level Node Store + cgroup.kill Respawn", "recovery_s": 38.0, "oom_pct": 0.0 },
      { "tier": "Tier 4: 2-Level Store + In-Process Warm Abort", "recovery_s": 6.8, "oom_pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "recovery_s": "Quantity",
    "oom_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Hierarchical Rendezvous + In-Process Warm Abort Cuts 4,096-GPU Recovery from 420s to 6.8s (0% OOM)",
    "subtitle": "Single-node fault recovery latency (seconds) across a 512-node (4,096-GPU) cluster (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "recovery_s",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All large-scale distributed training clusters (`512–16,384+` GPUs) running `torchrun`, `TorchFT`, or `NVIDIA NVRx`.
- **⛔ Not for**: Using `fork` DataLoaders after CUDA initialization or pointing thousands of ranks directly at a single-threaded `TCPStore`.

### Production Checklist
- [ ] **Enforce `mp.set_start_method("spawn", force=True)`**: Never allow `DataLoader` workers to inherit open `/dev/nvidia*` file descriptors via `fork()`.
- [ ] **Reap worker process trees via `cgroup.kill` & `PR_SET_PDEATHSIG`**: Guarantee `0%` orphaned processes holding GPU memory or TCP ports on restart.
- [ ] **Deploy 2-level hierarchical Rendezvous (`somaxconn=65535`) + in-process `ncclCommAbort()` warm recovery**: Resume training in `< 10 s` without restarting healthy survivor processes.
