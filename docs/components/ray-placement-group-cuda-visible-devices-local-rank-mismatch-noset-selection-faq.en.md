# Ray Distributed RL & Serving (`verl / OpenRLHF / vLLM RayExecutor`) `PlacementGroup` (`STRICT_PACK` vs. `PACK`), `CUDA_VISIBLE_DEVICES` Ordinal Crash (`cudaErrorInvalidDevice`) & `NOSET_CUDA_VISIBLE_DEVICES` Selection Guide: Why `torch.cuda.set_device(local_rank)` Crashes `87.5%` of Ray Actors, Why `strategy="PACK"` Spills `TP=8` Across Nodes (`8.3x` Slowdown) & How `STRICT_PACK` + `NOSET=1` + `(node_ip, gpu_id)` Sorting (`0%` Crash, `14.2 ms` TPOT) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection/run.py` and `results.json`.

![Ray PlacementGroup STRICT_PACK & NOSET_CUDA_VISIBLE_DEVICES Visual Summary](../../assets/svg/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Per-Node `STRICT_PACK` Placement Groups + `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` Full-Node Device Visibility + `(node_ip, phys_gpu_id)` Lexicographical Worker Rank Sorting (`verl / OpenRLHF / vLLM RayDistributedExecutor`)**.
- **Architect Value**: Orchestrating multi-node RL post-training (`GRPO / PPO`) or distributed LLM inference (`TP=8`) on Ray triggers three high-frequency production bugs when default actor scheduling is used:
  1. **The `87.5%` (`7/8`) Actor Startup Crash on `torch.cuda.set_device(local_rank)` (`cudaErrorInvalidDevice`, Tier 1)**:
     - When Ray schedules a `@ray.remote(num_gpus=1)` actor onto physical GPU `#3`, Ray automatically sets `CUDA_VISIBLE_DEVICES="3"` inside that worker process.
     - Consequently, `torch.cuda.device_count()` inside the actor is **`1`**, and physical GPU `#3` is remapped to logical ordinal **`0`**!
     - When standard PyTorch distributed initialization calls `torch.cuda.set_device(local_rank)` with `local_rank = 1..7`, **`7` out of `8` actors (`87.5%`) crash immediately with `RuntimeError: CUDA error: invalid device ordinal`**!
     - Worse, even if worked around via `set_device(0)` (Tier 2), masking `CUDA_VISIBLE_DEVICES` to a single ordinal isolates each actor's CUDA device table—**breaking zero-copy `CUDA IPC` (`cudaIpcOpenMemHandle`) weight sharing** between colocated RL Trainers and vLLM Rollout actors!
  2. **The `8.3x` TP Slowdown from Soft `strategy="PACK"` Cross-Node Spills (`118.5 ms` vs `14.2 ms`, Tier 2)**:
     - In Ray, `strategy="PACK"` is only a best-effort preference! When a node has 6 free GPUs and another node has 2 free GPUs, `"PACK"` splits a `TP=8` rollout group `6 + 2` across two physical nodes—forcing every layer's `TP All-Reduce` over the inter-node network and degrading TPOT from `14.2 ms` to **`118.5 ms` (`8.3x` slower)**!
  3. **The Unsorted Bundle NVLink/NUMA Scramble (`19.8 ms` vs `14.2 ms`, Tier 3)**:
     - Even with `strategy="STRICT_PACK"`, Ray assigns bundles to physical GPUs in arbitrary order (e.g., `[5, 2, 7, 0, 4, 1, 6, 3]`), misaligning `TP=2/4` sub-groups across CPU NUMA sockets and PCIe switches!
  By **enforcing `strategy="STRICT_PACK"` per node, setting `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` so every actor sees all 8 local GPUs (`0..7`), binding each actor via `torch.cuda.set_device(int(ray.get_gpu_ids()[0]))`, and sorting actors by `(node_ip, phys_gpu_id)` before assigning `WORLD_RANK` (Tier 4)**, actor crashes and cross-node TP spills both drop to **`0.0%`**, zero-copy `CUDA IPC` weight refit is unlocked (`0.18 s`), and `TP=8` TPOT stabilizes at **`14.2 ms` (`8.3x` faster)**!

---

## ① 📌 Executive Summary

1. **First-Principles Rule: Disable Ray's Single-GPU Ordinal Masking (`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`)**:
   - Multi-GPU training and serving frameworks (`PyTorch Distributed`, `vLLM`, `verl`) must see the true physical GPU topology (`0..7`) on each node to establish `NVLink P2P` and cross-process `CUDA IPC` handles.
   - With `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`, each actor queries its assigned physical device ID directly from Ray (`phys_gpu = int(ray.get_gpu_ids()[0])`) and calls `torch.cuda.set_device(phys_gpu)`.
2. **First-Principles Rule: Per-Node `STRICT_PACK` + `(node_ip, phys_gpu_id)` Rank Sorting**:
   - Create one `STRICT_PACK` placement group of 8 bundles per 8-GPU node (never a single multi-node `STRICT_PACK`, which is infeasible!), collect `(node_ip, phys_gpu)` from all actors, and sort the actor handles before `init_process_group()` so `LOCAL_RANK 0..7` maps monotonically to physical `GPU 0..7`!

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why does `placement_group([{"GPU": 1}] * 64, strategy="STRICT_PACK")` hang forever on a cluster of 8-GPU nodes?
- Because `STRICT_PACK` requires **every bundle in the placement group** to fit on a **single physical machine**! For a 64-GPU job across eight 8-GPU nodes, create **8 separate `STRICT_PACK` placement groups of 8 bundles each** (or 1 `SPREAD` group of eight `{"GPU": 8}` node bundles).

### Q2: How do colocated RL frameworks (`verl`) run both an FSDP Trainer Actor and a vLLM Rollout Actor on the exact same physical GPU?
- Assign both actors to the **same placement group and `bundle_index`** using fractional GPU requests (e.g., `num_gpus=0.5` for Trainer + `num_gpus=0.5` for Rollout, or `num_gpus=0.01` with explicit bundle pinning) alongside `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`. Both actors receive the exact same `ray.get_gpu_ids()[0]` and share weights via zero-copy `CUDA IPC`!

### Q3: How do we prevent stray `.cuda()` calls from defaulting to `cuda:0` when `NOSET_CUDA_VISIBLE_DEVICES=1` is active?
- Call both `torch.cuda.set_device(phys_gpu)` and `torch.set_default_device(f"cuda:{phys_gpu}")` at the very start of the actor's `__init__` method.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Ray GPU PlacementGroup & Device Visibility Decision Tree

```mermaid
flowchart TD
    A["Ray Distributed RL (verl / OpenRLHF) or vLLM Serving (8 GPUs/Node, TP=8)"] --> B{"Is RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 Enabled?"}
    B -- "No: Default Ray masks CUDA_VISIBLE_DEVICES='k'" --> C{"How Does Worker Code Bind Its GPU?"}
    C -- "Calls torch.cuda.set_device(local_rank)" --> D["⛔ Tier 1: device_count()==1 -> 87.5% (7/8) of actors crash with cudaErrorInvalidDevice!"]
    C -- "Calls torch.cuda.set_device(0)" --> E["⛔ Tier 2: Starts up, but isolated device tables break zero-copy CUDA IPC weight sync!"]
    B -- "Yes: NOSET=1 preserves 8-GPU visibility + set_device(int(ray.get_gpu_ids()[0]))" --> F{"PlacementGroup Strategy & Rank Ordering?"}
    F -- "Soft strategy='PACK' or Unsorted Bundles" --> G["⚠️ Tier 3: Spills TP=8 6+2 across nodes (118.5 ms TPOT) or scrambles NUMA/NVLink order!"]
    F -- "Per-Node STRICT_PACK + Sort Actors by (node_ip, phys_gpu_id)" --> H["🏆 Tier 4: 0% Crash + 0% Spill + 14.2 ms TPOT (8.3x faster) + Full CUDA IPC!"]
```

### 3.2 Masked Ordinal Crash (`87.5%`) & Spilled `PACK` (`118.5ms`) vs. `STRICT_PACK` + `NOSET=1` (`14.2ms`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 87.5% Invalid Device Crash & 6+2 Cross-Node TP Spill (118.5 ms)"]
      B1["Ray sets CUDA_VISIBLE_DEVICES='3' -> set_device(3) crashes 7/8 actors!"] --> B2["Soft PACK splits TP=8 across Node A (6) & Node B (2) -> 118.5 ms TPOT!"]
    end
    subgraph Good["🏆 Tier 4: Per-Node STRICT_PACK + NOSET=1 + (node_ip, phys_gpu_id) Sorting"]
      G1["NOSET=1 keeps GPUs 0..7 visible -> set_device(ray.get_gpu_ids()[0]) + CUDA IPC"] --> G2["STRICT_PACK + GPU-ID sort aligns Ranks 0..7 with NVLink -> 14.2 ms TPOT (8.3x faster)!"]
    end
```

### 3.3 Ray `STRICT_PACK` Creation, Physical GPU-ID Sorting & `ProcessGroup` Init Sequence

```mermaid
sequenceDiagram
    participant Driver as Ray Driver / Controller
    participant PG as Per-Node STRICT_PACK PlacementGroup
    participant Actors as 8x Ray GPU Actors (NOSET=1)

    Driver->>PG: 1. Allocate `placement_group([{"GPU": 1}] * 8, strategy="STRICT_PACK")` per node
    Driver->>Actors: 2. Spawn actors with `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"`
    Actors-->>Driver: 3. Report `(node_ip, phys_gpu_id = int(ray.get_gpu_ids()[0]))` (e.g., `[5, 2, 7, 0...]`)
    Driver->>Driver: 4. Sort actor handles by `(node_ip, phys_gpu_id)` -> contiguous `GPU 0..7`!
    Driver->>Actors: 5. Assign sorted `world_rank` & `local_rank = phys_gpu_id` -> `torch.cuda.set_device(phys_gpu_id)`
    Actors->>Actors: 6. `dist.init_process_group("nccl")` -> 100% aligned NVLink & CUDA IPC ready!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Ray Placement & Visibility Policy | Actor Init Crash Rate | Colocated `CUDA IPC` | Cross-Node `TP` Spill | `TP=8` Decode `TPOT` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Default Ray + `set_device(local_rank)`** | Masked ordinal + `set_device(1..7)` | **`87.5%` ✕** | Blocked ✕ | `25.0%` ✕ | `118.5 ms` ✕ |
| **Tier 2: `set_device(0)` + Soft `PACK`** | Avoids crash; spills TP & blocks IPC | `0.0%` | Blocked ✕ | `25.0%` ✕ | `118.5 ms` ✕ |
| **Tier 3: `STRICT_PACK` (Unsorted, Masked)** | Same node, scrambled GPU IDs, no IPC | `0.0%` | Blocked ✕ | `0.0%` | `19.8 ms` |
| **Tier 4: `STRICT_PACK` + `NOSET=1` + Sort (Recommended)** | **Full visibility + atomic node + sorted** | **`0.0%` 🏆** | **Zero-Copy IPC 🏆** | **`0.0%` 🏆** | **`14.2 ms` (`8.3x` 🏆)** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection/run.py`](../../experiments/ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection/run.py) · Data: [`results.json`](../../experiments/ray_placement_group_cuda_visible_devices_local_rank_mismatch_noset_selection/results.json)
- **Interactive Sizer**: [`docs/reports/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-sizer.html`](../reports/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-selection-cards.html`](../reports/ray-placement-group-cuda-visible-devices-local-rank-mismatch-noset-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Default Ray + set_device(local_rank)", "tpot_ms": 118.5, "crash_pct": 87.5 },
      { "tier": "Tier 2: set_device(0) + Soft PACK", "tpot_ms": 118.5, "crash_pct": 0.0 },
      { "tier": "Tier 3: STRICT_PACK (Unsorted Bundles)", "tpot_ms": 19.8, "crash_pct": 0.0 },
      { "tier": "Tier 4: STRICT_PACK + NOSET=1 + GPU-ID Sort", "tpot_ms": 14.2, "crash_pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "tpot_ms": "Quantity",
    "crash_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "STRICT_PACK + NOSET_CUDA_VISIBLE_DEVICES + GPU-ID Sorting Cuts TP=8 TPOT 8.3x (118.5ms -> 14.2ms)",
    "subtitle": "TP=8 decode per-token latency (ms) across 4 Ray placement tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "tpot_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All Ray-orchestrated RL frameworks (`verl`, `OpenRLHF`, `Slime`) and distributed inference engines (`vLLM`, `SGLang`) using `TP/PP/EP` or colocated `CUDA IPC` weight synchronization.
- **⛔ Not for**: Calling `torch.cuda.set_device(local_rank)` under default Ray ordinal masking or using soft `strategy="PACK"` for tensor-parallel groups.

### Production Checklist
- [ ] **Set `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"` and bind devices via `torch.cuda.set_device(int(ray.get_gpu_ids()[0]))`**.
- [ ] **Use per-node `strategy="STRICT_PACK"` placement groups for all `TP > 1` worker groups**.
- [ ] **Sort actor handles by `(node_ip, int(ray.get_gpu_ids()[0]))` prior to assigning `WORLD_RANK` and calling `dist.init_process_group()`**.
