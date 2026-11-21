# TPU & AI Hypercomputer — 7-Day Sprint

From data center fabric to JAX: a one-week path for a cloud / network architect moving into accelerator clusters — system anatomy, distributed training, inference & profiling, cluster control planes, and site turn-up.

**Time budget**: ≈20 hours: weekday evenings 2–2.5 h, one light reading night (1.5 h), one 6-hour hands-on block.

> Web version: https://turbo998.github.io/workshops/tpu-7day-sprint/index.en.html

## The week

| Day | Topic | Time |
|---|---|---|
| 1 | TPU system anatomy & where TPUs now run | 2.5h |
| 2 | Distributed training on TPU: JAX, XLA and sharding | 2.5h |
| 3 | Inference on TPU + performance profiling | 2.5h |
| 4 | Cluster control planes: TPU VM, GKE, Slurm, storage | 2.5h |
| 5 | The physical layer: site qualification, turn-up, operations | 1.5h (reading) |
| 6 | Hands-on lab day | 6h (hands-on) |
| 7 | Synthesis: explain it end to end | 2.5h |

### Day 1 — TPU system anatomy & where TPUs now run

*Build the mental model chip → host → cube → slice → pod → multislice, and the market shift of TPUs into partner and customer data centers.*

**Learn**
- Chip: MXU systolic arrays, TensorCore vs SparseCore, HBM; TPU7x (Ironwood) has 2 TensorCores + 4 SparseCores per chip and a dual-chiplet programming model
- Generations & roles: v5e (cost-efficient), v5p (training, 8,960-chip pod), v6e Trillium (256-chip pod), TPU7x Ironwood (9,216-chip superpod, 2,307 TF BF16 / 4,614 TF FP8 per chip), 8th gen split into TPU 8t (training) and TPU 8i (inference)
- Topology vocabulary: 4x4x4 cube, 3D torus with wrap-around only at full cube, accelerator type vs accelerator config, single-host vs multi-host slices (TPU7x: 4 chips per VM)
- Interconnect: ICI inside a slice, optical circuit switches (OCS) between cubes, ICI resiliency routing around optical faults, Multislice over the data-center network (DCN)
- Market context: TPU systems now deployed outside Google-run facilities — partner-operated sites (on-site setup, cabling, burn-in, acceptance testing), TPU-as-a-service ventures, and customer-premises systems for confidential / sovereign workloads


**Read**
- [TPU system architecture (docs)](https://docs.cloud.google.com/tpu/docs/system-architecture)
- [TPU7x (Ironwood) (docs)](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Inside the Ironwood TPU co-designed AI stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)
- [Eighth-generation TPUs: TPU 8t & 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)
- [How to Scale Your Model — ch. 1–2 (rooflines, TPUs)](https://jax-ml.github.io/scaling-book/)

**Deliverable**: A one-page cheat sheet: hierarchy diagram drawn from memory + a table of v5e / v5p / v6e / TPU7x / 8t / 8i with pod size, per-chip compute, HBM and target workload.

**Self-check**
- [ ] What travels over ICI and what over DCN in a 2-slice job?
- [ ] Why does a 4x4x8 slice behave differently from a 4x4x4 one?
- [ ] What fails, and what keeps running, when one OCS port goes dark?


### Day 2 — Distributed training on TPU: JAX, XLA and sharding

*Map parallelism strategies onto a device mesh and know which axis belongs on ICI vs DCN.*

**Learn**
- Parallelism menu: data, FSDP (ZeRO-3 style), tensor, pipeline, expert, sequence / context — and their collective (all-reduce, all-gather, reduce-scatter, all-to-all)
- JAX building blocks: jax.jit, Mesh / NamedSharding / PartitionSpec, shard_map, XLA GSPMD → Shardy partitioner
- Rule of thumb: bandwidth-hungry axes (tensor, FSDP) on ICI; data parallel across slices over DCN
- Reference stacks: MaxText (JAX, open-source LLM training), PyTorch paths (PyTorch/XLA, torchax), Pathways for very large multi-slice jobs
- Back-of-envelope: training FLOPs ≈ 6 × params × tokens; MFU = achieved / peak FLOPs


**Read**
- [How to Scale Your Model — sharding & transformer math chapters](https://jax-ml.github.io/scaling-book/)
- [JAX: distributed arrays & automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [MaxText](https://github.com/AI-Hypercomputer/maxtext)
- [Cloud TPU Multislice overview](https://docs.cloud.google.com/tpu/docs/multislice-introduction)

**Deliverable**: A worked sizing: chips needed to train a 70B dense model on 15T tokens in 30 days at 40% MFU on TPU7x — plus the mesh layout you would propose.

**Self-check**
- [ ] Why is FSDP usually preferred to pure data parallel for a 70B model?
- [ ] Which collective dominates tensor parallelism, and why must it stay on ICI?
- [ ] What changes when you move a PyTorch training loop to TPU?


### Day 3 — Inference on TPU + performance profiling

*Serve LLMs on TPU, size HBM for weights + KV cache, and triage bottlenecks with a profiler.*

**Learn**
- vLLM TPU via tpu-inference: one JAX lowering path for JAX and PyTorch (torchax) models; recommended generations v5e / v6e / TPU7x; chunked prefill and DCN-based prefill/decode disaggregation
- Prefill is compute-bound, decode is memory-bandwidth-bound; batch size, quantization (int8 / FP8) and KV cache drive cost per token
- KV sizing: bytes/token = 2 × layers × KV heads × head_dim × bytes; e.g. a 70B GQA model (80 layers, 8 KV heads, d=128, BF16) ≈ 0.33 MB per token
- Profiling: jax.profiler → XProf (trace viewer, HLO op profile, step-time breakdown); classify a step as compute-, memory-, collective- or input(infeed)-bound
- Custom kernels: Pallas for TPU (e.g. ragged paged attention) — when a kernel is worth writing vs a flag / sharding change


**Read**
- [vLLM TPU: unified backend for PyTorch and JAX](https://vllm.ai/blog/vllm-tpu)
- [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)
- [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)
- [XProf](https://github.com/openxla/xprof)
- [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)

**Deliverable**: A bottleneck triage tree (symptom → profiler evidence → fix) and an HBM budget for serving a 70B model at 32K context.

**Self-check**
- [ ] A 70B BF16 model is 140 GB — how many chips for weights + 32 concurrent 32K sequences?
- [ ] Profile shows long gaps between steps: what do you suspect first?
- [ ] When would you choose P/D disaggregation?


### Day 4 — Cluster control planes: TPU VM, GKE, Slurm, storage

*Know how jobs get onto accelerators, how capacity is obtained, and how training survives failures.*

**Learn**
- Provisioning: TPU VMs and queued resources; capacity modes (reservations, flex-start, spot) and long-term reservations
- GKE: TPU node pools (single- and multi-host), JobSet for gang-launched multi-host jobs, Kueue (incl. topology-aware scheduling) for quotas and queues, Multislice on GKE
- Slurm on Google Cloud via Cluster Toolkit — for HPC-native research teams
- Linux host layer: libtpu runtime, tpu-info for chip utilization / HBM, driver and image management
- Data & checkpoints: Cloud Storage + GCS FUSE, Hyperdisk ML, Managed Lustre; Orbax async checkpointing; goodput = productive time / total time
- Field case study (GPU, same control plane): DWS flex-start lifecycle — ~10-minute booking window, consume-provisioning-request annotations, 7-day cap; labels are not scheduling constraints (pin zones with topology.kubernetes.io/zone); reservations + COMPACT placement when the fabric must not span zones


**Read**
- [TPUs on GKE (docs)](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus)
- [JobSet](https://github.com/kubernetes-sigs/jobset)
- [Kueue](https://kueue.sigs.k8s.io/)
- [Cluster Toolkit (Slurm)](https://github.com/GoogleCloudPlatform/cluster-toolkit)
- [TPU machine types & capacity](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)
- [Case study: hypercomputer-training-jobs — DWS, Kueue, DCGM, NCCL smoke on GKE](https://github.com/turbo998/hypercomputer-training-jobs)

**Deliverable**: A reference architecture for a 2-slice training job on GKE (Kueue + JobSet + Multislice + GCS FUSE + Orbax) and a GKE-vs-Slurm decision table.

**Self-check**
- [ ] Why does a multi-host TPU job need gang scheduling?
- [ ] A slice loses a host at hour 30 — walk through recovery.
- [ ] When is Slurm the better answer than GKE?


### Day 5 — The physical layer: site qualification, turn-up, operations

*Understand what it takes to land an accelerator pod in a data center someone else operates.*

**Learn**
- Site qualification: power capacity and redundancy, rack density, liquid cooling (CDUs, facility water loop, leak detection), floor loading, fiber plant and DCN uplinks, physical security, logistics
- Pod layout: cube and rack placement, optical cabling to OCS, power and cooling zones as failure domains
- System turn-up: rack & stack, cabling validation, burn-in, firmware baseline, ICI link health, collective-bandwidth tests, a reference training job as acceptance gate
- Operational readiness: out-of-band management (BMC, Redfish / IPMI), telemetry into Prometheus (host, accelerator, facility), alerting, hardware repair & RMA flow, spares, remote-hands SLA
- Why this matters to goodput: failure domains + spares + fast repair + checkpointing = usable capacity


**Read**
- [Ironwood GA announcement (liquid cooling, fleet uptime)](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads/)
- [DMTF Redfish](https://www.dmtf.org/standards/redfish)
- [Open Compute Project](https://www.opencompute.org/)
- [SemiAnalysis: TPUv7 and TPU externalization](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the)

**Deliverable**: Reading notes only — tomorrow you turn them into the turn-up & readiness checklist.

**Self-check**
- [ ] Name five things you must confirm before a liquid-cooled pod can ship to a site.
- [ ] What is the acceptance gate between ‘racked’ and ‘handed to customer’?


### Day 6 — Hands-on lab day

*Touch real TPUs, measure, profile, and write the operational artifacts.*

**Learn**
- Lab A (1.5h) — Free TPU runtime (Kaggle or Colab; check the TPU type offered): list devices, build a mesh, run a sharded BF16 matmul, compute achieved TFLOP/s vs peak
- Lab B (1.5h) — Capture a trace with jax.profiler, open it in XProf, label each region of the step (compute / collective / idle)
- Lab C (1.5h, optional with cloud credits) — Serve a small model (e.g. Qwen3-4B) with vLLM TPU on a single-host v6e; record TTFT and tokens/s at 3 batch sizes
- Lab D (1.5h) — Write the TPU site turn-up & operational-readiness checklist and the Prometheus metric catalog (host / accelerator / OOB / facility)
- Lab E (optional, cloud credits) — Recreate the DWS flex-start flow on a small GKE pool (L4 GPU or a single-host TPU): ProvisioningRequest + consuming pod, then a Kueue ResourceFlavor / ClusterQueue / LocalQueue with an AdmissionCheck; write it up as a one-page postmortem-style note


**Lab A/B starter code**

```python
import time, jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

devs = jax.devices(); n = len(devs); print(devs)
mesh = jax.make_mesh((n,), ("x",))
N = 8192
a = jax.device_put(jnp.ones((N, N), jnp.bfloat16), NamedSharding(mesh, P("x", None)))
b = jax.device_put(jnp.ones((N, N), jnp.bfloat16), NamedSharding(mesh, P(None, None)))
f = jax.jit(lambda a, b: a @ b)
f(a, b).block_until_ready()                      # compile + warm-up
t = time.perf_counter()
for _ in range(10): c = f(a, b)
c.block_until_ready()
dt = (time.perf_counter() - t) / 10
tf = 2 * N**3 / dt / 1e12
print(f"{tf:.1f} TFLOP/s total, {tf/n:.1f} per device")

jax.profiler.start_trace("/tmp/trace")           # Lab B: open with XProf
for _ in range(5): f(a, b).block_until_ready()
jax.profiler.stop_trace()
```


**Read**
- [vLLM TPU quickstart (tpu-inference)](https://github.com/vllm-project/tpu-inference)
- [JAX profiling](https://docs.jax.dev/en/latest/profiling.html)

**Deliverable**: A lab notebook with numbers (TFLOP/s, a labelled trace screenshot, optional TTFT table) and two operational artifacts: the turn-up checklist and the metric catalog.

**Self-check**
- [ ] Why is your measured TFLOP/s below peak, and which part of the trace proves it?


### Day 7 — Synthesis: explain it end to end

*Turn the week into a story you can tell at a whiteboard in 10 minutes.*

**Learn**
- Whiteboard scenario: a regional customer wants a 4,096-chip TPU7x deployment in a partner data center — qualification → pod layout → turn-up → workload migration → operations
- GPU → TPU migration narrative: what moves easily (JAX, vLLM-served models), what needs work (custom CUDA kernels), how you de-risk with a PoC
- Fill the gaps you found in days 1–6; re-run the self-check questions without notes


**Read**
- [How to Scale Your Model — inference & practice chapters](https://jax-ml.github.io/scaling-book/)

**Deliverable**: A 10-minute whiteboard walkthrough, recorded once and reviewed.

**Self-check**
- [ ] Can you explain ICI vs DCN, MFU, KV sizing and the turn-up gate in plain language to a CIO?


## Worked numbers

- **Training sizing (Day 2)**: 6 × 70e9 × 15e12 ≈ 6.3e24 FLOPs. TPU7x peak 2,307 TF BF16 × 40% MFU ≈ 0.92 PF/chip ≈ 8.0e19 FLOPs/chip-day → ≈ 79,000 chip-days → ≈ 2,600 chips for 30 days.
- **KV sizing (Day 3)**: 2 × 80 × 8 × 128 × 2 B ≈ 0.33 MB/token → 32K context ≈ 10.7 GB per sequence → 32 concurrent sequences ≈ 344 GB, plus 140 GB BF16 weights ≈ 484 GB before activations/overhead.
- **Interpretation**: Weights alone exceed one chip’s HBM, so serving needs tensor parallelism across several chips (or FP8 weights) — and concurrency is limited by KV cache, not compute.


## Turn-up & operational-readiness checklist

**Site qualification**
- [ ] Power capacity & redundancy confirmed for target rack density
- [ ] Liquid cooling: CDU capacity & redundancy, facility water spec, leak detection
- [ ] Floor loading, rack footprint and service clearance
- [ ] Fiber plant, DCN uplinks and cross-connects
- [ ] Physical security & access control; import / logistics path

**Turn-up**
- [ ] Rack & stack against the approved pod layout
- [ ] Optical cabling validated end to end (ICI / OCS / DCN)
- [ ] Firmware & software baseline applied
- [ ] Burn-in and hardware diagnostics passed
- [ ] ICI link health and collective-bandwidth tests within spec (the GPU analogue: DCGM diagnostics + NCCL all_reduce_perf smoke gate)
- [ ] Reference training job reaches target step time — acceptance gate

**Operational readiness**
- [ ] OOB management reachable (BMC via Redfish / IPMI)
- [ ] Telemetry flowing to Prometheus: host, accelerator, OOB sensors, facility (CDU / PDU)
- [ ] Alert routing & escalation matrix agreed
- [ ] Hardware repair / RMA flow, spares pool, remote-hands SLA
- [ ] Checkpoint & restart runbook rehearsed; goodput target defined


## Context: where TPUs are going

- [Ironwood GA (Nov 2025)](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads/)
- [TPU 8t / 8i (Apr 2026)](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)
- [TPUs offered to AI cloud providers (DCD)](https://www.datacenterdynamics.com/en/news/google-offers-its-tpus-to-ai-cloud-providers-report/)
- [TPUs in customers’ own data centers (Nov 2026)](https://fool.com/investing/2026/11/12/should-you-buy-alphabet-stock-now-that-google-will-let-customers-run-its-ai-chips-in-their-own-data-centers)
