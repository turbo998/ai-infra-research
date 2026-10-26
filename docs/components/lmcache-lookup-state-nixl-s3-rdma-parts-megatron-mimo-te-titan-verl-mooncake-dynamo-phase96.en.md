# LMCache Callback LookupState, NIXL Multi-Part S3-over-RDMA, Mooncake Namespace Eviction Framework, verl Sampler Validation & Dynamo Shared ZMQ Context (Phase 96)

> **Evidence Classification**: `implementation` (pinned upstream commits) + `own-offline` (deterministic Python 3.12 standard-library verification; not GPU/RDMA hardware profiling).

## 1. Primary Upstream Source Evidence (10 Zero-Overlap Commits)

| Source ID | Repo / Commit | Key Implementation Path & Architectural Mechanism |
|---|---|---|
| `lmcache-mp-callback-driven-lookup-state-scheduler-pr5960-2026` | `LMCache/LMCache` `ff9185d419` (PR #5960) | Introduces `LookupState` driven by `MessagingFuture` callbacks across `LOOKUP` and `QUERY_PREFETCH_STATUS`, automatically releasing unretrieved server tail locks. |
| `lmcache-core-l1-l2-allowed-list-prefetch-storage-manager-pr5922-2026` | `LMCache/LMCache` `a81b4e5aea` (PR #5922) | Adds L1 and L2 memory-tier allowed-list filtering when submitting prefetch requests to the storage manager. |
| `nixl-obj-move-rdma-objects-larger-cuobject-in-parts-pr2406-2026` | `ai-dynamo/nixl` `99d469be5d` (PR #2406) | Splits S3-over-RDMA transfers exceeding `CUOBJ_MAX_MEMORY_REG_SIZE` ($4\text{ GiB} - 64\text{ KiB} = 4,294,901,760\text{ B}$) into bounded parts so `cuMemObjGetRDMAToken` never fails with `CU_OBJ_INVALID_VALUE`. |
| `nixl-core-preserve-active-request-state-release-reject-pr2175-2026` | `ai-dynamo/nixl` `5b565cb9a6` (PR #2175) | Stores `engine->releaseReqH` in a local `release_status` variable inside `nixlAgent::releaseXferReq` so active transfers remain `NIXL_IN_PROG` when backend release is rejected. |
| `megatron-honor-rope-args-nemotron-mimo-provider-pr8371-2026` | `NVIDIA/Megatron-LM` `964b1deefb` (PR #8371) | Propagates `position_embedding_type`, `rotary_base`, `rotary_percent`, and `seq_len_interpolation_factor` in the Nemotron MIMO language provider instead of hard-coding `"none"`. |
| `te-jax-fix-undefined-sr-rng-state-dry-run-encoder-pr3498-2026` | `NVIDIA/TransformerEngine` `f3826223b7` (PR #3498) | Binds `SR_KEY: sr_rng` in JAX single-GPU multiprocessing and model-parallel encoder `--dry-run` initialization. |
| `torchtitan-use-torch-accelerator-device-stream-memory-pr5450-2026` | `pytorch/torchtitan` `35f4a2e166` (PR #5450) | Replaces `torch.cuda` calls with device-agnostic `torch.accelerator`, `torch.Stream`, and `torch.Event` across EMA offload, RL weight staging, and `graph_trainer`. |
| `verl-validate-sampler-prob-records-before-support-align-8eac5b0-2026` | `volcengine/verl` `8eac5b0170` | Validates unique `head_ids`, enforces $\sum \exp(\text{head\_log\_probs}) \le 1.0 + 10^{-5}$, and checks sampled token logprob consistency before top-k support alignment. |
| `mooncake-store-eviction-framework-over-namespace-layer-pr4811-2026` | `kvcache-ai/Mooncake` `fff8e90952` (PR #4811) | Adds per-namespace `Evictor`, `EvictionPolicy` (hard-pin/active-lease protection + two-pass soft-pin deferral), and `OffloadingReclaimer` (disk-before-memory demotion). |
| `dynamo-kv-router-share-zmq-contexts-dial-replay-per-gap-pr17253-2026` | `ai-dynamo/dynamo` `8222ff098c` (PR #17253) | Shares a process-wide `OnceLock<zmq::Context>` (`DYN_ZMQ_IO_THREADS=4`) across KV-event SUB sockets and dials ephemeral DEALER replay sockets per sequence gap. |

## 2. Mathematical Modeling & Architect Exercises

1. **NIXL Multi-Part S3-over-RDMA Splitting**:
   A $10\text{ GiB}$ object is partitioned against $L_{\max} = 4\text{ GiB} - 64\text{ KiB} = 4,294,901,760\text{ B}$ into $3$ parts (`[4294901760, 4294901760, 2147614720]` bytes), avoiding `CU_OBJ_INVALID_VALUE` on multi-GiB CUDA 13.4 arenas.
2. **Dynamo Shared ZMQ Context & Mooncake Two-Pass Namespace Eviction**:
   Sharing one 4-thread ZMQ context across 32 subscriber sockets reduces I/O threads from `128` to `4` (`32.0x` reduction). Mooncake's two-pass `Evictor` reclaims unpinned expired-lease objects first (`t1/k1` demoted directly via existing disk replica) and soft-pinned expired objects second (`t1/k2` offloaded to disk then demoted), leaving hard-pinned (`t1/k3`) and active-lease (`t1/k4`) entries untouched.

## 3. Reproducible Offline Verification

```bash
python experiments/lmcache_lookup_state_nixl_s3_rdma_parts_megatron_mimo_te_titan_verl_mooncake_dynamo_phase96/run.py --check
```
Verified in `summary.json`: `num_rdma_parts == 3`, `status_after_rejected_release == "NIXL_IN_PROG"`, `thread_reduction_ratio == 32.0`, and `impossible_mass_rejected == True`.
