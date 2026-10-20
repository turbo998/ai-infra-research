# FlashInfer Rubin MoE Finalize Barrier, EP Split Route Exclusion & SM103a Peer-Outer All-Gather Matmul

Deterministic Python 3.12 standard-library model verifying:
1. `flashinfer-ai/flashinfer` commit `91b201a670ea5d86df8e0e4d601a510f606eac01` (`#6307`): Inserting `cute.arch.cp_async_bulk_commit_group()` + `cute.arch.cp_async_bulk_wait_group(0, read=True)` before `self.epilog_sync_barrier.arrive_and_wait()` eliminates `7/8` (`87.5%`) shared-memory `sC` overwrite races during Rubin MoE finalize reductions.
2. `flashinfer-ai/flashinfer` commit `69f082fe759f23e3e37d457da9075c11c4e15eee` (`#6840`): Mapping masked non-local picks to `masked_expert = (local_expert_offset + num_local_experts) % num_experts` excludes non-local routes from local expert GEMM execution (`6,821` vs `12,288` evaluations across 3 CUDA Graph replays, `-44.49%`, and `0` evaluations on all-remote replays).
3. `flashinfer-ai/flashinfer` commits `41b39c4af9ea7e3cad075effb97b571a57d51b78` (`#6841`) and `3bf6d3202239a107989451257c6f594cb5d2f008` (`#6740`): SM103a `po` (`peer-outer`, form code `4`) persistent walk at `[4, "bfloat16", "k_major", 128, 14336, "po"]` cuts peer-readiness transitions from `224` to `4` (`-98.21%`), and `tcgen05.wait::ld.sync.aligned;` before `barrier.sync 9, 288;` eliminates `9/9` consumer-warp TMEM deallocation hazards.

## Run & Check
```bash
python experiments/flashinfer_rubin_moe_barrier_ep_split_ag_matmul/run.py --check
```
