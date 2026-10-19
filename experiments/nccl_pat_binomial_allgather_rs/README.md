# NCCL PAT (`NCCL_ALGO_PAT`) Parallel Aggregated Trees AllGather & ReduceScatter Simulator

Deterministic CPU/offline simulator grounded in `NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`):
- `src/include/collectives.h` (`PatRSAlgorithm<T>`, `PatAGAlgorithm<T>`, `mirrorInvert`, `firstBitSet`)
- `src/device/all_gather.h` & `src/device/reduce_scatter.h` (`RunWorkColl<..., NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE>`)

## Usage

```bash
python experiments/nccl_pat_binomial_allgather_rs/run.py
python experiments/nccl_pat_binomial_allgather_rs/run.py --check
```
