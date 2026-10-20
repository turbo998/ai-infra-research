# PyTorch CUDA Copy-on-Write (`COW`) Completion Events & `c10d` `ProcessGroupNCCLLazy` MemPool Forwarding Experiment

Deterministic Python 3.12 standard-library model verifying `pytorch/pytorch` commits `20f82b619b` (`#200800`) and `47bc8be46b` (`#201432`):
- Allocation-stream `cudaMemcpyAsync` + `waitStream` eliminates `8/8` stale CUDA Graph replay reads caused by legacy default-stream `cudaMemcpy`.
- Explicit `torch._C._cow_record_done(tensor)` records `COWCompletion` events and releases `cow_retired_data`, enabling in-place last-sibling stealing (`256.0 MiB` vs `512.0 MiB` peak memory, `-50.0%`; `10.0 ms` vs `45.0 ms` stream wait stall, `-77.78%`).
- `ProcessGroupNCCLLazy.register_mem_pool(pool, symm)` / `deregister_mem_pool(pool)` forwards symmetric window lifecycle to the primary communicator (`0` leaked pools on teardown).

## Run & Verify

```powershell
python experiments\pytorch_cuda_cow_nccl_lazy_mempool\run.py --check
```
