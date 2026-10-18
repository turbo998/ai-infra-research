# NCCL Zero-Copy User Buffer Registration (`ncclCommRegister` / `CUMEM` Fabric Handles) Experiment

Deterministic offline simulator (`measurement_class: own-offline`) grounded in `NVIDIA/nccl` (`commit 3e46c089f4ae4732722a419eb6096aead7e47a30`):
- `src/register/register.cc` (`blob 8ea5b4fa0601970883b42fc8914cf8ec6ca1d92e`)
- `src/register/coll_reg.cc` (`blob d41a33894b179d7f1a984f29c223be34a2affd80`)
- `src/allocator.cc` (`blob 417dd64bc8259b41848fea7b58ee77f1c6da0c84`)
- `src/transport/nvls.cc` (`blob 155f6adf30223e19fb52fa4ed1c30b2014f27c44`)

## Usage
```bash
python experiments/nccl_zero_copy_registration/run.py
python experiments/nccl_zero_copy_registration/run.py --check
```
