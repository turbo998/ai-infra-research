# PyTorch `CUDASymmetricMemory` 与 NVLS `multimem` 集合通信算例

复现 `pytorch/pytorch` (`CUDASymmetricMemoryOps.cu` 与 `CUDASymmetricMemory-inl.cuh`) 的无锁系统级信号垫 `sync_remote_blocks`、Unicast P2P `one_shot`/`two_shot` 与 NVLS `multimem` 硬件多播延迟、流量倍数及 BF16/FP32 数值漂移边界。

## 运行

```powershell
python experiments\symm_mem_multimem_collective\run.py --check
```
