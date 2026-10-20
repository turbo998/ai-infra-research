# Experiment: TransformerEngine Bounded Symmetric All-Reduce & NVFP4 Dynamo Metadata + DeepGEMM Mega-MHC CUDA Graph Epoch Barrier

## Purpose (研究目的 / Bilingual Overview)
- **中文**：使用 Python 3.12 标准库复现并验证 `NVIDIA/TransformerEngine` 对称内存 All-Reduce 四分幂次分桶缓存（`#3709`）、NVFP4 `torch.compile` / Dynamo 假张量（`FakeTensor`）Swizzled Scale 对齐推导与 SM120 RHT 矩阵转发（`#3628`、`#3669`），以及 `deepseek-ai/DeepGEMM` `sm100_mega_mhc` 在 CUDA Graph 重放下的 128B 首行尾部 `Epoch` + `exit_count` 屏障状态机（`#478`）。
- **English**: Deterministic Python 3.12 standard-library model verifying `NVIDIA/TransformerEngine` bounded quarter-power-of-two symmetric all-reduce scratch caching (`#3709`), NVFP4 `torch.compile` / Dynamo fake execution swizzled-scale metadata alignment and SM120 RHT matrix forwarding (`#3628`, `#3669`), and `deepseek-ai/DeepGEMM` `sm100_mega_mhc` CUDA Graph split-barrier `Epoch` retirement (`#478`).

## Run & Check
```bash
python experiments/te_nvfp4_symm_ar_deepgemm_mhc_epoch/run.py --check
```
