# vLLM V1 KV Cache 量化（`FP8`、`INT4 + RHT + 零点隐写术`、`KIVI`、`NVFP4`）算例

复现 `vllm-project/vllm` (`int4_per_token_head.py` 与 `triton_reshape_and_cache_flash.py`) 的随机 Walsh-Hadamard 旋转 (`single_rht`)、FP32 Scale 尾数零点隐写术 (`((k_scale_bits & -16) | (k_zp_int & 0xF))`) 与 7 种 KV Cache 量化格式在 18x Key 离群通道下的注意力保真度与 H200 容量/TPOT 收益。

## 运行

```powershell
python experiments\kv_cache_quant_kivi_nvfp4\run.py --check
```
