# Multi-LoRA Serving Architecture Selection Experiment (`vLLM / SGLang Punica & S-LoRA` vs. Merged-Weight Pods)

Reproducible capacity and TCO model comparing 4 architectures for serving `500` tenant-specific LoRA adapters (`r=32`, `320 MB` each) on a shared `Llama-3.3-70B-FP8` base model across required `H200` GPUs, cold-adapter swap `TTFT` penalty, `TPOT` BGMV overhead, and monthly TCO.

## Run
```bash
python experiments/multi_lora_serving_slora_punica_vllm_lora_adapter_hot_swapping_selection/run.py
python experiments/multi_lora_serving_slora_punica_vllm_lora_adapter_hot_swapping_selection/run.py --check
```
