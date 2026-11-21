# Enterprise SFT Fine-Tuning Architecture Selection Experiment (`LoRA+ / DoRA + Multipack Block-Diagonal Packing + Liger-Kernel Fused Cross-Entropy` vs. `Padded Full SFT`)

Reproducible throughput, peak HBM, and cost model comparing 4 Supervised Fine-Tuning (`SFT`) architectures on a `70B/72B` model (`152K` vocabulary, `200,000` variable-length instruction examples).

## Run
```bash
python experiments/sft_fine_tuning_neftune_sequence_packing_qlora_vs_full_ft_lora_plus_selection/run.py
python experiments/sft_fine_tuning_neftune_sequence_packing_qlora_vs_full_ft_lora_plus_selection/run.py --check
```
