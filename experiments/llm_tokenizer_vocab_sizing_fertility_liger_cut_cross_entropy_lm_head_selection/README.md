# LLM Tokenizer Vocab Sizing (`32K vs. 152K`) & Fused Linear Cut Cross-Entropy (`Liger FLCE / Apple CCE`) Experiment

Reproducible analytical model comparing 4 tokenizer vocabulary and LM Head loss architectures across `10,000` Chinese/Code characters in inference and `32,768` tokens/GPU in training:
- **Tier 1 (`32K BPE` + Standard `nn.CrossEntropyLoss`)**: `14,500` tokens per `10k` Chinese chars (`1.0x` baseline speed), `8.4 GB` logits HBM.
- **Tier 2 (`152K Multilingual BPE` + Standard `nn.CrossEntropyLoss`)**: `6,500` tokens per `10k` Chinese chars (`2.23x` inference speedup & `-55%` KV Cache), **but `39.8 GB` logits HBM explosion in training (`OOM` under FSDP2)**.
- **Tier 3 (`152K Multilingual BPE` + `Megatron Vocab-Parallel Cross-Entropy` at `TP=8`)**: `5.0 GB` logits HBM, requires `TP=8`.
- **Tier 4 (`152K Multilingual BPE` + `Fused Linear Cut Cross-Entropy Liger FLCE / Apple CCE`, Recommended)**: `6,500` tokens (`2.23x` inference speedup) and only **`0.12 GB` logits HBM (`331x` memory reduction, works at `TP=1` FSDP2)**.

## Run & Verify

```bash
python experiments/llm_tokenizer_vocab_sizing_fertility_liger_cut_cross_entropy_lm_head_selection/run.py
python experiments/llm_tokenizer_vocab_sizing_fertility_liger_cut_cross_entropy_lm_head_selection/run.py --check
```
