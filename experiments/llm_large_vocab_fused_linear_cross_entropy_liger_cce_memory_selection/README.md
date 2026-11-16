# Round 301: Large-Vocab (152K–256K) Fused Linear Cross-Entropy (Liger-Kernel / Cut-Cross-Entropy) Selection

Deterministic evaluator comparing 4 `lm_head + CrossEntropyLoss` execution architectures at `32K` sequence length (`V=152,064`) across peak logits/CE memory (`GB`), step latency (`ms`), and training throughput (`tok/s`).
