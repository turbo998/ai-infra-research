# Round 326: SFT & RL Gradient Accumulation Loss Reduction Bias (`reduction='mean' / K` vs. Global Valid-Token Sum) Selection

Deterministic evaluator comparing 4 gradient accumulation and data-parallel loss normalization policies across short-vs-long token gradient weight ratio (`x`), true global-batch deviation (`%`), and long-form accuracy (`%`).
