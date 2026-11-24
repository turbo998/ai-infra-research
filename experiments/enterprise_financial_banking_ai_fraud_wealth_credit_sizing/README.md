# Experiment #601: Enterprise Financial & Banking AI Platform Sizer

Reproduces the `29ms` inline P99 fraud scoring budget (`11ms` XGBoost + `18ms` `1.5B-FP8` Single-Token Logprob Scorer on `6%` grey-zone transactions) and the `14`-GPU unified banking fleet (`4× L40S + 2× H100 + 8× H200` = `$35,600/mo`, `-94.5%` vs. naive inline `70B`) (`docs/components/enterprise-financial-banking-ai-fraud-wealth-credit-sizing-faq.md`).
