# LLM Hallucination Detection & Citation Grounding Experiment (`LettuceDetect / HHEM` vs. `Logprob` vs. `LLM-as-a-Judge`)

Reproducible analytical model comparing 4 real-time RAG hallucination detection architectures across `1,000,000` daily enterprise RAG queries (`4,096` context tokens + `350` output tokens):
- **Tier 1 (`Synchronous 70B LLM-as-a-Judge`)**: `91.5%` recall, `+1,850 ms` blocking P99 latency, `$255,000/month` verification cost.
- **Tier 2 (`Raw Token Logprob Heuristic`)**: `52.0%` recall, `28.0%` false-positive rate (`+0.5 ms`).
- **Tier 3 (`Token-Level NLI Span Detector LettuceDetect / HHEM-2.1`)**: `88.5%` recall, `4.2%` false-positive rate, `+18.0 ms` latency, `$4,200/month`.
- **Tier 4 (`Streaming Sentence NLI + Deterministic Number/Entity Guard + 6% Gray-Zone Judge Escalation`, Recommended)**: `96.8%` recall, `1.1%` false-positive rate, `+16.0 ms` net tail latency, and `$19,500/month` (`-92.4%` cost vs. Tier 1).

## Run & Verify

```bash
python experiments/llm_hallucination_detection_attribution_hhem_lettucedetect_logprob_entropy_selection/run.py
python experiments/llm_hallucination_detection_attribution_hhem_lettucedetect_logprob_entropy_selection/run.py --check
```
