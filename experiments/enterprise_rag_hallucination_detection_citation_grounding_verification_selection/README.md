# Enterprise RAG Hallucination Prevention, Inline Citation Grounding (`HHEM / MiniCheck NLI`) & Numeric Verification (`R508`)

Deterministic quantitative evaluation model comparing four enterprise RAG hallucination-detection and citation-grounding architectures across **25,000,000 monthly RAG responses**:
- **Tier 1**: Unconstrained Prompt Instructions ("Only Answer Using Provided Context and Cite Sources")
- **Tier 2**: Synchronous Secondary `70B / GPT-4o LLM-as-a-Judge` Blocking Every Full Response
- **Tier 3**: Post-Hoc Embedding Cosine Similarity Matching Between Generated Sentences and Chunks
- **Tier 4**: Constrained Integer Citation IDs (`[^1]..[^K]`) + Sentence-Streaming `HHEM-2.1 / MiniCheck NLI` (`28ms` on L4) + Deterministic Numeric/Date Set-Intersection Gate

## Verification

```bash
python experiments/enterprise_rag_hallucination_detection_citation_grounding_verification_selection/run.py --check
python -m unittest tests/test_enterprise_rag_hallucination_detection_citation_grounding_verification_selection.py
```
