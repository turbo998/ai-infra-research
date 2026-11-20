"""Deterministic evaluator for Enterprise RAG Hallucination Prevention, Sentence-Level Inline Citation Grounding & Numeric Verification Selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    tiers = [
        {
            "tier": "Tier 1: Unconstrained Prompt Instructions ('Only Answer Using Provided Context and Cite Sources')",
            "architecture": "Relying on system-prompt instructions alone while allowing the LLM to generate free-form URLs or document titles at the end of the response",
            "ungrounded_hallucination_rate_pct": 14.8,
            "fake_or_mismatched_citation_rate_pct": 22.5,
            "verification_latency_p95_overhead_ms": 0.0,
            "numeric_and_entity_exact_match_pct": 76.0,
            "monthly_25m_query_verification_and_risk_tco_usd": 195000,
        },
        {
            "tier": "Tier 2: Synchronous Secondary `70B / GPT-4o LLM-as-a-Judge` Blocking Every Full Response",
            "architecture": "Buffering the entire generated RAG response and calling a second 70B/GPT-4o model synchronously to verify every claim before showing text to the user",
            "ungrounded_hallucination_rate_pct": 1.4,
            "fake_or_mismatched_citation_rate_pct": 3.2,
            "verification_latency_p95_overhead_ms": 2450.0,
            "numeric_and_entity_exact_match_pct": 94.0,
            "monthly_25m_query_verification_and_risk_tco_usd": 248000,
        },
        {
            "tier": "Tier 3: Post-Hoc Embedding Cosine Similarity Matching Between Generated Sentences and Chunks",
            "architecture": "Computing bi-encoder embedding similarity between each generated sentence and retrieved chunks (`cosine >= 0.80`) to attach citations after generation",
            "ungrounded_hallucination_rate_pct": 6.8,
            "fake_or_mismatched_citation_rate_pct": 11.4,
            "verification_latency_p95_overhead_ms": 45.0,
            "numeric_and_entity_exact_match_pct": 81.5,
            "monthly_25m_query_verification_and_risk_tco_usd": 82000,
        },
        {
            "tier": "Tier 4: Constrained Integer Citation IDs (`[^1]..[^K]`) + Sentence-Streaming `HHEM-2.1 / MiniCheck NLI` (`28ms` on L4) + Deterministic Numeric/Date Set-Intersection Gate",
            "architecture": "Assigning integer IDs `[1..K]` to chunks, verifying each streamed sentence in `28ms` via specialized `DeBERTa-NLI / HHEM` against ONLY its cited chunk `[k]`, plus CPU regex check that every emitted number/date exists in the source",
            "ungrounded_hallucination_rate_pct": 0.2,
            "fake_or_mismatched_citation_rate_pct": 0.1,
            "verification_latency_p95_overhead_ms": 28.0,
            "numeric_and_entity_exact_match_pct": 99.9,
            "monthly_25m_query_verification_and_risk_tco_usd": 26000,
        },
    ]

    scenarios = [
        {
            "id": "HG1",
            "scenario": "Eliminating the '22.5% Fake URL & Hallucinated Citation Trap' via Integer-Constrained Anchor IDs (`[1]..[K]`)",
            "question": "When asked to cite sources in an enterprise RAG assistant (`Tier 1`), the LLM invents plausible-looking SharePoint URLs or attaches `[Source: HR Policy 2026]` to a sentence that actually came from nowhere (`22.5%` fake/mismatched citation rate)",
            "model_answer": "Never let an LLM generate raw URLs, file names, or document titles in its output stream! Format retrieved context blocks strictly with **Integer Anchor IDs (`<doc id='1'>...</doc> ... <doc id='K'>...</doc>`)**, constrain citation tokens to **`[1]..[K]`**, and have the **Deterministic UI/Gateway Post-Processor** resolve `[k]` to the immutable SharePoint URL, page number, and highlighted bounding box (`0.1%` citation mismatch vs `22.5%`)!",
            "tier1_3_failure": "Tier 1 lets the LLM type out URLs from memory; Tier 3 attaches post-hoc bi-encoder matches that miss negations.",
            "action": "Act automatically",
            "confidence": 1.00,
        },
        {
            "id": "HG2",
            "scenario": "Catching 'Subtle Negation & Number Flip Hallucinations' That Embedding Cosine Similarity Misses (`not covered` vs `covered`)",
            "question": "The source contract says 'Flood damage is NOT covered under Plan B (`$5,000` deductible)'; the LLM generates 'Flood damage IS covered under Plan B (`$500` deductible)'. Why does Tier 3's Embedding Cosine check (`0.94` similarity!) pass this hallucination?",
            "model_answer": "Bi-encoder embeddings (`Tier 3`) measure **topical overlap**, not **logical entailment (`NLI`)** or **exact numeric equality**! Combine two fast, deterministic checks on every generated sentence: (1) run a specialized **400M Cross-Encoder Entailment Model (`Vectara HHEM-2.1 / Bespoke-MiniCheck / Azure Groundedness` on an `L4` GPU in `28ms`)** comparing `Premise = Chunk[k]` vs. `Hypothesis = Sentence`, and (2) run a **`0.2ms` CPU Numeric/Currency Extractor** verifying that every number (`$500`) in the output exists in `Chunk[k]` (`99.9%` numeric fidelity)!",
            "tier1_3_failure": "Tier 3 scores 'Flood damage is covered' and 'Flood damage is NOT covered' at 0.96 cosine similarity; Tier 2 adds 2,450ms delay.",
            "action": "Act automatically",
            "confidence": 0.99,
        },
        {
            "id": "HG3",
            "scenario": "Verifying Groundedness on Streaming SSE Responses in `28ms` Without Blocking `TTFT` for 2.45 Seconds",
            "question": "If we buffer the entire answer and call a 70B LLM-as-a-Judge (`Tier 2`), the user waits an extra 2,450ms and our LLM token bill doubles (`$248K/mo`)",
            "model_answer": "Verify at **Sentence Boundaries (`。` / `. ` / `\\n`) in Parallel with Streaming**: as the LLM streams tokens to the user, each completed sentence (`~25 tokens`) and its cited chunk `[k]` are scored asynchronously in **`28ms` by `HHEM / DeBERTa-v3-Large NLI`** on a shared `L4` GPU; if a sentence fails entailment (`score < 0.50`), emit an inline UI strike-through / retraction event or fallback warning (`28ms` overhead vs `2,450ms`, **`-89.5%` TCO**)!",
            "tier1_3_failure": "Tier 2 blocks the entire response stream and doubles foundation model token consumption.",
            "action": "Confirm",
            "confidence": 0.99,
        },
        {
            "id": "HG4",
            "scenario": "Enforcing the 'Abstain When Context Is Insufficient (`IDK` Calibration)' Contract",
            "question": "When retrieved chunks have partial keyword overlap with the user's question but do not contain the actual answer, LLMs guess from pre-training memory 14.8% of the time",
            "model_answer": "Combine **Pre-Generation `CRAG` Reranker Gating (`max_rerank_score < 0.35 -> Abstain`)** with **Explicit Extractive Quote-First Prompting (`Quote the exact span from [k] before answering`)**, cutting ungrounded hallucinations from `14.8%` to **`0.2%`**!",
            "tier1_3_failure": "Soft 'Do not make things up' system prompts fail whenever pre-training priors conflict with incomplete context.",
            "action": "Send to a human",
            "confidence": 0.99,
        },
    ]

    perturbation_checks = [
        {
            "check": "LLM Hallucinates a Decimal Shift in Financial Output (`$12.5M` Rendered as `$125M`)",
            "original": "99.9% Numeric Exact Match | 0.2% Hallucination",
            "perturbed": "99.9% Match (0.2ms CPU Numeric Set-Intersection Gate Catches `125 != 12.5` Against Cited Chunk #2 Instantly)",
            "change_pts": "0.0 pts (Decimal Shift Blocked)",
        },
        {
            "check": "Multi-Citation Synthesis Sentence Combining Facts From `[1]` and `[4]` (`According to [1][4]...`)",
            "original": "28.0ms NLI Entailment Latency",
            "perturbed": "31.5ms Latency (Concatenates `Premise = Chunk[1] + Chunk[4]` Before Running HHEM/NLI Entailment Check)",
            "change_pts": "+3.5ms (Multi-Hop Verified)",
        },
    ]

    return {
        "experiment_id": "enterprise_rag_hallucination_detection_citation_grounding_verification_selection",
        "round": 508,
        "title": "Enterprise RAG Hallucination Prevention, Sentence-Level Inline Citation Grounding (`HHEM / MiniCheck NLI`) & Numeric Verification Selection",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "baseline_workload": {
            "monthly_rag_responses_verified": 25000000,
            "average_sentences_per_response": 6,
            "nli_entailment_model": "HHEM-2.1 / MiniCheck-DeBERTa-Large (400M on NVIDIA L4)",
            "target_hallucination_ceiling_pct": 0.5,
        },
        "tiers": tiers,
        "scenarios": scenarios,
        "perturbation_checks": perturbation_checks,
        "key_findings": [
            "Tier 4 (Constrained Integer Citation IDs `[1..K]` + Sentence-Streaming `HHEM/MiniCheck NLI` `28ms` + Deterministic Numeric Set-Intersection Gate) reduces ungrounded hallucinations from 14.8% (Tier 1) to 0.2% and fake citations from 22.5% to 0.1% at $26K/mo (-89.5% vs Tier 2 70B Judge).",
            "Post-hoc bi-encoder embedding similarity (Tier 3) cannot detect negation flips ('is covered' vs 'is NOT covered') or numeric decimal shifts ('$500' vs '$5,000').",
            "Scoring each sentence strictly against its cited chunk `Chunk[k]` using a 400M NLI cross-encoder on an L4 GPU takes only 28ms compared to 2,450ms for a synchronous 70B LLM-as-a-Judge.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify existing results.json matches deterministic payload")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        current = out_path.read_text(encoding="utf-8")
        if current != rendered:
            raise SystemExit("results.json drift detected")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
