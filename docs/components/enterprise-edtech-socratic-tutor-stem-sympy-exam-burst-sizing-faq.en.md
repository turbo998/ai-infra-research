# Enterprise EdTech, Socratic AI Tutor, SymPy STEM Step Verifier & Exam-Peak Elastic Burst Sizing Guide

[![Back to Catalog](https://img.shields.io/badge/Back_to-Catalog-blue)](../full-catalog-en.md)
[![中文版](https://img.shields.io/badge/中文-简体中文版-blue)](./enterprise-edtech-socratic-tutor-stem-sympy-exam-burst-sizing-faq.md)
[![Interactive H5 Lab](https://img.shields.io/badge/H5-Interactive_Study_Lab-10b981)](../h5/edtech-socratic-tutor-stem-sympy-exam-burst-sizer.html)
[![7-Card Report](https://img.shields.io/badge/7--Card-Benchmark_Deck-purple)](../reports/enterprise-edtech-socratic-tutor-stem-sympy-exam-burst-sizing-cards.html)
[![Interactive Sizer](https://img.shields.io/badge/Sizer-Capacity_Calculator-orange)](../reports/enterprise-edtech-socratic-tutor-stem-sympy-exam-burst-sizing-sizer.html)

![EdTech AI Socratic Guardrail, SymPy Verifier & Exam Elastic Burst Overview](../../assets/svg/enterprise-edtech-socratic-tutor-stem-sympy-exam-burst-sizing-faq-summary.svg)

---

## 1. Executive Summary (3-Minute Read)

> **Bottom Line for Architects**: Deploying raw conversational LLMs in K-12, university, or corporate certification platforms fails on three fronts: **(1) Students jailbreak prompt-only tutors into giving away final homework answers `34.0%` of the time; (2) Pure VLMs hallucinate on handwritten algebra/calculus steps (`11.8%` step-grading error rate on sign flips or valid shortcut steps); and (3) Provisioning dedicated GPUs for the `19:00–22:00` homework spike and final-exam week (`6–8×` average load) leaves GPUs `78%` idle the rest of the day**. The production reference architecture combines **`Qwen2.5-VL-7B` Line-by-Line LaTeX Extraction + `SymPy` Computer Algebra System (`4.5 ms` exact step-equivalence verification, `99.4%` step accuracy) + a 3-Stage Socratic Finite State Machine with Streaming Final-Answer Redaction (`0.0%` answer leakage) + Curriculum Prefix Caching (`84%` hit rate) + a 3-Hour Evening Spot/Serverless Burst Pool (`-79.4%` monthly TCO)**.

| Subsystem | Naive Anti-Pattern | Production Neuro-Symbolic EdTech Stack | Quantified Impact |
| :--- | :--- | :--- | :--- |
| **1. Handwritten STEM Step Grading** | Send photo directly to a 72B VLM (`11.8%` false step praise/penalty on minus signs) | **`Qwen2.5-VL-7B` LaTeX transcription ($E_1 \dots E_n$) + CPU `SymPy` `simplify(E_i - E_{i-1}) == 0`** (`4.5 ms`) | **`99.4%` step accuracy**, **`-94.2%`** cost per problem |
| **2. Socratic Tutoring (Anti-Cheating)** | System prompt "do not give the answer" (`34.0%` direct leakage under student persuasion) | **3-Stage Socratic FSM (`Concept Hint` $\to$ `Sub-Goal` $\to$ `Worked Analog`) + Streaming Answer Regex/Value Blocker** | **`0.0%` answer leakage**, **`+0.42 SD`** learning gain |
| **3. Shared Textbook Context** | Re-prefill 4,000 curriculum & rubric tokens on every student turn | **Canonical `(Chapter_ID, Problem_ID)` prefix + `SGLang RadixAttention` cross-student KV sharing** | **`84.0%` prefix cache hit**, **`190 ms` P95 TTFT** |
| **4. 19:00–22:00 Homework & Exam Surge** | 24×7 dedicated `64× H100` sized for peak `800 QPS` (`$147,200/mo`, `22.4%` avg utilization) | **Base `8× L40S + 4× H200` (daytime + nightly 1.2M essay grading) + 3h evening Spot/Serverless burst** | **`$30,300/mo` (`-79.4%` TCO)**, **`86.5%`** GPU utilization |

---

## 2. Core Architectural Q&A

### Q1: Why do pure VLMs fail at grading handwritten math/physics steps, and how does `VLM + SymPy` solve it?
- **Token-Probability Hallucination**: When a student writes 6 lines of algebraic manipulation and drops a minus sign on Line 2, or takes a valid two-step shortcut, a pure VLM evaluates plausibility rather than symbolic equivalence—producing an **`11.8%` step-level error rate**.
- **Decoupling Perception from Symbolic Verification**:
  1. **`Qwen2.5-VL-7B` (`L40S`)** transcribes each handwritten line into structured LaTeX equations `[E_1, E_2, ..., E_n]` without judging correctness;
  2. **`SymPy` (`4.5 ms` on CPU)** parses each line and checks symbolic equivalence `Equivalent(E_i, E_{i-1})`. The exact line where equivalence breaks is pinpointed deterministically;
  3. **`Qwen2.5-32B` Socratic Tutor** receives the verified diff (`Step 2 -> Step 3 sign flip on -3`) and asks a targeted guiding question.

### Q2: How do we guarantee `0.0%` direct answer leakage when students try to trick the AI?
- Before generating any response, `SymPy` computes the ground-truth solution $S^*$ and registers its numerical/symbolic forms in a **streaming output guardrail**. The dialogue is governed by a **3-Stage Socratic FSM** that only advances when the student submits a valid intermediate step; even at the final fallback stage, the tutor demonstrates a **worked analog problem with different numbers** rather than solving the student's assigned problem.

### Q3: How do we size GPUs for an `800 QPS` evening homework spike without wasting 78% of capacity all day?
- Provision a **base cluster (`8× L40S + 4× H200`, `$19.6K/mo`)** for daytime traffic (`~180 QPS`), auto-scale a **3-hour Spot + Serverless burst pool (`19:00–22:00`, `$10.7K/mo`)** pre-warmed at `18:50` via `Run:ai Model Streamer`, and run **1.2M asynchronous essay gradings and personalized quiz generation** on the base `4× H200` between `23:00–07:00`.

---

## 3. Pre-Production Checklist

- [ ] **Low-Confidence OCR Confirmation**: Highlight handwritten symbols with VLM token confidence `< 0.85` for student confirmation before running `SymPy`.
- [ ] **500-Prompt Student Jailbreak Gate**: Validate `0.0%` final-answer leakage against adversarial "pretend I already solved it" student prompts.
- [ ] **18:50 Scheduled Pre-Warming**: Trigger Spot GPU scale-out and curriculum prefix cache warming 10 minutes before the `19:00` homework spike.
- [ ] **Student PII Cropping (COPPA / FERPA)**: Strip student names, school headers, and faces at the edge before storing homework images.
