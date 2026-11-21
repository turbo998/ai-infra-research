# Enterprise IDE Code Completion (FIM) & Coding Agent Fast-Apply Selection Experiment (`7B/14B FIM + Speculative Fast-Apply + Tree-Sitter AST Merkle Index` vs. `Shared 70B Pool`)

Reproducible latency, acceptance rate, and TCO model comparing 4 enterprise AI coding platform architectures across `10,000` developers for real-time `Tab` Fill-in-the-Middle (`FIM`) completion and `4,800`-token (`800`-line) full-file Fast-Apply rewrites.

## Run
```bash
python experiments/ide_code_completion_fim_coding_agent_speculative_edits_cursor_rag_selection/run.py
python experiments/ide_code_completion_fim_coding_agent_speculative_edits_cursor_rag_selection/run.py --check
```
