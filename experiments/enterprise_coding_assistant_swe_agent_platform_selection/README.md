# Enterprise AI Coding Assistant & Autonomous `SWE Agent` Platform Architecture Selection (`R486`)

Deterministic evaluation suite comparing four enterprise AI software engineering (`Coding Assistant` + `Autonomous SWE Agent`) architectures across `2,000` developers and `4,500` internal repositories (`15 Million` LOC monorepo):

1. **Tier 1**: Un-Indexed Generic Web Chat Copy-Paste (`18.5%` multi-file PR resolution, `38.0%` hallucinated internal APIs, `16.4%` vulnerability escape, `$190,000/mo` rework + license TCO)
2. **Tier 2**: Single-File IDE `Fill-in-the-Middle (FIM)` Autocomplete Only (`34.0%` multi-file resolution, `145ms` FIM latency, `21.5%` API hallucination, `$125,000/mo` TCO)
3. **Tier 3**: Frontier Model on Raw File Grep Without AST/SCIP Index or SAST Sandbox (`62.4%` resolution, `890ms` FIM lag, `7.2%` vulnerability escape, `$148,000/mo` TCO)
4. **Tier 4**: Dual-Engine SWE Platform (`7B/14B FIM` @ `125ms` + `Tree-sitter AST / SCIP` Repo-Indexed Frontier Agent + Ephemeral `LSP + Test + CodeQL/Semgrep SAST` Self-Healing Sandbox) (`84.6%` multi-file PR resolution, `125ms` FIM P95 latency, `0.0%` vulnerability escape, `0.6%` API hallucination, `$52,000/mo` net TCO)

## Usage

```bash
python experiments/enterprise_coding_assistant_swe_agent_platform_selection/run.py
python experiments/enterprise_coding_assistant_swe_agent_platform_selection/run.py --check
python -m unittest tests/test_enterprise_coding_assistant_swe_agent_platform_selection.py
```
