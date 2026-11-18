# Enterprise Browser-Use & Computer-Use GUI Agent Architecture Evaluator 

Deterministic analytical evaluator comparing four enterprise GUI / Browser automation architectures across `200,000` 15-step ERP/CRM/Expense workflows per month:

1. **Tier 1 (`tier1_pure_vision_screenshot_pixel_click`)**: Pure Vision Screenshot Pixel-Click Loop (`71.5%` workflow completion rate, `51.0 s` avg duration, `38,000` tokens/workflow, `$96,000/mo`).
2. **Tier 2 (`tier2_raw_html_dom_dump`)**: Raw HTML DOM Dump (`64.0%` completion rate, `32.5 s` avg duration, `145,000` tokens/workflow, `$68,000/mo`).
3. **Tier 3 (`tier3_distilled_axtree_plus_set_of_marks`)**: Distilled Accessibility Tree (`AXTree`) + Set-of-Marks (`SoM`) Visual Grounding (`88.5%` completion rate, `14.2 s` avg duration, `16,500` tokens/workflow, `$16,400/mo`).
4. **Tier 4 (`tier4_mcp_playwright_replay_self_healing_hitl`)**: API/MCP-First + Deterministic Playwright Selector Replay (`85%` cached zero-token execution, invoking `AXTree + SoM` LLM self-healing only when UI selectors drift) + Isolated Sandbox & Human Sensitive-Step Handoff (`98.2%` completion rate [`+26.7 pts` vs. Tier 1], `4.6 s` avg duration [`11.1x` faster than Tier 1], `$5,200/mo` [`-94.6%` cost savings vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_browser_computer_use_gui_agent_selection/run.py
python experiments/enterprise_browser_computer_use_gui_agent_selection/run.py --check
```
