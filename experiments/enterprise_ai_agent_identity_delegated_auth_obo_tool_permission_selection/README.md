# Enterprise AI Agent Identity, Delegated `OBO` Authorization & `MCP` Tool-Permission Boundary Selection (`R469`)

Deterministic evaluation suite comparing four enterprise AI Agent identity, `MCP` tool authentication, and authorization boundary architectures across `15,000` enterprise users (`42` connected MCP/API tool servers, `650,000` daily tool calls):

1. **Tier 1**: Shared 'God-Mode' Service Principal / Static Master API Key (`100.0%` confused-deputy escalation escape, `24.8%` cross-user data leak, `0.0%` per-user audit attribution, `$175,000/mo` risk/ops TCO)
2. **Tier 2**: System-Prompt Role Rules + Unscoped User Bearer Token Pass-Through (`42.5%` prompt-injection escalation escape, `11.4%` cross-user leak, `$128,000/mo` TCO)
3. **Tier 3**: Delegated On-Behalf-Of (`OBO`) Token Exchange Only (`4.2%` escalation escape on over-scoped user tokens, `6.4%` write blast radius, `$64,000/mo` TCO)
4. **Tier 4**: Distinct `Entra Agent ID` + Scoped `OBO` Token Exchange (`User ∩ Agent`) + `OPA/Cedar` Argument Gate + Step-Up Consent (`0.0%` confused-deputy escape, `0.0%` cross-user leak, `0.0%` destructive blast radius, `100.0%` dual-actor audit attribution, `$32,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_ai_agent_identity_delegated_auth_obo_tool_permission_selection/run.py
python experiments/enterprise_ai_agent_identity_delegated_auth_obo_tool_permission_selection/run.py --check
python -m unittest tests/test_enterprise_ai_agent_identity_delegated_auth_obo_tool_permission_selection.py
```
