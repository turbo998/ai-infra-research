# Enterprise Omnichannel Customer Support Agent, Deterministic Refund StateGraph & Human Warm-Handoff Evaluator

Deterministic evaluator comparing four enterprise omnichannel customer support agent architectures across 1.5M monthly conversations, 120,000 monthly refund/exchange requests, and 8,500 active/historical policy documents:
- **Tier 1**: Legacy Rigid Keyword Decision-Tree Bot + Cold Human Queue Transfer (`34.0%` FCR, `0.2%` policy violation rate, `195.0 s` repeat explanation time, `$280,000/mo`)
- **Tier 2**: Unconstrained Prompt-Driven LLM Directly Calling Order & Refund APIs (`68.5%` FCR, `11.8%` unauthorized refund / policy violation rate, `140.0 s` repeat time, `$345,000/mo`)
- **Tier 3**: Standard Vector Knowledge-Base RAG + Read-Only Order Lookup + Cold Escalation (`56.2%` FCR, `4.5%` stale-policy violation rate, `165.0 s` repeat time, `$142,000/mo`)
- **Tier 4**: Effective-Date KB RAG + Deterministic `LangGraph` Refund Policy Gate + Structured Human Warm Handoff (`84.6%` FCR, `0.0%` unauthorized refund violation rate, `18.0 s` warm handoff sync, `$54,000/mo`)

## Usage
```bash
python experiments/enterprise_customer_support_omnichannel_agent_stategraph_handoff_selection/run.py
python experiments/enterprise_customer_support_omnichannel_agent_stategraph_handoff_selection/run.py --check
```
