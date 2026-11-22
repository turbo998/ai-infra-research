# Enterprise `MCP + A2A` Multi-Agent Context Explosion, Top-K Tool-Schema Filtering & KV Sizer (`#498`)

Reproducible analytical model comparing 4 multi-agent orchestration architectures across `200 enterprise MCP tools` and `5 A2A sub-agents` (`20 tasks/sec` enterprise workload):
- **Tier 1 (`Naive 200-MCP-Tool Dump + Full A2A History Copy`)**: Injects `28,000` tool-schema tokens every turn (`245,000 tokens/task`), drops tool accuracy to `61.5%`, stalls `P95 E2EL` at `19.4s`, and requires **`48x H200 GPUs`** (`$480 / 1,000 tasks`).
- **Tier 2 (`Static 200-MCP-Tool Prefix + Full A2A History Copy`)**: `210,000 tokens/task`, `64.0%` tool accuracy, `28x H200 GPUs`.
- **Tier 3 (`Top-8 Semantic MCP Filter + Full A2A History Copy`)**: `84,000 tokens/task`, `93.8%` tool accuracy, `14x H200 GPUs`.
- **Tier 4 (`Top-8 Semantic MCP Filter + Scoped A2A Artifact Handoff + Prefix Pinning`)**: Cuts tokens per task by **`89.2%` (`26,500` vs. `245,000`)**, raises tool accuracy to **`94.6%`**, speeds up `P95 E2EL` **`6.26x` (`3.1s` vs. `19.4s`)**, and runs `20 tasks/sec` on just **`6x H200 GPUs` (`-87.5%` GPUs)**.

## Run
```bash
python experiments/enterprise_mcp_a2a_multi_agent_tool_schema_bloat_kv_sizing_selection/run.py
```
