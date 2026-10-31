# SGLang Joint-LRU Unified KV Eviction & Rust DP-Abort, vLLM NIXL 1.5.2 & TRT-LLM Owning Fragment (Phase 161)

Deterministic standard-library analytical model (`own-offline`) for Phase 161 (`priority: 170`) Track 2:
- **SGLang**: `a65c9c596905` (`joint_lru` shared eviction across Full + SWA KV pools), `60b183d5c3ce` (Rust server attention-DP abort fan-out + listener map), `2410a09c6e1a` (DeepSeek V4 device pool buffer descriptors).
- **vLLM / TRT-LLM / DeepEP**: `78161c28da3b`, `886b845ef14c`, `06f36c122297`, `2d1a7d170eb8`, `3302a317466d`, `f6aa38fe06cd`, `93eb6eb23812`.
