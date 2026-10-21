# SGLang Host Extend Completion Marker & TBO Inheritance, `SO_REUSEPORT` Worker Listeners, Rust Response-Ring Health Probe Ack, TensorRT-LLM KV V2 Cleanup & Multimodal Responses API, and vLLM Shared Stream Processor

> **Phase 55 (`priority: 64`) · First-Hand Commit-Level Deep Dive**
> - `sgl-project/sglang@a3d00eff58b6c22ea7539f882cc10c537634200b` (`#45487`)
> - `sgl-project/sglang@7e5256d1ba3aaa628bb8b756fafb904a76a1123a` (`#45482`)
> - `sgl-project/sglang@ca72dad953171ea4f580d89f70f999fc89501067` (`#45475`)
> - `sgl-project/sglang@68623cae9256ad8ee00269c8c1d52e4c2915daa5` (`#45670`)
> - `NVIDIA/TensorRT-LLM@1c7fadcfbe50911d784658e902d1bd003d89fdd7` (`#20253`) & `271ed9f22ab85e5ce2b45e326f5b41cbb249bd93` (`#20231`)
> - `vllm-project/vllm@6ae1051bd9120c9e2974a8ad502b4c9716db6b9f` (`#62741`) & `5ea2c7ff1e64b36b45bf6bd502367c85633b2d8c` (`#62706`)

---

## 1. SGLang Host `contains_last_prefill_chunk` & Mamba Ping-Pong Track Gather (`#45487`, `#45670`)

- **Host Prompt-Completion Tracking (`#45487`)**: Computes `contains_last_prefill_chunk` on the host in `ForwardBatch.init_new` by excluding running `decoding_reqs` and checking whether any extend request satisfies `req.extend_end == len(req.full_untruncated_fill_ids)`, propagating the flag across Two-Batch Overlap (`filter_batch` children).
- **Mamba Ping-Pong Track Gathering (`#45670`)**: Uses `sb._gather_mamba_track_indices(ping_pong_buffers, req_pool_indices, track_positions)` to index each request's 2-slot ping-pong buffer via `track_position` (`0` or `1`).

---

## 2. SGLang `SO_REUSEPORT` Listeners & Rust Health Probe Acknowledgement (`#45482`, `#45475`)

- **Per-Worker `SO_REUSEPORT` (`#45482`)**: Bumps `uvicorn>=0.51` and adds `--enable-http-worker-reuseport` so each Uvicorn HTTP/tokenizer worker binds its own `SO_REUSEPORT` socket.
- **Rust Response-Ring Health Ack (`#45475`)**: When `self.rust_server is not None`, `maybe_send_health_check_signal()` clears `return_health_check_ipcs` and calls `self.rust_server.push_health_ack()` (`frame_health_ack()`), answering all skipped probes with a single frame on the Rust response ring instead of writing to an unconsumed ZMQ socket.

---

## 3. TensorRT-LLM KV V2 / Responses Multimodal API & vLLM Frontend Stream Convergence (`#20253`, `#20231`, `#62741`, `#62706`)

- **TRT-LLM Warmup Cleanup & Multimodal Responses API (`#20253`, `#20231`)**: Removes the extra dummy max-draft speculative CUDA Graph capture workaround (`original_max_draft_len`) and unpacks `(mm_data, mm_embeddings)` in `responses_utils.py` while forwarding `multimodal_server_config` for `input_image` parts.
- **vLLM Rust `max_tool_calls` & Shared `ChatStreamProcessor` (`#62741`, `#62706`)**: Adds `--exclude-tools-when-tool-choice-none`, replaces `parallel_tool_calls: bool` with `max_tool_calls: Option<usize>` in `StructuredEventState`, and unifies per-choice stream delta processing via `ChatStreamProcessor`.

---

## 4. Deterministic Verification

Run `python experiments/sglang-extend-host-prompt-reuseport-rust-probe-trtllm-v2-vllm-stream/run.py --check` to verify all invariants.
