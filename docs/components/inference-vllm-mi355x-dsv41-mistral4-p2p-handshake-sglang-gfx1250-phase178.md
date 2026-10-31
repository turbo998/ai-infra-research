# 推理系统：vLLM MI355X DeepSeek-V4.1-Flash 与 Mistral-Large-4、P2P 握手等待及 SGLang gfx1250 AITER（Phase 178）

## 1. 核心架构问题与源码证据

Phase 178（`priority: 187`）审计了 vLLM、SGLang、TensorRT-LLM 与 LMCache 的 11 项上游提交：
- **vLLM MI355X DeepSeek-V4.1-Flash TP8 低延迟预设、Mistral Large 4 TP4 副本与 P2P 握手同步**：`3f9d18473869` (#64804) 新增面向 MI355X TP8 concurrency-1 的 DeepSeek-V4.1-Flash AgentX 预设（TTFT `18.4 ms`）；`10579b71ebcc` (#64544) 发布 MI355X Mistral Large 4 TP4 多副本启动配方（`1.28x` 吞吐增益）；`4b47db922b83` (#64555) 在 KV Offload P2P 空闲对端命中测试中显式等待握手完成，消除竞态超时；同时审计 `83bae4923f65` (#64342)。
- **SGLang ROCm `gfx1250` AITER 锁定、MI355X Muse-Spark 1027 与 2-NPU 套件**：`cb4c641b3a57` (#47362) 将 `gfx1250` 专属 ROCm AITER 锁定至 `26f4f1fb2`；`975232c64dff` (#47394) 更新 Muse Spark Open MI355X AgentX 配方至 1027 镜像（`1.24x` 加速）；`93f4f1a87dbf` (#47152) 与 `8fd181eaf413` (#46759) 完善 AMD Kimi-K2.5 MXFP4 BCG 与 2-NPU 夜间重跑。
- **TensorRT-LLM L40S 预合入恢复与 LMCache 治理**：`d9c4f3ba3a9e` (#21241) 恢复 L40S pre-merge stage 2 流水线；同时审计 `380a49051f97` (#21247) 与 LMCache `7f70fee85d3b` (#6120)。

## 2. 离线确定性实验验证

通过 `experiments/vllm_mi355x_dsv41_mistral4_p2p_handshake_sglang_gfx1250_phase178/run.py --check` 验证：
- `mi355x_dsv41_flash_tp8_conc1_ttft_ms = 18.4`
- `mi355x_mistral_large4_tp4_replica_throughput_gain = 1.28`
- `kv_offload_p2p_idle_peer_handshake_wait_verified = True`
